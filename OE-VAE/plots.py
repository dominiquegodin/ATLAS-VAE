import numpy             as np
import multiprocessing   as mp
import matplotlib.pyplot as plt
import sys, warnings
from matplotlib    import pylab, ticker
from sklearn       import metrics
from scipy.spatial import distance
from functools     import partial
from utils         import loss_function, inverse_scaler, get_4v, bump_hunter


def plot_results(y_true, X_true, X_pred, sample, n_dims, model, metrics, wp_metric, output_dir):
    print('PLOTTING RESULTS:')
    manager = mp.Manager(); X_losses = manager.dict()
    X_true_dict = {metric:X_true for metric in metrics if metric!='Latent'}
    if False:
        X_true_dict['Inputs'] = sample['constituents']
    arguments = [(X_true_dict[metric], X_pred, n_dims, metric, X_losses) for metric in metrics if metric!='Latent']
    processes = [mp.Process(target=loss_function, args=arg) for arg in arguments]
    for job in processes: job.start()
    for job in processes: job.join()
    if 'Latent' in metrics: #Adding latent space KLD metric and loss
        model(X_true)
        X_losses['Latent'] = model.losses[0].numpy()
    #max_significance(y_true, X_losses[wp_metric], sample, output_dir); sys.exit()
    arguments  = [(y_true, X_losses, sample['M'], sample['weights'], metrics, wp_metric, output_dir)]
    processes  = [mp.Process(target=mass_sculpting, args=arg) for arg in arguments]
    processes += [mp.Process(target=ROC_curves, args=(y_true,X_losses,sample['weights'],metrics,output_dir,'gain' ))]
    processes += [mp.Process(target=ROC_curves, args=(y_true,X_losses,sample['weights'],metrics,output_dir,'sigma'))]
    #arguments  = [(y_true, X_losses[metric], sample['weights'], metric, output_dir) for metric in metrics]
    #processes += [mp.Process(target=loss_distributions, args=arg) for arg in arguments]
    #processes += [mp.Process(target=pt_reconstruction, args=(X_true, X_pred, y_true, sample['weights'], output_dir))]
    for job in processes: job.start()
    for job in processes: job.join()


def max_significance(y_true, X_loss, sample, output_dir, n_cuts=20):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, X_loss, pos_label=0, sample_weight=sample['weights'])
    step = max(1,len(fpr)//n_cuts)
    fpr, tpr, losses = fpr[::step], tpr[::step], thresholds[::step]
    sample = {key:sample[key] for key in ['JZW','M','weights']}
    global get_sigma
    def get_sigma(sample, X_loss, losses, tpr, idx):
        cut_sample = {key:sample[key][X_loss>losses[idx]] for key in ['JZW','M','weights']}
        try:
            return 100*tpr[idx], bump_hunter(cut_sample, make_histo=False)
        except:
            return None, None
    with mp.Pool() as pool:
        pool_list = pool.map(partial(get_sigma, sample, X_loss, losses, tpr), np.arange(len(losses)))
    sig_eff, sigma = zip(*list(pool_list))
    sig_eff, sigma = [n for n in sig_eff if n!=None], [n for n in sigma if n!=None]
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    plt.plot(sig_eff, sigma, label='', color='tab:blue', lw=2, zorder=1)
    max_val = sigma[-1]
    plt.text(100.4, max_val, format(max_val,'.1f'), {'color':'gray', 'fontsize':12}, va="center", ha="left")
    max_val = np.max(sigma)
    max_eff = sig_eff[np.argmax(sigma)]
    axes.axhline(max_val, xmin=max_eff/100, xmax=1, ls='--', linewidth=1., color='gray')
    plt.text(100.4, max_val, format(max_val,'.1f'), {'color':'gray', 'fontsize':12}, va="center", ha="left")
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.tick_params(axis='both', which='major', labelsize=14)
    pylab.xlim(0, 100)
    #pylab.ylim(np.floor(np.min(sigma)), np.ceil(np.max(sigma)))
    pylab.ylim(0, 5)
    plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.ylabel('Significance', fontsize=25)
    file_name = output_dir+'/'+'max_sigma.png'
    print('Saving maximum significance to:', file_name); plt.savefig(file_name)


def loss_distributions(y_true, X_loss, weights, metric, output_dir, n_bins=100):
    min_dict = {'MSE':0  , 'MAE':0  , 'X-S'   :0  , 'JSD'   :0,   'EMD':0,
                'KSD':0  , 'KLD':0  , 'Inputs':0  , 'Latent':0}
    max_dict = {'MSE':0.001, 'MAE':0.05, 'X-S'   :0.5 , 'JSD'   :1, 'EMD':1,
                'KSD':1.0  , 'KLD':0.5 , 'Inputs':0.02, 'Latent':2e3}
    min_loss  = min_dict[metric]
    max_loss  = min(max_dict[metric], np.max(X_loss))
    bin_width = (max_loss-min_loss)/n_bins
    bins      = list(np.arange(min_loss, max_loss, bin_width)) + [max_loss]
    labels    = [r'$t\bar{t}$ jets', 'QCD jets']; colors = ['tab:orange', 'tab:blue']
    if np.any(weights) == None: weights = np.array(len(y_true)*[1.])
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    for n in set(y_true):
        hist_weights  = weights[y_true==n]
        hist_weights *= 100/np.sum(hist_weights)/bin_width
        pylab.hist(X_loss[y_true==n], bins, histtype='step', weights=hist_weights, label=labels[n],
                   log=True if metric in ['DNN', 'FCN'] else False, color=colors[n], lw=2)
    pylab.xlim(min_dict[metric], max_loss)
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.tick_params(axis='both', which='major', labelsize=14)
    label = 'KLD latent loss' if metric=='Latent' else metric+' reconstruction loss'
    plt.xlabel(label, fontsize=24)
    plt.ylabel('Distribution density (%)', fontsize=24)
    plt.legend(loc='upper right', fontsize=18)
    file_name = output_dir+'/'+metric+'_loss.png'
    print('Saving loss distributions to:', file_name); plt.savefig(file_name)


def mass_distances(y_true, X_losses, X_mass, weights, metric, distances_dict, n_cuts=100):
    X_loss = X_losses[metric]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, X_loss, pos_label=0, sample_weight=weights)
    step = max(1,len(fpr)//n_cuts)
    fpr, tpr, losses = fpr[::step], tpr[::step], thresholds[::step]
    bkg_loss    =  X_loss[y_true==1]
    bkg_mass    =  X_mass[y_true==1]
    bkg_weights = weights[y_true==1]
    KSD = []; JSD = []; sig_eff = []; bkg_eff = []
    P = np.histogram(bkg_mass, bins=100, range=(0,500), weights=bkg_weights)[0]
    for n in range(len(losses)):
        bkg_mass_cut    = bkg_mass   [bkg_loss>losses[n]]
        bkg_weights_cut = bkg_weights[bkg_loss>losses[n]]
        if len(bkg_mass_cut) != 0:
            Q = np.histogram(bkg_mass_cut, bins=100, range=(0,500), weights=bkg_weights_cut)[0]
            KSD += [ KS_distance(bkg_mass, bkg_mass_cut, bkg_weights, bkg_weights_cut) ]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                JSD += [ distance.jensenshannon(P, Q) ]
            sig_eff += [ 100*tpr[n] ]
            bkg_eff += [ 100*fpr[n] ]
    distances_dict[metric] = KSD, JSD, sig_eff, bkg_eff


def mass_sculpting(y_true, X_losses, X_mass, weights, metrics_list, wp_metric, output_dir):
    def make_plot(distance_type, wp_metric):
        for metric in metrics_list:
            label = str(metric)+(' metric' if len(metrics_list)==1 else '')
            if distance_type == 'KSD': dist, _, sig_eff, bkg_eff = distances_dict[metric]
            if distance_type == 'JSD': _, dist, sig_eff, bkg_eff = distances_dict[metric]
            plt.plot(sig_eff, dist, label=label, color=colors_dict[metric], lw=2, zorder=1)
            if metric == wp_metric:
                P = []; labels = []
                for bkg_rej,marker in zip([95,90,80,50], ['o','^','s','D']):
                    idx = (np.abs(100-np.array(bkg_eff) - bkg_rej)).argmin()
                    labels += ['$\epsilon_{\operatorname{bkg}}$: '+format(100-bkg_rej,'>2d')+'%']
                    color = colors_dict[metric]
                    P += [plt.scatter(sig_eff[idx], dist[idx], s=40, marker=marker, color=color, zorder=10)]
                L = plt.legend(P, labels, loc='upper right', fontsize=14, ncol=1)
        plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
        plt.ylabel(distance_type+' (bkg)', fontsize=25)
        pylab.xlim(0, 100); pylab.ylim(0, 1.0)
        ncol = 1 if len(metrics_list)==1 else 2 if len(metrics_list)<9 else 3
        plt.legend(loc='upper center', fontsize=15, ncol=ncol); plt.gca().add_artist(L)
        pylab.grid(True); axes = plt.gca()
        axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
        axes.tick_params(axis='both', which='major', labelsize=14)
    colors_dict = {'MSE':'tab:green', 'MAE'   :'tab:brown' , 'X-S'   :'tab:purple',
                   'JSD':'tab:blue' , 'EMD'   :'tab:orange', 'KSD'   :'black'     ,
                   'KLD':'tab:red'  , 'Inputs':'gray'      , 'Latent':'tab:cyan'}
    manager   = mp.Manager(); distances_dict = manager.dict()
    arguments = [(y_true, X_losses, X_mass, weights, metric, distances_dict) for metric in metrics_list]
    processes = [mp.Process(target=mass_distances, args=arg) for arg in arguments]
    for job in processes: job.start()
    for job in processes: job.join()
    plt.figure(figsize=(11,15));
    plt.subplot(2, 1, 1); make_plot('KSD', wp_metric)
    plt.subplot(2, 1, 2); make_plot('JSD', wp_metric)
    file_name = output_dir + '/' + 'mass_sculpting.png'
    print('Saving mass sculpting     to:', file_name); plt.savefig(file_name)


def signal_gain(sample, cut_sample, output_dir, n_bins=50, m_range=(0,200)):
    def get_histo(sample):
        M       = sample['M']
        weights = sample['weights']
        JZW     = sample['JZW']
        sig_histo, mass = np.histogram(M[JZW==-1], bins=n_bins, range=m_range, weights=weights[JZW==-1])
        bkg_histo, mass = np.histogram(M[JZW>= 0], bins=n_bins, range=m_range, weights=weights[JZW>= 0])
        return mass, sig_histo, bkg_histo
    mass, sig_histo    , bkg_histo     = get_histo(sample)
    mass, sig_cut_histo, bkg_cut_histo = get_histo(cut_sample)
    mass = (mass[:-1] + mass[1:])/2
    gain = bkg_histo*sig_cut_histo / (bkg_cut_histo*sig_histo)
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    plt.plot(mass, gain, label='gain', color='tab:blue', lw=2)
    file_name = output_dir + '/' + 'signal_gain.png'
    print('Saving signal gain to:', file_name); plt.savefig(file_name)


def plot_distributions(samples, output_dir, sig_bins, bkg_bins, plot_var, sig_tag,
                       normalize=False, density=True, log=True, file_name=''):
    if sig_tag == 'W'  : tag = r'$W$'
    if sig_tag == 'top': tag = r'$t\bar{t}$'
    if plot_var == 'pt':
        #labels = {0:[sig_tag,'QCD','All'], 1:[sig_tag+'$ (weighted)','QCD (weighted)','All (cut)']}
        labels = {0:[tag+' (X-S weighted)'       ,'QCD (X-S weighted)'       ,'All'],
                  1:[tag+' (flat $p_t$ weighted)','QCD (flat $p_t$ weighted)','All (cut)']}
    if plot_var == 'M':
        labels = {0:[tag,'QCD','All'], 1:[tag+' (cut)','QCD (cut)','All (cut)']}
    colors = ['tab:orange', 'tab:blue', 'tab:brown']; alphas = [1, 0.5]
    n_bins = [sig_bins, bkg_bins, bkg_bins]
    xlabel = {'pt':'$p_t$', 'M':'$M$'}[plot_var]
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    if not isinstance(samples, list):
        samples = [samples]
    for n in [0, 1]:
        for m in range(len(samples)):
            sample = samples[m]
            condition = sample['JZW']==-1 if n==0 else sample['JZW']>= 0 if n==1 else sample['JZW']>=-2
            if not np.any(condition):
                continue
            variable = np.float32(sample[plot_var][condition])
            weights = sample['weights'][condition]
            min_val, max_val = np.min(variable), np.max(variable)
            if plot_var == 'M':
                min_val, max_val = 0, 571.3449 #400
            bin_width = (max_val - min_val) / n_bins[n]
            bins      = [min_val + k*bin_width for k in np.arange(n_bins[n]+1)]
            if normalize:
                weights *= 100/(np.sum(sample['weights'])) #100/(np.sum(weights))
            if density:
                indices  = np.searchsorted(bins, variable, side='right')
                weights /= np.take(np.diff(bins), np.minimum(indices, len(bins)-1)-1)
            pylab.hist(variable, bins, histtype='step', weights=weights, color=colors[n],
                       lw=2, log=log, alpha=alphas[m], label=labels[m][n])
    if normalize:
        if log:
            #pylab.xlim(0, 2000); pylab.ylim(1e-5, 1e0)
            pylab.xlim(0, 6000); pylab.ylim(1e-4, 1e-1)
        else:
            pylab.xlim(0, 6000); pylab.ylim(0, 100)
            pylab.yticks(np.arange(0,0.025,0.005))
            #pylab.xlim(0, 6000); pylab.ylim(0, 0.05)
            #pylab.yticks(np.arange(0,0.06,0.01))
    else:
        #pylab.xlim(0, 2000); pylab.ylim(1e-5, 1e4)
        pylab.xlim(0, 6000); pylab.ylim(1e0, 1e7)
    if plot_var == 'M':
        pylab.xlim(0, 400); pylab.ylim(1e0, 1e7)
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    if not log: axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.tick_params(axis='both', which='major', labelsize=14)
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel(xlabel+' (GeV)', fontsize=24)
    y_label = (' density' if density else '')
    if normalize:
        y_label += ' (%)'
    elif sig_tag == 'top':
        #y_label += ' ('+r'36.2 fb$^{-1}$'+')'
        y_label += ' ('+r'58.5 fb$^{-1}$'+')'
    plt.ylabel('Distribution' + y_label, fontsize=24)
    plt.legend(loc='upper right', ncol=1 if len(samples)==1 else 2, fontsize=18)
    if file_name == '':
        file_name = (plot_var if plot_var=='pt' else 'mass')+'_dist.png'
    file_name = output_dir+'/'+file_name
    print('Saving', plot_var, 'distributions to:', file_name); plt.savefig(file_name)


def pt_reconstruction(X_true, X_pred, y_true, weights, output_dir, n_bins=200):
    pt_true = get_4v(X_true)['pt']; pt_pred = get_4v(X_pred)['pt']
    if np.any(weights) == None: weights = np.array(len(y_true)*[1.])
    min_value = min(np.min(pt_true), np.min(pt_pred))
    max_value = max(np.max(pt_true), np.max(pt_pred))
    bin_width = (max_value - min_value) / n_bins
    bins      = [min_value + k*bin_width for k in np.arange(n_bins+1)]
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    labels = [r'$t\bar{t}$', 'QCD']; colors = ['tab:orange', 'tab:blue']
    for n in set(y_true):
        hist_weights  = weights[y_true==n]
        hist_weights *= 100/np.sum(hist_weights)/bin_width
        pylab.hist(pt_true[y_true==n], bins, histtype='step', weights=hist_weights,
                   label=labels[n]         , lw=2, color=colors[n], alpha=1)
        pylab.hist(pt_pred[y_true==n], bins, histtype='step', weights=hist_weights,
                   label=labels[n]+' (rec)', lw=2, color=colors[n], alpha=0.5)
    pylab.xlim(0.4, 0.5)
    pylab.ylim(0, 0.5)
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('$p_t$ (scaler space)', fontsize=24)
    #plt.xlabel('$p_t$ (GeV)', fontsize=24)
    plt.ylabel('Distribution density (%/GeV)', fontsize=24)
    plt.legend(loc='upper right', ncol=2, fontsize=18)
    file_name = output_dir+'/'+'pt_reconstruction.png'
    print('Saving pt reconstruction  to:', file_name); plt.savefig(file_name)
def quantile_reconstruction(y_true, X_true, X_pred, sample, scaler, output_dir):
    pt_reconstruction(X_true, X_pred, y_true, sample['weights'], output_dir)
    #pt_reconstruction(X_pred, X_pred, y_true, None             , output_dir)
    #X_true = inverse_scaler(X_true, scaler)
    #pt_reconstruction(sample['clusters'], X_true, y_true, None, output_dir)
    #X_pred = inverse_scaler(X_pred, scaler)
    #pt_reconstruction(sample['clusters'], X_pred, y_true, None, output_dir)


def ROC_curves(y_true, X_losses, weights, metrics_list, output_dir, ROC_type='gain', wp=[1,10]):
    colors_dict = {'MSE':'tab:green', 'MAE'   :'tab:brown' , 'X-S'   :'tab:purple',
                   'JSD':'tab:blue' , 'EMD'   :'tab:orange', 'KSD'   :'black'     ,
                   'KLD':'tab:red'  , 'Inputs':'gray'      , 'Latent':'tab:cyan'}
    metrics_dict = ROC_rates(y_true, X_losses, weights, metrics_list)
    plt.figure(figsize=(11,15))
    plt.subplot(2, 1, 1); pylab.grid(True); axes = plt.gca()
    for metric in metrics_list:
        fpr, tpr = [metrics_dict[metric][key] for key in ['fpr','tpr']]
        len_0 = np.sum(fpr==0)
        plt.text(0.98, 0.65-4*metrics_list.index(metric)/100, 'AUC: '+format(metrics.auc(fpr,tpr), '.3f'),
                 {'color':colors_dict[metric], 'fontsize':12}, va='center', ha='right', transform=axes.transAxes)
        plt.plot(100*tpr[len_0:], 1/fpr[len_0:], label=metric, lw=2, color=colors_dict[metric])
    metrics_scores = [[metrics_dict[metric][key] for key in ['fpr','tpr']] for metric in metrics_list]
    scores     = [np.max([1/fpr[np.argwhere(100*tpr >= val)[0]] for fpr,tpr in metrics_scores]) for val in wp]
    scores_idx = [np.argmax([1/fpr[np.argwhere(100*tpr >= val)[0]] for fpr,tpr in metrics_scores]) for val in wp]
    for n in np.arange(len(wp)):
        color = colors_dict[metrics_list[scores_idx[n]]]
        axes.axhline(scores[n], xmin=wp[n]/100, xmax=1, ls='--', linewidth=1., color='gray')
        plt.text(100.4, scores[n], str(int(scores[n])), {'color':color, 'fontsize':12}, va="center", ha="left")
        axes.axvline(wp[n], ymin=0, ymax=np.log(scores[n])/np.log(1e4), ls='--', linewidth=1., color='gray')
    pylab.xlim(0,100)
    pylab.ylim(1,1e3)
    plt.yscale('log')
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    plt.yticks([10**int(n) for n in np.arange(0,5)], ['1','$10^1$','$10^2$','$10^3$','$10^4$'])
    plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.ylabel('$1/\epsilon_{\operatorname{bkg}}$', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper right', fontsize=15, ncol=2 if len(metrics_list)<9 else 3)
    plt.subplot(2, 1, 2); pylab.grid(True); axes = plt.gca()
    max_ROC_var = 0
    for metric in metrics_list:
        fpr, tpr = [metrics_dict[metric][key] for key in ['fpr','tpr']]
        len_0    = np.sum(fpr==0)
        if ROC_type == 'gain':
            ROC_var = tpr[len_0:]/fpr[len_0:]
        if ROC_type == 'sigma':
            n_sig = np.sum(weights[y_true==0])
            n_bkg = np.sum(weights[y_true==1])
            ROC_var = n_sig*tpr[len_0:]/np.sqrt(n_bkg*fpr[len_0:])
        max_ROC_var = max(max_ROC_var, np.max(ROC_var[100*tpr[len_0:]>=1]))
        plt.plot(100*tpr[len_0:], ROC_var, label=metric, lw=2, color=colors_dict[metric])
    if ROC_type == 'sigma':
        val = ROC_var[-1]
        axes.axhline(val, xmin=0, xmax=1, ls='--', linewidth=1., color='gray')
        plt.text(100.4, val, format(val,'.1f'), {'color':'gray', 'fontsize':12}, va="center", ha="left")
    pylab.xlim(1,100)
    pylab.ylim(0,np.ceil(max_ROC_var))
    plt.xscale('log')
    plt.xticks([1,10,100], ['1','10', '100'])
    axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    location = 'upper right'
    plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
    if ROC_type == 'gain':
        plt.ylabel('$G_{S/B}=\epsilon_{\operatorname{sig}}/\epsilon_{\operatorname{bkg}}$', fontsize=25)
    if ROC_type == 'sigma':
        plt.ylabel('$\sigma=n_{\operatorname{sig}}\epsilon_{\operatorname{sig}}$/'
                   +'$\sqrt{n_{\operatorname{bkg}}\epsilon_{\operatorname{bkg}}}$', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper left', fontsize=15, ncol=2 if len(metrics_list)<9 else 3)
    file_name = output_dir + '/' + 'ROC_' + ROC_type + '.png'
    print('Saving', format(ROC_type,'5s'), 'ROC curves   to:', file_name); plt.savefig(file_name)


def get_rates(y_true, X_losses, weights, metric, return_dict):
    fpr, tpr, _ = metrics.roc_curve(y_true, X_losses[metric], pos_label=0, sample_weight=weights)
    return_dict[metric] = {'fpr':fpr, 'tpr':tpr}


def ROC_rates(y_true, X_losses, weights, metrics_list):
    manager      = mp.Manager(); return_dict = manager.dict()
    arguments    = [(y_true, X_losses, weights, metric, return_dict) for metric in metrics_list]
    processes    = [mp.Process(target=get_rates, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    return {metric:return_dict[metric] for metric in metrics_list}


def KS_distance(dist_1, dist_2, weights_1=None, weights_2=None):
    if np.any(weights_1) == None: weights_1 = np.ones_like(dist_1)
    if np.any(weights_2) == None: weights_2 = np.ones_like(dist_2)
    idx_1     = np.argsort(dist_1)
    idx_2     = np.argsort(dist_2)
    dist_1    = dist_1[idx_1]
    dist_2    = dist_2[idx_2]
    weights_1 = weights_1[idx_1]
    weights_2 = weights_2[idx_2]
    dist_all  = np.concatenate([dist_1, dist_2])
    weights_1 = np.hstack([0, np.cumsum(weights_1)/sum(weights_1)])
    weights_2 = np.hstack([0, np.cumsum(weights_2)/sum(weights_2)])
    cdf_1     = weights_1[tuple([np.searchsorted(dist_1, dist_all, side='right')])]
    cdf_2     = weights_2[tuple([np.searchsorted(dist_2, dist_all, side='right')])]
    return np.max(np.abs(cdf_1 - cdf_2))
