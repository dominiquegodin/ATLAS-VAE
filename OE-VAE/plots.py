import numpy             as np
import multiprocessing   as mp
import matplotlib.pyplot as plt
import os, sys, pickle, warnings
from matplotlib    import pylab, ticker
from sklearn       import metrics
from scipy.spatial import distance
from functools     import partial
from utils         import loss_function, inverse_scaler, get_4v, bump_hunter, make_sample, latent_loss, get_idx


def plot_results(y_true, X_true, X_pred, sample, n_dims, model, metrics, wp_metric, sig_data, output_dir):
    print('\nPLOTTING RESULTS:')
    manager = mp.Manager(); X_losses = manager.dict()
    X_true_dict = {metric:X_true for metric in metrics if metric!='Latent'}
    if False: X_true_dict['Inputs'] = sample['constituents']
    arguments = [(X_true_dict[metric], X_pred, n_dims, metric, X_losses) for metric in metrics if metric!='Latent']
    processes = [mp.Process(target=loss_function, args=arg) for arg in arguments]
    for job in processes: job.start()
    for job in processes: job.join()
    """ Adding latent space KLD metric and loss """
    if 'Latent' in metrics: X_losses['Latent'] = latent_loss(X_true, model)
    best_loss  = bump_scan(y_true, X_losses[wp_metric], wp_metric, sample, sig_data, output_dir)
    processes  = [mp.Process(target=ROC_curves, args=(y_true, X_losses, sample['weights'], metrics, output_dir))]
    arguments  = [(y_true, X_losses, sample['m'], sample['weights'], metrics, wp_metric, output_dir)]
    processes += [mp.Process(target=mass_correlation, args=arg) for arg in arguments]
    arguments  = [(y_true, X_losses[metric], sample['weights'], metric, best_loss, output_dir) for metric in metrics]
    processes += [mp.Process(target=loss_distributions, args=arg) for arg in arguments]
    #processes += [mp.Process(target=pt_reconstruction, args=(X_true, X_pred, y_true, sample['weights'], output_dir))]
    for job in processes: job.start()
    for job in processes: job.join()


def bump_scan(y_true, X_loss, wp_metric, sample, sig_data, output_dir, n_cuts=200):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, X_loss, pos_label=0, sample_weight=sample['weights'])
    step    = 100/n_cuts
    tpr_val = np.append(np.arange(0,20,step), np.arange(20,100,step))
    idx     = np.minimum(np.searchsorted(100*tpr, tpr_val, side='right'), len(tpr)-1)
    sample  = {key:sample[key] for key in ['JZW','m','pt','weights']}
    global get_sigma
    def get_sigma(sample, X_loss, losses, idx):
        cut_sample = {key:sample[key][X_loss>losses[idx]] for key in sample}
        try   : return bump_hunter(cut_sample, make_histo=False)
        except: return None
    with mp.Pool() as pool:
        sigma = np.array(pool.map(partial(get_sigma, sample, X_loss, thresholds), idx))
    sig_eff, losses = np.take(100*tpr, idx), np.take(thresholds, idx)
    none_filter = sigma != np.array(None)
    sig_eff, losses, sigma = sig_eff[none_filter], losses[none_filter], sigma[none_filter]
    plt.figure(figsize=(13,8)); pylab.grid(True); axes = plt.gca()
    plt.plot(sig_eff, sigma, label='', color='tab:blue', lw=2, zorder=1)
    factor  = np.floor(np.log10(np.max(sigma)))
    val_pre = '.'+str(int(max(1,-factor+1)))+'f'
    factor = 10**factor
    pylab.xlim(0, 100)
    pylab.ylim(factor*np.floor(np.min(sigma/factor)), factor*np.ceil(np.max(sigma/factor)))
    end_val = sigma[-1]
    plt.text(100.4, end_val, format(end_val,val_pre), {'color':'gray', 'fontsize':14}, va="center", ha="left")
    max_val = np.max(sigma)
    max_eff = sig_eff[np.argmax(sigma)]
    axes.axhline(max_val, xmin=max_eff/100, xmax=1, ls='--', linewidth=1., color='dimgray')
    plt.text(100.4, max_val, format(max_val,val_pre), {'color':'dimgray', 'fontsize':14}, va="center", ha="left")
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.ylabel('Significance', fontsize=25)
    file_name = output_dir+'/'+'BH_sigma.png'
    print('Saving max significance  to:', file_name); plt.savefig(file_name)
    """ Printing bkg suppression and bum hunting plots at maximum significance cut """
    best_loss = {'metric':wp_metric, 'sig_eff':sig_eff[np.argmax(sigma)], 'loss':losses[np.argmax(sigma)]}
    cut_sample = {key:sample[key][X_loss>best_loss['loss']] for key in sample}
    bump_hunter(cut_sample, output_dir, cut_type='best', print_info=False)
    for var in ['m','pt']: plot_distributions([sample,cut_sample], sig_data, plot_var=var, bin_sizes={'m':2.5,'pt':10},
                                              output_dir=output_dir, file_name='BH_bkg_supp_'+var+'.png')
    return best_loss


def loss_distributions(y_true, X_loss, weights, metric, best_loss, output_dir, n_bins=100):
    min_dict = {'MSE':0  , 'MAE':0   , 'X-S'   :0   , 'JSD'   :0, 'EMD':0,
                'KSD':0  , 'KLD':0   , 'Inputs':0   , 'Latent':0}
    max_dict = {'MSE':50 , 'MAE':0.05, 'X-S'   :0.5 , 'JSD'   :1, 'EMD':1,
                'KSD':1.0, 'KLD':0.5 , 'Inputs':0.02, 'Latent':20}
    min_loss  = min_dict[metric]
    max_loss  = min(max_dict[metric], np.max(X_loss))
    bin_width = (max_loss-min_loss)/n_bins
    bins      = list(np.arange(min_loss, max_loss, bin_width)) + [max_loss]
    labels    = [r'$t\bar{t}$ jets', 'QCD jets']; colors = ['tab:orange', 'tab:blue']
    if np.any(weights) == None: weights = np.array(len(y_true)*[1.])
    plt.figure(figsize=(13,8)); pylab.grid(True); axes = plt.gca()
    for n in set(y_true):
        hist_weights  = weights[y_true==n]
        hist_weights *= 100/np.sum(hist_weights)/bin_width
        h = pylab.hist(X_loss[y_true==n], bins, histtype='step', weights=hist_weights, label=labels[n],
                   log=True if metric in ['DNN','FCN'] else False, color=colors[n], lw=2)
    if metric == best_loss['metric']:
        axes.axvline(best_loss['loss'], ymin=0, ymax=1, ls='--', linewidth=1., color='dimgray')
        plt.text(best_loss['loss']*1.02, axes.get_ylim()[1]*0.98, 'best cut',
                 {'color':'dimgray', 'fontsize':10}, va="center", ha="left")
    pylab.xlim(min_dict[metric], max_loss)
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.tick_params(axis='both', which='major', labelsize=14)
    label = 'KLD latent loss' if metric=='Latent' else metric+' reconstruction loss'
    plt.xlabel(label, fontsize=24)
    plt.ylabel('Distribution density (%)', fontsize=24)
    plt.legend(loc='upper right', fontsize=18)
    output_dir += '/'+'metrics_losses'
    if not os.path.isdir(output_dir): os.mkdir(output_dir)
    file_name = output_dir+'/'+metric+'_loss.png'
    print('Saving metric loss       to:', file_name); plt.savefig(file_name)


def plot_history(output_dir, first_epoch=0, x_step=10):
    losses = pickle.load(open(output_dir+'/'+'history.pkl', 'rb'))
    plt.figure(figsize=(13,8)); pylab.grid(True); axes = plt.gca()
    epochs = np.arange(1+first_epoch, len(list(losses.values())[0])+1)
    for key, loss in losses.items():
        plt.plot(epochs, loss[first_epoch:], label=key, lw=2)
    pylab.xlim(1, epochs[-1])
    plt.xticks(np.append(1, np.arange(x_step, epochs[-1]+x_step, x_step)))
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    pylab.ylim(0, min(100, np.max(losses['Train loss'][1:])))
    plt.xlabel('Epoch', fontsize=25)
    plt.ylabel('Loss', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='upper right', fontsize=18)
    file_name = output_dir+'/plots/'+'train_history.png'
    print('Saving training history  to:', file_name); plt.savefig(file_name)


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
            #KSD += [ KS_distance(bkg_mass, bkg_mass_cut, bkg_weights, bkg_weights_cut) ]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                JSD += [ distance.jensenshannon(P, Q) ]
            sig_eff += [ 100*tpr[n] ]
            bkg_eff += [ 100*fpr[n] ]
    distances_dict[metric] = KSD, JSD, sig_eff, bkg_eff


def mass_correlation(y_true, X_losses, X_mass, weights, metrics_list, wp_metric, output_dir):
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
    plt.figure(figsize=(13,8)); make_plot('JSD', wp_metric)
    #plt.figure(figsize=(11,16));
    #plt.subplot(2, 1, 1); make_plot('KSD', wp_metric)
    #plt.subplot(2, 1, 2); make_plot('JSD', wp_metric)
    file_name = output_dir + '/' + 'mass_correlation.png'
    print('Saving mass sculpting    to:', file_name); plt.savefig(file_name)


def signal_gain(sample, cut_sample, output_dir, n_bins=50, m_range=(0,200)):
    def get_histo(sample):
        mass    = sample['m']
        weights = sample['weights']
        JZW     = sample['JZW']
        sig_histo, mass = np.histogram(mass[JZW==-1], bins=n_bins, range=m_range, weights=weights[JZW==-1])
        bkg_histo, mass = np.histogram(mass[JZW>= 0], bins=n_bins, range=m_range, weights=weights[JZW>= 0])
        return mass, sig_histo, bkg_histo
    mass, sig_histo    , bkg_histo     = get_histo(sample)
    mass, sig_cut_histo, bkg_cut_histo = get_histo(cut_sample)
    mass = (mass[:-1] + mass[1:])/2
    gain = bkg_histo*sig_cut_histo / (bkg_cut_histo*sig_histo)
    plt.figure(figsize=(13,8)); pylab.grid(True); axes = plt.gca()
    plt.plot(mass, gain, label='gain', color='tab:blue', lw=2)
    file_name = output_dir + '/' + 'signal_gain.png'
    print('Saving signal gain to:', file_name); plt.savefig(file_name)


def plot_distributions(samples, sig_data, plot_var, bin_sizes, output_dir, file_name='', weight_type='None',
                       normalize=True, density=True, log=True, plot_weights=False):
    if   'top' in sig_data: tag = r'$t\bar{t}$'
    elif 'BSM' in sig_data: tag = 'BSM'
    elif 'OoD' in sig_data: tag = 'OoD'
    if 'OoD' in sig_data: labels = {0:[tag,'QCD'], 1:[tag+' (weighted)','QCD (weighted)']}
    else: labels = {0:[tag,'QCD'], 1:[tag+' (cut)','QCD (cut)']}
    colors = ['tab:orange', 'tab:blue', 'tab:brown']; alphas = [1, 0.5]
    xlabel = {'pt':'$p_t$', 'm':'$m$', 'rljet_n_constituents':'Number of constituents'}[plot_var]
    plt.figure(figsize=(13,8)); pylab.grid(True); axes = plt.gca()
    if not isinstance(samples, list): samples = [samples]
    for m in [0,1]:
        for n in range(len(samples)):
            sample = samples[n]
            condition = sample['JZW']==-1 if m==0 else sample['JZW']>= 0 if m==1 else sample['JZW']>=-2
            if not np.any(condition): continue
            variable = np.float32(sample[plot_var][condition])
            weights = sample['weights'][condition]
            if 'flat' in weight_type: min_val, max_val = max(0, np.min(variable))        , np.max(variable        )
            else                    : min_val, max_val = max(0, np.min(sample[plot_var])), np.max(sample[plot_var])
            bins = get_idx(max_val, bin_size=bin_sizes[plot_var], min_val=min_val, integer=False, tuples=False)
            if plot_weights:
                ''' Printing weights histogram '''
                pylab.hist(weights, bins=np.arange(40000), histtype='step', color=colors[m],
                           lw=2, log=log, alpha=alphas[n], label=labels[n][m])
                continue
            if normalize:
                if weight_type == 'None': weights *= 100/(np.sum(samples[0]['weights']))
                else                    : weights *= 100/(np.sum(sample    ['weights'])) #100/(np.sum(weights))
            if density:
                indices  = np.searchsorted(bins, variable, side='right')
                weights /= np.take(np.diff(bins), np.minimum(indices, len(bins)-1)-1)
            pylab.hist(variable, bins, histtype='step', weights=weights, color=colors[m],
                       lw=2, log=log, alpha=alphas[n], label=labels[n][m])
    if not density or plot_weights:
        pass
    elif 'OoD' in sig_data:
        if   plot_var == 'm' : pylab.xlim(0, 1200); pylab.ylim(1e0, 1e5)
        elif plot_var == 'pt': pylab.xlim(0, 3000); pylab.ylim(1e0, 1e5)
    elif 'Geneva' in sig_data:
        if   plot_var == 'm' : pylab.xlim(0, 500) ; pylab.ylim(1e-2, 1e5)
        elif plot_var == 'pt': pylab.xlim(0, 2000); pylab.ylim(1e-2, 1e5)
    else:
        if   plot_var == 'm' : pylab.xlim(0, 500) ; pylab.ylim(1e0, 1e6)
        elif plot_var == 'pt': pylab.xlim(0, 2000); pylab.ylim(1e0, 1e6)
    if normalize:
        if   plot_var == 'm' : pylab.ylim(1e-6, 1e0)
        elif plot_var == 'pt': pylab.ylim(1e-6, 1e0)
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    if not log: axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    plt.xlabel(xlabel+' (GeV)', fontsize=24)
    y_label = ' density' if density else ''
    if normalize: y_label += ' (%)'
    elif sig_data in ['top-UFO','BSM']: y_label += ' ('+r'58.5 fb$^{-1}$'+')'
    plt.ylabel('Distribution' + y_label, fontsize=24)
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='upper right', ncol=1 if len(samples)==1 else 2, fontsize=18)
    if file_name == '':
        file_name = (plot_var if plot_var=='pt' else 'mass')+'_dist.png'
    file_name = output_dir+'/'+file_name
    print('Saving', format(plot_var, '2s'), 'distributions  to:', file_name); plt.savefig(file_name)


def ROC_curves(y_true, X_losses, weights, metrics_list, output_dir, wps=[1,10]):
    colors_dict = {'MSE':'tab:green', 'MAE'   :'tab:brown' , 'X-S'   :'tab:purple',
                   'JSD':'tab:blue' , 'EMD'   :'tab:orange', 'KSD'   :'black'     ,
                   'KLD':'tab:red'  , 'Inputs':'gray'      , 'Latent':'tab:cyan'}
    metrics_dict = ROC_rates(y_true, X_losses, weights, metrics_list)
    """ Background rejection plot """
    plt.figure(figsize=(13,8)); pylab.grid(True); axes = plt.gca()
    for metric in metrics_list:
        fpr, tpr = [metrics_dict[metric][key] for key in ['fpr','tpr']]
        len_0 = np.sum(fpr==0)
        plt.text(0.98, 0.65-4*metrics_list.index(metric)/100, 'AUC: '+format(metrics.auc(fpr,tpr), '.3f'),
                 {'color':colors_dict[metric], 'fontsize':14}, va='center', ha='right', transform=axes.transAxes)
        plt.plot(100*tpr[len_0:], 1/fpr[len_0:], label=metric, lw=2, color=colors_dict[metric])
    metrics_scores = [[metrics_dict[metric][key] for key in ['fpr','tpr']] for metric in metrics_list]
    for wp in wps:
        fpr_list = np.array([fpr[np.argwhere(100*tpr >= wp)[0]] for fpr,tpr in metrics_scores])
        if np.all(fpr_list > 1e-4):
            score = 1/np.min(fpr_list)
            color = colors_dict[metrics_list[np.argmin(fpr_list)]]
            axes.axhline(score, xmin=wp/100, xmax=1, ls='--', linewidth=1., color='dimgray')
            plt.text(100.4, score, str(int(score)), {'color':color, 'fontsize':14}, va="center", ha="left")
            axes.axvline(wp, ymin=0, ymax=np.log(score)/np.log(1e4), ls='--', linewidth=1., color='dimgray')
    pylab.xlim(0,100)
    pylab.ylim(1,1e4)
    plt.yscale('log')
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    plt.yticks([10**int(n) for n in np.arange(0,5)], ['1','$10^1$','$10^2$','$10^3$','$10^4$'])
    plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.ylabel('$1/\epsilon_{\operatorname{bkg}}$', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='upper right', fontsize=15, ncol=2 if len(metrics_list)<9 else 3)
    file_name = output_dir + '/' + 'bkg_rejection.png'
    print('Saving bkg rejection     to:', file_name); plt.savefig(file_name)
    """ Signal gain plot """
    plt.figure(figsize=(13,8)); pylab.grid(True); axes = plt.gca()
    max_ROC_var = 0
    for metric in metrics_list:
        fpr, tpr = [metrics_dict[metric][key] for key in ['fpr','tpr']]
        len_0    = np.sum(fpr==0)
        ROC_var  = tpr[len_0:]/fpr[len_0:]
        max_ROC_var = max(max_ROC_var, np.max(ROC_var[100*tpr[len_0:]>=1]))
        plt.plot(100*tpr[len_0:], ROC_var, label=metric, lw=2, color=colors_dict[metric])
    pylab.xlim(1,100)
    pylab.ylim(0,np.ceil(max_ROC_var))
    plt.xscale('log')
    plt.xticks([1,10,100], ['1','10', '100'])
    axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    location = 'upper right'
    plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.ylabel('$G_{S/B}=\epsilon_{\operatorname{sig}}/\epsilon_{\operatorname{bkg}}$', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='upper left', fontsize=15, ncol=2 if len(metrics_list)<9 else 3)
    file_name = output_dir + '/' + 'signal_gain.png'
    print('Saving signal gain       to:', file_name); plt.savefig(file_name)
    """ Significance plot """
    plt.figure(figsize=(13,8)); pylab.grid(True); axes = plt.gca()
    max_ROC_var = 0
    for metric in metrics_list:
        fpr, tpr = [metrics_dict[metric][key] for key in ['fpr','tpr']]
        len_0    = np.sum(fpr==0)
        n_sig = np.sum(weights[y_true==0])
        n_bkg = np.sum(weights[y_true==1])
        ROC_var = n_sig*tpr[len_0:]/np.sqrt(n_bkg*fpr[len_0:])
        max_ROC_var = max(max_ROC_var, np.max(ROC_var[100*tpr[len_0:]>=1]))
        plt.plot(100*tpr[len_0:], ROC_var, label=metric, lw=2, color=colors_dict[metric])
    val = ROC_var[-1]
    #axes.axhline(val, xmin=0, xmax=1, ls='--', linewidth=1., color='dimgray')
    plt.text(100.4, val, format(val,'.1f'), {'color':'dimgray', 'fontsize':14}, va="center", ha="left")
    pylab.xlim(1,100)
    pylab.ylim(0,np.ceil(max_ROC_var))
    plt.xscale('log')
    plt.xticks([1,10,100], ['1','10', '100'])
    axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    location = 'upper right'
    plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.ylabel('$\sigma=n_{\operatorname{sig}}\epsilon_{\operatorname{sig}}$/'
               +'$\sqrt{n_{\operatorname{bkg}}\epsilon_{\operatorname{bkg}}}$', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='upper left', fontsize=15, ncol=2 if len(metrics_list)<9 else 3)
    file_name = output_dir + '/' + 'significance.png'
    print('Saving significance      to:', file_name); plt.savefig(file_name)


def combined_plots(n_test, n_top, output_dir, plot_var, n_dims=4, n_constituents=20):
    sample_topo = make_sample(n_constituents, 'qcd-topo', 'top-topo', n_test, n_top, n_dims, adjust_weights=True)
    sample_UFO  = make_sample(n_constituents, 'qcd-UFO' , 'BSM'     , n_test, n_top, n_dims, adjust_weights=True)
    samples = [sample_UFO, sample_topo]
    plot_distributions(samples, output_dir, plot_var, sig_bins=200, bkg_bins=400, sig_tag='top')
    sys.exit()


def pt_reconstruction(X_true, X_pred, y_true, weights, output_dir, n_bins=200):
    pt_true = get_4v(X_true)['pt']; pt_pred = get_4v(X_pred)['pt']
    if np.any(weights) == None: weights = np.array(len(y_true)*[1.])
    min_value = min(np.min(pt_true), np.min(pt_pred))
    max_value = max(np.max(pt_true), np.max(pt_pred))
    bin_width = (max_value - min_value) / n_bins
    bins      = [min_value + k*bin_width for k in np.arange(n_bins+1)]
    plt.figure(figsize=(13,8)); pylab.grid(True); axes = plt.gca()
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
