import numpy             as np
import multiprocessing   as mp
import matplotlib.pyplot as plt
from matplotlib    import pylab, ticker
from sklearn       import metrics
from scipy.spatial import distance
from scipy         import stats
from utils         import loss_function, inverse_scaler, get_4v
import sys


def plot_results(y_true, X_true, X_pred, sample, n_dims, metrics, model, output_dir):
    print('PLOTTING RESULTS:')
    manager     = mp.Manager(); X_losses = manager.dict()
    X_true_dict = {metric:X_true for metric in metrics}; #X_true_dict['Inputs'] = sample['constituents']
    arguments   = [(X_true_dict[metric], X_pred, n_dims, metric, X_losses) for metric in metrics]
    processes   = [mp.Process(target=loss_function, args=arg) for arg in arguments]
    for job in processes: job.start()
    for job in processes: job.join()
    # Adding latent space KLD metric
    metrics   += ['Latent']; model(X_true); X_losses['Latent'] = model.losses[0].numpy()
    processes  = [mp.Process(target=ROC_curves, args=(y_true, X_losses, sample['weights'], metrics, output_dir))]
    #processes += [mp.Process(target=pt_reconstruction, args=(X_true, X_pred, y_true, sample['weights'], output_dir))]
    arguments  = [(y_true, X_losses[metric], sample['weights'], metric, output_dir) for metric in metrics]
    processes += [mp.Process(target=loss_distributions, args=arg) for arg in arguments]
    arguments  = [(y_true, X_losses, sample['M'], metrics, output_dir)]
    processes += [mp.Process(target=mass_correlation, args=arg) for arg in arguments]
    for job in processes: job.start()
    for job in processes: job.join()


def loss_distributions(y_true, X_loss, weights, metric, output_dir, n_bins=100):
    if metric in ['DNN', 'FCN']:
        tagger_distributions(y_true, X_loss, weights, metric, output_dir)
        return
    min_dict = {'MSE':0  , 'MAE':0  , 'X-S'   :0  , 'JSD'   :0,   'EMD':0,
                'KSD':0  , 'KLD':0  , 'Inputs':0  , 'Latent':0}
    #max_dict = {'MSE':0.5, 'MAE':0.5, 'X-S'   :0.5, 'JSD'   :0.4, 'EMD':25,
    #            'KSD':1.0, 'KLD':0.5, 'Inputs':1  , 'Latent':1e5}
    max_dict = {'MSE':0.001, 'MAE':0.05, 'X-S'   :0.5, 'JSD'   :1, 'EMD':1,
                'KSD':1.0, 'KLD':0.5, 'Inputs':0.02  , 'Latent':2e3}
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


def mass_correlation(y_true, X_losses, X_mass, metrics, output_dir, n_bins=100):
    #metrics = ['JSD']
    colors_dict = {'MSE':'tab:green', 'MAE'   :'tab:brown' , 'X-S'   :'tab:purple',
                   'JSD':'tab:blue' , 'EMD'   :'tab:orange', 'KSD'   :'black'     ,
                   'KLD':'tab:red'  , 'Inputs':'gray'      , 'Latent':'tab:cyan'}
    #max_dict = {'MSE':0.5, 'MAE':0.5, 'X-S'   :0.5, 'JSD'   :0.4, 'EMD':25,
    #            'KSD':1.0, 'KLD':0.5, 'Inputs':1  , 'Latent':1e5}
    max_dict = {'MSE':0.001, 'MAE':0.05, 'X-S'   :0.5, 'JSD'   :1, 'EMD':1,
                'KSD':1.0, 'KLD':0.5, 'Inputs':0.02  , 'Latent':2e3}
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.ylabel('KS distance', fontsize=25)
    labels = [r'$t\bar{t}$ jets', 'QCD jets']; colors = ['tab:orange', 'tab:blue']
    def get_KSD(y_true, X_losses, X_mass, metric, KSD_dict):
        X_loss    = X_losses[metric]
        min_loss  = np.min(X_loss)
        max_loss  = min(max_dict[metric], np.max(X_loss))
        bin_width = (max_loss-min_loss)/n_bins
        losses    = [min_loss + k*bin_width for k in np.arange(n_bins)]
        KSD = []; sig_eff = []; bkg_eff = []
        for loss in losses:
            sig_loss     = X_loss[y_true==0]
            bkg_loss     = X_loss[y_true==1]
            bkg_mass     = X_mass[y_true==1]
            bkg_mass_cut = bkg_mass[bkg_loss>loss]
            if len(bkg_mass_cut) != 0:
                KSD     += [ stats.ks_2samp(bkg_mass, bkg_mass_cut)[0] ]
                sig_eff += [ 100*np.sum(sig_loss>loss)/len(sig_loss)   ]
                bkg_eff += [ 100*np.sum(bkg_loss>loss)/len(bkg_loss)   ]
        KSD_dict[metric] = KSD, sig_eff, bkg_eff
    manager   = mp.Manager(); KSD_dict = manager.dict()
    arguments = [(y_true, X_losses, X_mass, metric, KSD_dict) for metric in metrics]
    processes = [mp.Process(target=get_KSD, args=arg) for arg in arguments]
    for job in processes: job.start()
    for job in processes: job.join()
    for metric in metrics:
        label = str(metric)+(' metric' if len(metrics)==1 else '')
        KSD, sig_eff, bkg_eff = KSD_dict[metric]
        plt.plot(sig_eff, KSD, label=label, color=colors_dict[metric], lw=2)
        if len(metrics) == 1:
            for bkg_rej in [95,90,80,50]:
                idx = (np.abs(100-np.array(bkg_eff) - bkg_rej)).argmin()
                label = '$\epsilon_{\operatorname{bkg}}$: '+format(100-bkg_rej,'>2d')+'%'
                plt.scatter(sig_eff[idx], KSD[idx], s=40, label=label, zorder=10)
    '''
    for metric in metrics:
        X_loss    = X_losses[metric]
        min_loss  = np.min(X_loss)
        max_loss  = min(max_dict[metric], np.max(X_loss))
        bin_width = (max_loss-min_loss)/n_bins
        losses    = [min_loss + k*bin_width for k in np.arange(n_bins)]
        KSD     = []
        sig_eff = []
        bkg_eff = []
        for loss in losses:
            sig_loss     = X_loss[y_true==0]
            bkg_loss     = X_loss[y_true==1]
            bkg_mass     = X_mass[y_true==1]
            bkg_mass_cut = bkg_mass[bkg_loss>loss]
            if len(bkg_mass_cut) != 0:
                KSD     += [ stats.ks_2samp(bkg_mass, bkg_mass_cut)[0] ]
                sig_eff += [ 100*np.sum(sig_loss>loss)/len(sig_loss)   ]
                bkg_eff += [ 100*np.sum(bkg_loss>loss)/len(bkg_loss)   ]
        label = str(metric)+(' metric' if len(metrics)==1 else '')
        plt.plot(sig_eff, KSD, label=label, color=colors_dict[metric], lw=2)
    if len(metrics) == 1:
        for bkg_rej in [95,90,80,50]:
            idx = (np.abs(100-np.array(bkg_eff) - bkg_rej)).argmin()
            #label = '$1-\epsilon_{\operatorname{bkg}}$: '+format(bkg_rej/100,'.2f')
            label = '$\epsilon_{\operatorname{bkg}}$: '+format(100-bkg_rej,'>2d')+'%'
            plt.scatter(sig_eff[idx], KSD[idx], s=40, label=label)
    '''
    pylab.xlim(0, 100)
    pylab.ylim(0, 1.0)
    ncol = 1 if len(metrics)==1 else 2 if len(metrics)<9 else 3
    plt.legend(loc='upper right', fontsize=15, ncol=ncol)
    file_name = output_dir + '/' + 'mass_sculpting.png'
    print('Saving mass correlations  to:', file_name); plt.savefig(file_name)


def var_distributions(samples, output_dir, sig_bins, bkg_bins, var, normalize=True, density=True, log=True):
    labels = {0:[r'$t\bar{t}$', 'QCD', 'All'], 1:[r'$t\bar{t}$ (cut)', 'QCD (cut)','All (cut)']}
    labels = {0:[r'$W$', 'QCD', 'All'], 1:[r'$W$', 'QCD (cut)','All (cut)']}
    colors = ['tab:orange', 'tab:blue', 'tab:brown']; alphas = [1, 0.5]
    n_bins = [sig_bins, bkg_bins, bkg_bins]
    xlabel = {'pt':'$p_t$', 'M':'$M$'}[var]
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    if not isinstance(samples, list): samples = [samples]
    for n in [0, 1]:
        for sample in samples:
            condition = sample['JZW']==-1 if n==0 else sample['JZW']>= 0 if n==1 else sample['JZW']>=-2
            if not np.any(condition): continue
            variable  = np.float32(sample[var][condition]); weights = sample['weights'][condition]
            min_val, max_val = np.min(variable), np.max(variable)
            if var == 'M': min_val, max_val = 0, 300
            bin_width = (max_val - min_val) / n_bins[n]
            bins      = [min_val + k*bin_width for k in np.arange(n_bins[n]+1)]
            if normalize:
                weights *= 100/(np.sum(sample['weights'])) #100/(np.sum(weights))
                if density:
                    indices  = np.searchsorted(bins, variable, side='right')
                    weights /= np.take(np.diff(bins), np.minimum(indices, len(bins)-1)-1)
            pylab.hist(variable, bins, histtype='step', weights=weights, color=colors[n], lw=2, log=log,
                       alpha=alphas[samples.index(sample)], label=labels[samples.index(sample)][n])
    if normalize:
        if log:
            #pylab.xlim(0, 2000); pylab.ylim(1e-5, 1e0)
            pylab.xlim(0, 6000); pylab.ylim(1e-4, 1e-1)
        else:
            pylab.xlim(0, 6000); pylab.ylim(0, 0.02)
            pylab.yticks(np.arange(0,0.025,0.005))
            #pylab.xlim(0, 6000); pylab.ylim(0, 0.05)
            #pylab.yticks(np.arange(0,0.06,0.01))
    else: pylab.xlim(0, 300); pylab.ylim(1e-1, 1e7)
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    if not log: axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.tick_params(axis='both', which='major', labelsize=14)
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel(xlabel+' (GeV)', fontsize=24)
    y_label = (' density' if normalize and density else '')+' ('+('%' if normalize else r'36 fb$^{-1}$') + ')'
    plt.ylabel('Distribution' + y_label, fontsize=24)
    plt.legend(loc='upper right', ncol=1 if len(samples)==1 else 2, fontsize=18)
    file_name = output_dir+'/'+'var_distributions.png'
    print('Saving pt distributions to:', file_name); plt.savefig(file_name)


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
    #pylab.xlim(0, 6000)
    #pylab.xlim(10, 24)
    pylab.xlim(0.4, 0.5)
    #pylab.ylim(0, 0.5)
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('$p_t$ (quantile space)', fontsize=24)
    #plt.xlabel('$p_t$ (GeV)', fontsize=24)
    plt.ylabel('Distribution density (%/GeV)', fontsize=24)
    plt.legend(loc='upper right', ncol=2, fontsize=18)
    file_name = output_dir+'/'+'pt_reconstruction.png'
    print('Saving pt reconstruction  to:', file_name); plt.savefig(file_name)
#'''
def quantile_reconstruction(y_true, X_true, X_pred, sample, scaler, output_dir):
    pt_reconstruction(X_true, X_pred, y_true, sample['weights'], output_dir)
    #pt_reconstruction(X_pred, X_pred, y_true, None             , output_dir)
    #X_true = inverse_scaler(X_true, scaler)
    #pt_reconstruction(sample['clusters'], X_true, y_true, None, output_dir)
    #X_pred = inverse_scaler(X_pred, scaler)
    #pt_reconstruction(sample['clusters'], X_pred, y_true, None, output_dir)
#'''


def ROC_curves(y_true, X_losses, weights, metrics_list, output_dir, wp=[1,10], sigma=False):
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
    max_gain = 0
    for metric in metrics_list:
        fpr, tpr = [metrics_dict[metric][key] for key in ['fpr','tpr']]
        len_0    = np.sum(fpr==0)
        if sigma:
            n_sig = np.sum(weights[y_true==0])
            n_bkg = np.sum(weights[y_true==1])
            gain  = n_sig*tpr[len_0:]/np.sqrt(n_bkg*fpr[len_0:])
        else:
            gain     = tpr[len_0:]/fpr[len_0:]
        max_gain = max(max_gain, np.max(gain[100*tpr[len_0:]>=1]))
        plt.plot(100*tpr[len_0:], gain, label=metric, lw=2, color=colors_dict[metric])
    pylab.xlim(1,100)
    pylab.ylim(1,np.ceil(max_gain))
    plt.xscale('log')
    plt.xticks([1,10,100], ['1','10', '100'])
    if len(set(metrics_list)&{'DNN','FCN'})!=0 and not sigma:
        plt.yscale('log')
        location = 'upper right'
    else:
        axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        location = 'upper right'
    plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
    if sigma:
        plt.ylabel('$\sigma=n_{\operatorname{sig}}\epsilon_{\operatorname{sig}}$/'
                   +'$\sqrt{n_{\operatorname{bkg}}\epsilon_{\operatorname{bkg}}}$', fontsize=25)
    else:
        plt.ylabel('$G_{S/B}=\epsilon_{\operatorname{sig}}/\epsilon_{\operatorname{bkg}}$', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper left', fontsize=15, ncol=2 if len(metrics_list)<9 else 3)
    file_name = output_dir+'/'+'ROC_curves.png'
    print('Saving ROC curves         to:', file_name); plt.savefig(file_name)


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


def tagger_distributions(y_true, y_prob, weights, metric, output_dir):
    label_dict={0:r'$t\bar{t}$', 1:'QCD'}; color_dict={0:'tab:orange', 1:'tab:blue'}
    def logit(x, delta=1e-16):
        x = np.clip(np.float64(x), delta, 1-delta)
        return np.log10(x) - np.log10(1-x)
    def class_histo(y_true, y_prob, bins, colors):
        for n in set(y_true):
            class_var      = y_prob[y_true==n]
            class_weights  =  weights[y_true==n]
            class_weights *= 100/np.sum(class_weights)
            pylab.hist(class_var, bins=bins, label=label_dict[n], histtype='step',
                       weights=class_weights, log=True, color=colors[n], lw=2, alpha=1)[0]
            class_weights = len(class_var)*[100/len(class_var)]
            pylab.hist(class_var, bins=bins, label=label_dict[n]+' (unweighted)', histtype='step',
                       weights=class_weights, log=True, color=colors[n], lw=2, alpha=0.5)[0]
    plt.figure(figsize=(11,15))
    plt.subplot(2, 1, 1); pylab.grid(True); axes = plt.gca()
    pylab.xlim(0,100); pylab.ylim(1e-3, 1e2)
    plt.xticks(np.arange(0,101,step=10))
    bin_step = 0.5; bins = np.arange(0, 100+bin_step, bin_step)
    class_histo(y_true, 100*y_prob, bins, color_dict)
    plt.xlabel(metric+' tagger (%)', fontsize=25)
    plt.ylabel('Distribution (%)', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2,3,0,1]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
               loc='upper right', fontsize=16 , ncol=1)
    plt.subplot(2, 1, 2); pylab.grid(True); axes = plt.gca()
    x_min=-7; x_max=5; pylab.xlim(x_min, x_max); pylab.ylim(1e-3, 1e1)
    pos  =                   [  10**float(n)      for n in np.arange(x_min,0)       ]
    pos += [0.5]           + [1-10**float(n)      for n in np.arange(-1,-x_max-1,-1)]
    lab  =                   ['$10^{'+str(n)+'}$' for n in np.arange(x_min+2,0)     ]
    lab += [1,10,50,90,99] + ['99.'+n*'9'         for n in np.arange(1,x_max-1)     ]
    plt.xticks(logit(np.array(pos)), lab, rotation=15)
    bin_step = 0.1; bins = np.arange(x_min-1, x_max+1, bin_step)
    class_histo(y_true, logit(y_prob), bins, color_dict)
    plt.xlabel(metric+' tagger (%)', fontsize=25)
    plt.ylabel('Distribution (%)', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2,3,0,1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
               loc='upper right', fontsize=16, ncol=1)
    plt.subplots_adjust(top=0.95, bottom=0.1, hspace=0.2)
    file_name = output_dir+'/'+metric+'_loss.png'
    print('Saving loss distributions to:', file_name); plt.savefig(file_name)
