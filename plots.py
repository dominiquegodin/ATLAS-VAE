import numpy             as np
import multiprocessing   as mp
import matplotlib.pyplot as plt
from matplotlib import pylab, ticker
from sklearn    import metrics
from utils      import loss_function, get_4v, apply_cut


def var_distributions(samples, output_dir, sig_bins, bkg_bins, var, normalize=True, density=True, log=True):
    labels = {0:[r'$t\bar{t}$', 'QCD','All'], 1:[r'$t\bar{t}$ (cut)', 'QCD (cut)','All (cut)']}
    colors = {0:['tab:orange', 'tab:blue', 'tab:brown'], 1:['orange', 'skyblue', 'darkgoldenrod']}
    n_bins = [sig_bins, bkg_bins, bkg_bins]
    xlabel = {'pt':'$p_t$', 'M':'$M$'}[var]
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    if not isinstance(samples, list): samples = [samples]
    for sample in samples:
        for n in [0,1]:
            condition = sample['JZW']==-1 if n==0 else sample['JZW']>= 0 if n==1 else sample['JZW']>=-2
            if not np.any(condition): continue
            variable  = np.float32(sample[var][condition]); weights = sample['weights'][condition]
            bin_width = (np.max(variable) - np.min(variable)) / n_bins[n]
            bins      = [np.min(variable) + k*bin_width for k in np.arange(n_bins[n]+1)]
            if normalize:
                #weights *= 100/(np.sum(weights))
                weights *= 100/(np.sum(sample['weights']))
                if density:
                    indices  = np.searchsorted(bins, variable, side='right')
                    weights /= np.take(np.diff(bins), np.minimum(indices, len(bins)-1)-1)
            label = labels[samples.index(sample)][n]; color = colors[samples.index(sample)][n]
            pylab.hist(variable, bins, histtype='step', weights=weights, label=label, color=color, lw=2, log=log)
    if normalize:
        if log:
            #pylab.xlim(0, 300); pylab.ylim(1e-5, 1e1)
            #pylab.xlim(0, 2000); pylab.ylim(1e-5, 1e0)
            pylab.xlim(0, 6000); pylab.ylim(1e-3, 1e-1)
        else: pylab.xlim(0, 6000); pylab.ylim(0, 0.018)
    else: pylab.xlim(0, 300); pylab.ylim(1e0, 1e7)
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    if not log: axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.tick_params(axis='both', which='major', labelsize=14)
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel(xlabel+' (GeV)', fontsize=24)
    y_label = (' density' if normalize and density else '') + ' (' + ('%' if normalize else 'entries') + ')'
    plt.ylabel('Distribution' + y_label, fontsize=24)
    plt.legend(loc='upper right', ncol=2, fontsize=18)
    file_name = output_dir+'/'+'var_distributions.png'
    print('\nSaving pt distributions to:', file_name, '\n'); plt.savefig(file_name)


def pt_reconstruction(X_true, X_pred, y_true, weights, output_dir):
    pt_true = get_4v(X_true)['pt']; pt_pred = get_4v(X_pred)['pt']
    if np.any(weights) == None: weights = np.array(len(y_true)*[1.])
    bin_width = 0.2
    bins = list(np.arange(0, np.max(pt_true), bin_width))
    bins = [0] + [n for n in bins if n > np.min(pt_true)] + [np.max(pt_true)+1e-3]
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    labels = [r'$t\bar{t}$ jets', 'QCD jets']; colors = ['tab:orange', 'tab:blue']
    for n in set(y_true):
        hist_weights  = weights[y_true==n]
        hist_weights *= 100/np.sum(hist_weights)
        pylab.hist(pt_true[y_true==n], bins, histtype='step', weights=hist_weights,
                   label=labels[n], lw=2, color=colors[n])
        pylab.hist(pt_pred[y_true==n], bins, histtype='step', weights=hist_weights,
                   label=labels[n]+' (rec)', color=colors[n], lw=2, ls='--')
    pylab.xlim(0, 2)
    #plt.xticks(np.arange(10, 26, 5))
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('$p_t$ (scaler space)', fontsize=24)
    plt.ylabel('Distribution (%)', fontsize=24)
    plt.legend(loc='upper right', fontsize=18)
    file_name = output_dir+'/'+'pt_reconstruction.png'
    print('Saving pt reconstruction  to:', file_name); plt.savefig(file_name)


def loss_distributions(y_true, X_loss, metric, weights, output_dir, n_bins=100):
    min_dict  = {'JSD':0  , 'MSE':0   , 'MAE':0  , 'EMD':0 , 'KLD':0  , 'X-S':0,   'B-1':0, 'B-2':0.0245}
    max_dict  = {'JSD':0.3, 'MSE':0.16, 'MAE':0.3, 'EMD':30, 'KLD':0.3, 'X-S':0.5, 'B-1':1, 'B-2':0.0265}
    min_loss  = min_dict[metric]
    max_loss  = min(max_dict[metric], np.max(X_loss))
    bin_width = (max_loss-min_loss)/n_bins
    bins      = list(np.arange(min_loss, max_loss, bin_width)) + [max_loss]
    labels    = [r'$t\bar{t}$ jets', 'QCD jets']; colors = ['tab:orange', 'tab:blue']
    if np.any(weights) == None: weights = np.array(len(y_true)*[1.])
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    for n in set(y_true):
        hist_weights  = weights[y_true==n]
        hist_weights *= 1/np.sum(hist_weights)/bin_width
        pylab.hist(X_loss[y_true==n], bins, histtype='step', weights=hist_weights,
                   log=False, label=labels[n], color=colors[n], lw=2)
    pylab.xlim(min_dict[metric], max_loss)
    #pylab.xticks(np.arange(0,0.16,0.05))
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel(metric+' reconstruction loss', fontsize=24)
    plt.ylabel('Distribution density', fontsize=24)
    plt.legend(loc='upper left', fontsize=18)
    file_name = output_dir+'/'+metric+'_loss.png'
    print('Saving loss distributions to:', file_name); plt.savefig(file_name)


def mass_correlation(y_true, X_loss, metric, X_mass, output_dir, n_bins=100):
    min_dict = {'JSD':0  , 'MSE':0   , 'MAE':0  , 'EMD':0 , 'KLD':0  , 'X-S':0,   'B-1':0, 'B-2':0.0245}
    max_dict = {'JSD':0.3, 'MSE':0.16, 'MAE':0.3, 'EMD':30, 'KLD':0.3, 'X-S':0.5, 'B-1':1, 'B-2':0.0265}
    max_loss = min(max_dict[metric], np.max(X_loss))
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    pylab.xlim(min_dict[metric], max_loss)
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Cut on '+metric+' reconstruction loss', fontsize=24)
    plt.ylabel('Mean jet mass (GeV)', fontsize=24)
    labels = [r'$t\bar{t}$ jets', 'QCD jets']; colors = ['tab:orange', 'tab:blue']
    for n in set(y_true):
        n_loss    = X_loss[y_true==n]
        n_mass    = X_mass[y_true==n]
        min_loss  = np.min(n_loss)
        max_loss  = min(max_dict[metric], np.max(n_loss))
        bin_width = (max_loss-min_loss)/n_bins
        losses    = [min_loss + k*bin_width for k in np.arange(n_bins)]
        masses    = [np.mean(n_mass[n_loss>=loss]) for loss in losses]
        plt.plot(losses, masses, label=labels[n], color=colors[n], lw=2)
    plt.legend(loc='upper left', fontsize=18)
    file_name = output_dir+'/'+metric+'_correlation.png'
    print('Saving mass correlations  to:', file_name); plt.savefig(file_name)


def get_rates(X_true, X_pred, y_true, weights, jets, metric, return_dict):
    X_loss      = loss_function(X_true, X_pred, metric)
    if metric == 'B-2': X_loss = loss_function(jets, X_pred, metric)
    fpr, tpr, _ = metrics.roc_curve(y_true, X_loss, pos_label=0, sample_weight=weights)
    return_dict[metric] = {'fpr':fpr, 'tpr':tpr}
def ROC_rates(X_true, y_true, X_pred, weights, jets, metrics_list):
    manager      = mp.Manager(); return_dict = manager.dict()
    arguments    = [(X_true, y_true, X_pred, weights, jets, metric, return_dict) for metric in metrics_list]
    processes    = [mp.Process(target=get_rates, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    return {metric:return_dict[metric] for metric in metrics_list}
def ROC_curves(X_true, y_true, X_pred, sample, output_dir, metrics_list, wp):
    colors_dict  = {'JSD':'tab:blue', 'MSE':'tab:green' , 'EMD':'tab:orange', 'B-1':'gray',
                    'KLD':'tab:red' , 'X-S':'tab:purple', 'MAE':'tab:brown' , 'B-2':'darkgray'}
    metrics_dict = ROC_rates(X_true, y_true, X_pred, sample['weights'], sample['jets'], metrics_list)
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
        #plt.text(wp[n], 0.735, str(int(wp[n])), {'color':'tab:blue', 'fontsize':12}, va="center", ha="center")
    pylab.xlim(0,100)
    pylab.ylim(1,1e3)
    plt.yscale('log')
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    plt.yticks([10**int(n) for n in np.arange(0,5)], ['1','$10^1$','$10^2$','$10^3$','$10^4$'])
    plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.ylabel('$1/\epsilon_{\operatorname{bkg}}$', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper right', fontsize=16, ncol=2)
    plt.subplot(2, 1, 2); pylab.grid(True); axes = plt.gca()
    max_gain = 0
    for metric in metrics_list:
        fpr, tpr = [metrics_dict[metric][key] for key in ['fpr','tpr']]
        len_0    = np.sum(fpr==0)
        gain     = tpr[len_0:]/fpr[len_0:]
        max_gain = max(max_gain, np.max(gain[100*tpr[len_0:]>=1]))
        plt.plot(100*tpr[len_0:], gain, label=metric, lw=2, color=colors_dict[metric])
    pylab.xlim(1,100)
    pylab.ylim(1,np.ceil(max_gain))
    plt.xscale('log')
    plt.xticks([1,10,100], ['1','10', '100'])
    axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.ylabel('$G_{S/B}=\epsilon_{\operatorname{sig}}/\epsilon_{\operatorname{bkg}}$', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper right', fontsize=16, ncol=2)
    file_name = output_dir+'/'+'ROC_curves.png'
    print('Saving ROC curves         to:', file_name); plt.savefig(file_name)
