import numpy             as np
import multiprocessing   as mp
import matplotlib.pyplot as plt
from matplotlib          import pylab, ticker
from sklearn             import metrics
from utils               import loss_function, get_4v


def var_distributions(sample, output_dir, var, sig_bins, bkg_bins, density=True, log=True):
    labels = [r'$t\bar{t}$ jets', 'QCD jets','All jets']
    colors = ['tab:orange', 'tab:blue', 'tab:brown']
    n_bins = [sig_bins, bkg_bins, bkg_bins]
    xlabel = {'pt':'$p_t$', 'M':'$M$'}[var]
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    for n in [0, 1]:
        condition = sample['JZW']==-1 if n==0 else sample['JZW']>= 0 if n==1 else sample['JZW']>=-2
        if not np.any(condition): continue
        variable  = sample[var][condition]; weights = sample['weights'][condition]
        min_val   = np.min(variable)
        bin_width = (np.max(variable) - min_val) / n_bins[n]
        bins      = list(np.arange(min_val, np.max(variable), bin_width)) + [np.max(variable)+1e-3]
        weights  *= 100/(np.sum(sample['weights'])) #weights *= 100/(np.sum(weights))
        if density:
            indices  = np.searchsorted(bins, variable, side='right')
            weights /= np.take(np.diff(bins), np.minimum(indices, len(bins)-1)-1)
        pylab.hist(variable, bins, histtype='step', weights=weights,
                   log=log, label=labels[n], lw=2, color=colors[n])
    #pylab.xlim(0, 300)
    pylab.xlim(0, 6000)
    #pylab.ylim(1e-4, 1)
    pylab.ylim(1e-3, 1e-1)
    #pylab.ylim(0, 0.016)
    pylab.ylim(1e-5, 1e1)
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    if not log: axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.tick_params(axis='both', which='major', labelsize=14)
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel(xlabel+' (GeV)', fontsize=24)
    plt.ylabel('Distribution density (%' + ('/GeV' if density else '') + ')', fontsize=24)
    plt.legend(loc='upper right', fontsize=18)
    file_name = output_dir+'/'+'var_distributions.png'
    print('Saving pt distributions to:', file_name, '\n'); plt.savefig(file_name)


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
    pylab.xlim(10, 25)
    plt.xticks(np.arange(10, 26, 5))
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('$p_t$ (scaler space)', fontsize=24)
    plt.ylabel('Distribution (%)', fontsize=24)
    plt.legend(loc='upper right', fontsize=18)
    file_name = output_dir+'/'+'pt_reconstruction.png'
    print('Saving pt reconstruction  to:', file_name); plt.savefig(file_name)


def loss_distributions(y_true, X_loss, metric, weights, output_dir, n_bins=100):
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    bin_width = np.max(X_loss)/n_bins; bins = np.arange(0, np.max(X_loss)+bin_width, bin_width)
    if metric == 'EMD': bin_width = 30/n_bins; bins = np.arange(0, 30+bin_width, bin_width)
    if metric == 'KLD': bin_width = 0.3/n_bins; bins = np.arange(0, 30+bin_width, bin_width)
    labels = [r'$t\bar{t}$ jets', 'QCD jets']; colors = ['tab:orange', 'tab:blue']
    if np.any(weights) == None: weights = np.array(len(y_true)*[1.])
    for n in set(y_true):
        hist_weights  = weights[y_true==n]
        hist_weights *= 100/np.sum(hist_weights)/bin_width
        pylab.hist(X_loss[y_true==n], bins, histtype='step', weights=hist_weights,
                   log=False, label=labels[n], color=colors[n], lw=2)
    if metric == 'JSD': pylab.xlim(0, 0.3)
    if metric == 'MSE': pylab.xlim(0, 0.15); pylab.xticks(np.arange(0,0.16,0.05))
    if metric == 'EMD': pylab.xlim(0, 25)
    if metric == 'KLD': pylab.xlim(0, 0.3)
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel(metric+' reconstruction loss', fontsize=24)
    plt.ylabel('Distribution density (%/'+metric+')', fontsize=24)
    plt.legend(loc='upper right', fontsize=18)
    file_name = output_dir+'/'+metric+'_loss.png'
    print('Saving loss distributions to:', file_name); plt.savefig(file_name)


def mass_correlation(y_true, X_loss, metric, X_mass, output_dir, n_bins=100):
    losses = []; masses = []
    for n in set(y_true):
        n_loss   = X_loss[y_true==n]
        n_mass   = X_mass[y_true==n]
        loss_min = np.min(n_loss); loss_max = np.max(n_loss); loss_step = (loss_max-loss_min)/n_bins
        if metric == 'EMD': loss_min = np.min(n_loss); loss_max = 30; loss_step = (loss_max-loss_min)/n_bins
        if metric == 'KLD': loss_min = np.min(n_loss); loss_max = 0.3; loss_step = (loss_max-loss_min)/n_bins
        losses  += [np.arange(loss_min, loss_max, loss_step)[:-2]]
        masses  += [[np.mean(n_mass[n_loss>loss]) for loss in losses[n]]]
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    if metric == 'JSD': pylab.xlim(0, 0.3)
    if metric == 'MSE': pylab.xlim(0, 0.15)
    if metric == 'EMD': pylab.xlim(0, 25)
    if metric == 'KLD': pylab.xlim(0, 0.3)
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Cut on '+metric+' reconstruction loss', fontsize=24)
    plt.ylabel('Mean jet mass (GeV)', fontsize=24)
    labels = [r'$t\bar{t}$ jets', 'QCD jets']; colors = ['tab:orange', 'tab:blue']
    for n in np.arange(np.max(y_true)+1):
        plt.plot(losses[n], masses[n], label=labels[n], color=colors[n], lw=2)
    plt.legend(loc='upper left', fontsize=18)
    file_name = output_dir+'/'+metric+'_correlation.png'
    print('Saving mass correlations  to:', file_name); plt.savefig(file_name)


def get_rates(X_true, X_pred, y_true, weights, metric, return_dict):
    X_loss      = loss_function(X_true, X_pred, metric)
    fpr, tpr, _ = metrics.roc_curve(y_true, X_loss, pos_label=0, sample_weight=weights)
    return_dict[metric] = {'fpr':fpr, 'tpr':tpr}
def ROC_rates(X_true, y_true, X_pred, weights, metrics_list):
    manager      = mp.Manager(); return_dict = manager.dict()
    arguments    = [(X_true, y_true, X_pred, weights, metric, return_dict) for metric in metrics_list]
    processes    = [mp.Process(target=get_rates, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    return {metric:return_dict[metric] for metric in metrics_list}
def ROC_curves(X_true, y_true, X_pred, weights, output_dir, metrics_list, wps):
    colors_dict  = {'JSD':'tab:blue', 'MSE':'tab:green' , 'EMD':'tab:orange',
                    'KLD':'tab:red' , 'X-S':'tab:purple', 'MAE':'tab:brown'}
    metrics_dict = ROC_rates(X_true, y_true, X_pred, weights, metrics_list)
    plt.figure(figsize=(11,15))
    plt.subplot(2, 1, 1); pylab.grid(True); axes = plt.gca()
    for metric in metrics_list:
        fpr, tpr = [metrics_dict[metric][key] for key in ['fpr','tpr']]
        len_0 = np.sum(fpr==0)
        #plt.text(0.96, 0.45-4*metrics_list.index(metric)/100, 'AUC: '+format(metrics.auc(fpr,tpr), '.3f'),
        plt.text(0.96, 0.47-4*metrics_list.index(metric)/100, 'AUC: '+format(metrics.auc(fpr,tpr), '.3f'),
                 {'color':colors_dict[metric], 'fontsize':12}, va='center', ha='right', transform=axes.transAxes)
        plt.plot(100*tpr[len_0:], 1/fpr[len_0:], label=metric, lw=2, color=colors_dict[metric])
    metrics_scores = [[metrics_dict[metric][key] for key in ['fpr','tpr']] for metric in metrics_list]
    scores = [np.max([1/fpr[np.argwhere(100*tpr >= val)[0]] for fpr,tpr in metrics_scores]) for val in wps]
    for n in np.arange(len(wps)):
        axes.axhline(scores[n], xmin=wps[n]/100, xmax=1, ls='--', linewidth=1., color='tab:blue')
        plt.text(100.4, scores[n], str(int(scores[n])), {'color':'tab:blue', 'fontsize':12}, va="center", ha="left")
        axes.axvline(wps[n], ymin=0, ymax=np.log(scores[n])/np.log(1e4), ls='--', linewidth=1., color='tab:blue')
        #plt.text(wps[n], 0.735, str(int(wps[n])), {'color':'tab:blue', 'fontsize':12}, va="center", ha="center")
    pylab.xlim(0,100)
    pylab.ylim(1,1e3)
    plt.yscale('log')
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    plt.yticks([10**int(n) for n in np.arange(0,5)], ['1','$10^1$','$10^2$','$10^3$','$10^4$'])
    plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.ylabel('$1/\epsilon_{\operatorname{bkg}}$', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper right', fontsize=16)
    plt.legend(loc='upper right', fontsize=16)
    plt.subplot(2, 1, 2); pylab.grid(True); axes = plt.gca()
    for metric in metrics_list:
        fpr, tpr = [metrics_dict[metric][key] for key in ['fpr','tpr']]
        len_0 = np.sum(fpr==0)
        plt.plot(100*tpr[len_0:], tpr[len_0:]/fpr[len_0:], label=metric, lw=2, color=colors_dict[metric])
    pylab.xlim(1,100)
    pylab.ylim(1,5)
    plt.xscale('log')
    plt.xticks([1,10,100], ['1','10', '100'])
    plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.ylabel('$G_{S/B}=\epsilon_{\operatorname{sig}}/\epsilon_{\operatorname{bkg}}$', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper right', fontsize=16)
    file_name = output_dir+'/'+'ROC.png'
    print('Saving ROC curves         to:', file_name); plt.savefig(file_name)
