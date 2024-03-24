import numpy             as np
import multiprocessing   as mp
import matplotlib.pyplot as plt
import os, sys, h5py, time, pickle, warnings, itertools
from threading  import Thread
from matplotlib import pylab, ticker, patches, colors as mcolors, font_manager
from sklearn    import metrics
from scipy      import spatial, interpolate, special
from functools  import partial
from utils      import make_discriminant, latent_loss, inverse_scaler, get_4v, bump_hunter
from utils      import n_constituents, jets_pt, get_idx, get_bins, bin_deco
import matplotlib#; matplotlib.use('Agg')


def plot_results(valid_data, sig_data, output_dir, apply_cuts, best_cut=None, disc='Autoencoder'):
    print((sig_data+': plotting performance results').upper())
    sample, y_true, X_loss = [valid_data[key] for key in ['sample','y_true','X_loss']]
    if   'top'  in sig_data: sig_label = 'Top'
    elif 'VZ'   in sig_data: sig_label = 'VZ'
    elif 'BSM'  in sig_data: sig_label = 'BSM'
    elif 'OoD'  in sig_data: sig_label = 'OoD'
    elif '2HDM' in sig_data: sig_label = '2HDM'
    else                   : sig_label = 'N.A.'

    best_cuts  = bump_scan(y_true, X_loss[disc], disc, sample, sig_label, output_dir)
    #best_cuts   = bump_scan_2d(y_true, X_loss, sample, sig_label, output_dir)
    arguments  = (y_true, X_loss, sample['weights'], disc, sig_label, best_cuts, output_dir)
    #arguments  = (y_true, X_loss, sample['weights'], disc, sig_label, best_cuts, output_dir, valid_data)
    processes  = [mp.Process(target=ROC_curves, args=arguments)]
    #arguments  = [(y_true, X_loss, sample, metrics, sig_label, best_cuts, output_dir)]
    #processes += [mp.Process(target=plot_correlations, args=arg) for arg in arguments]
    #arguments  = [(y_true, X_loss[disc], sample['weights'], output_dir, sig_label, best_cuts['cuts'], disc)
    #              for disc in best_cuts['cuts']]
    #processes += [mp.Process(target=plot_discriminant, args=arg) for arg in arguments]
    for job in processes: job.start()
    for job in processes: job.join()
    if apply_cuts == 'ON':
        generate_cuts(y_true, sample, X_loss[loss_metric], loss_metric, sig_label, output_dir)
    print()


def smoothing(x, y, sort=False):
    if sort: idx = np.argsort(x, kind='mergesort')
    else   : idx = np.arange(len(x))
    x, y = x[idx], np.maximum.accumulate(y[idx])
    idx = np.unique(y, return_index=True)[1]
    return x[idx], y[idx]


def binary_dics_eff(valid_data, n_idx_1=1000, n_idx_2=1000):
    global binary_rates; start_time = time.time()
    def get_indices(fpr, n_idx):
        #idx = np.int_(np.linspace(0, len(fpr)-1, min(int(n_idx),len(fpr))))
        eff_val = np.logspace(np.log10(np.min(fpr)), np.log10(1), num=n_idx)
        return np.minimum(np.searchsorted(fpr, eff_val, side='left'), len(fpr)-1)
    def binary_rates(y_true, weights, disc_1, fpr_1, tpr_1, thresholds_1, disc_2, n_idx_2, idx_1):
        selection = disc_1 >= thresholds_1[idx_1]
        fpr_2, tpr_2, _ = get_rates(y_true[selection], disc_2[selection], weights[selection])
        idx_2 = get_indices(fpr_2, n_idx_2)
        fpr, tpr = fpr_1[idx_1]*fpr_2[idx_2], tpr_1[idx_1]*tpr_2[idx_2]
        indices = np.r_[np.where(np.diff(fpr))[0], len(fpr)-1]
        return list(zip(fpr[indices], tpr[indices]))
    valid_sample = valid_data['sample']; y_true = valid_data['y_true']
    disc_1, disc_2 = valid_data['X_loss']['Autoencoder'], valid_data['X_loss']['Discriminator']
    fpr_1, tpr_1, thresholds_1 = get_rates(y_true, disc_1, valid_sample['weights'])
    fpr_2, tpr_2, thresholds_2 = get_rates(y_true, disc_2, valid_sample['weights'])
    idx_1 = get_indices(fpr_1, n_idx_1)
    with mp.Pool() as pool:
        func_args = (y_true, valid_sample['weights'], disc_1, fpr_1, tpr_1, thresholds_1, disc_2, n_idx_2)
        eff_rates = pool.map(partial(binary_rates, *func_args), idx_1)
    fpr_3, tpr_3 = [np.array(n) for n in zip(*np.concatenate(eff_rates))]
    fpr_3, tpr_3 = smoothing(fpr_3, tpr_3, sort=True)
    return {'Autoencoder':(fpr_1,tpr_1), 'Discriminator':(fpr_2,tpr_2), 'Auto+Disc':(fpr_3,tpr_3)}


def generate_cuts(y_true, sample, X_loss, loss_metric, sig_label, output_dir, cut_types=['bkg_eff','gain']):
    print('\nAPPLYING CUTS ON SAMPLE:')
    def plot_suppression(sample, sig_label, X_loss, positive_rates, bkg_eff=None, file_name=None):
        cut_sample = make_cut(y_true, X_loss, sample, positive_rates, loss_metric, cut_type, bkg_eff)
        if len(cut_sample['weights']) > 0:
            if file_name is None: file_name  = 'bkg_suppression/bkg_eff_' + format(bkg_eff,'1.0e')
            sample_distributions([sample,cut_sample], sig_label, output_dir, file_name)
    if not os.path.isdir(output_dir+'/bkg_suppression'): os.mkdir(output_dir+'/bkg_suppression')
    positive_rates = get_rates(y_true, X_loss, sample['weights'])
    for cut_type in cut_types:
        if cut_type in ['bkg_eff']:
            processes = [mp.Process(target=plot_suppression, args=(sample, sig_label, X_loss, positive_rates,
                         bkg_eff)) for bkg_eff in [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 2e1, 5e1]]
            for job in processes: job.start()
            for job in processes: job.join()
        if cut_type in ['gain','sigma']:
            file_name = 'bkg_suppression/best_' + cut_type ; print()
            plot_suppression(sample, sig_label, X_loss, positive_rates, file_name=file_name)


def tSNE(y_true, X_true, model, output_dir, file_name='tSNE_scatter.pkl'):
    if not os.path.isfile(output_dir+'/'+file_name):
        from sklearn.manifold import TSNE
        z_mean, _, _ = model.encoder(X_true)
        embedding    = TSNE(n_jobs=-1, random_state=0, perplexity=30, learning_rate=100, verbose=1)
        z_embedded   = embedding.fit_transform(z_mean)
        pickle.dump(z_embedded, open(output_dir+'/'+file_name,'wb'), protocol=4)
    else:
        z_embedded = pickle.load(open(output_dir+'/'+file_name,'rb'))
    plt.figure(figsize=(12,8)); pylab.grid(True); ax = plt.gca()
    labels = [r'$t\bar{t}$', 'QCD']; colors = ['tab:orange', 'tab:blue']
    for n in set(y_true):
        x = z_embedded[:,0][y_true==n]
        y = z_embedded[:,1][y_true==n]
        plt.scatter(x, y, color=colors[n], s=10, label=labels[n], alpha=0.1)
    pylab.xlim(-1e-4, 1e-4); pylab.ylim(-1e-4, 1e-4)
    leg = plt.legend(loc='upper right', fontsize=18)
    for lh in leg.legendHandles: lh.set_alpha(1)
    file_name = output_dir+'/'+'tSNE_scatter.png'
    print('Saving tSNE 2D-embedding to:', file_name); plt.savefig(file_name)


def plot_4v_distributions(output_dir, scaler, n_dims=3, normalize=True):
    from utils import apply_scaler
    data_path = '/opt/tmp/godin/AD_data'
    file_dict = {
                 'top (old)':'formatted_converted_20210430_ttbar_allhad_pT_450_1200_nevents_1M.h5'             ,
                 'top (new)':'formatted_converted_20210430_ttbar_allhad_pT_450_1200_nevents_1M_alltransform.h5',
                 }
    plt.figure(figsize=(13,8)); pylab.grid(True); ax = plt.gca()
    for label in file_dict:
        data = h5py.File(data_path+'/'+file_dict[label],"r")
        jets = np.float32(data['constituents'][:900000])
        if n_dims == 3:
            """ Using (px, py, px) instead of (E, px, py, px) """
            shape = jets.shape
            jets  = np.reshape(jets        , (-1,shape[1]//4,4))
            jets  = np.reshape(jets[...,1:], (shape[0],-1)     )
        #jets = apply_scaler(jets, n_dims, scaler)
        jets = np.reshape(jets, (-1,n_dims))
        px, py, pz = [jets[:,n] for n in range(jets.shape[-1])]
        E = np.sqrt(px**2 + py**2 + pz**2)
        weights = np.ones_like(px, dtype=np.float32)
        if normalize: weights = weights * 100/np.sum(weights)
        bins = np.linspace(-200,1000,200)
        pylab.hist(px, bins=bins, histtype='step', weights=weights, lw=2, label=label, log=True)
    pylab.xlim(-200, 1000)
    pylab.ylim(1e-6, 1e2)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('$p_x$', fontsize=24)
    plt.ylabel('Distribution'+(' (%)' if normalize else '' ), fontsize=24)
    plt.legend(loc='upper right', ncol=1, fontsize=18)
    file_name = output_dir+'/'+'px_distribution.png'
    print('Saving 4v distribution to:', file_name); plt.savefig(file_name)


def plot_mean_pt(output_dir):
    data_path = '/opt/tmp/godin/AD_data'
    file_dict = {
                 'QCD (old)':'formatted_converted_20210629_QCDjj_pT_450_1200_nevents_10M.h5'        ,
                 'top (old)':'formatted_converted_20210430_ttbar_allhad_pT_450_1200_nevents_1M.h5'  ,
                 'top (new)':'formatted_converted_20211213_ttbar_allhad_pT_450_1200_nevents_10M_fixed.h5',
                 'H-OoD'    :'H_HpHm_generation_merged_with_masses_20_40_60_80_reformatted_nghia.h5',
                 }
    plt.figure(figsize=(13,8)); pylab.grid(True); ax = plt.gca()
    data    = h5py.File(data_path+'/'+file_dict['top (old)'],"r")
    n_const = n_constituents(data['constituents'])
    pt      =        jets_pt(data['constituents'])
    n_list  = np.linspace(10,100,10, dtype=int)
    for n in n_list:
        label = 'n_const $\leqslant$ '+str(n)
        plt.plot(np.arange(1,n+1), np.mean(pt[n_const<=n][:,:n], axis=0), label=label, lw=2)
    pylab.xlim(0, 100)
    pylab.ylim(100, 600)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Number of Constituents', fontsize=24)
    plt.ylabel('Mean $p_t$ (GeV)', fontsize=24)
    plt.legend(loc='lower right', ncol=2, fontsize=16)
    file_name = output_dir+'/'+'mean_pt.png'
    print('Saving mean pt plots to:', file_name); plt.savefig(file_name)


def plot_constituents(output_dir, normalize=True, log=True):
    data_path = '/opt/tmp/godin/AD_data'
    file_dict = {
                 'QCD (old)':'formatted_converted_20210629_QCDjj_pT_450_1200_nevents_10M.h5'        ,
                 'top (old)':'formatted_converted_20210430_ttbar_allhad_pT_450_1200_nevents_1M.h5'  ,
                 'top (new)':'formatted_converted_20211213_ttbar_allhad_pT_450_1200_nevents_10M_fixed.h5',
                 'H-OoD'    :'H_HpHm_generation_merged_with_masses_20_40_60_80_reformatted_nghia.h5',
                 }
    plt.figure(figsize=(13,8)); pylab.grid(True); ax = plt.gca()
    histos = []
    for label in file_dict:
        data    = h5py.File(data_path+'/'+file_dict[label],"r")
        n_const = n_constituents(data['constituents'])
        bins    = np.arange(-0.5,np.max(n_const)+1)
        weights = np.ones_like(n_const, dtype=np.float32)
        if normalize: weights = weights * 100/np.sum(weights)
        histos += [pylab.hist(n_const, bins=bins, histtype='step', weights=weights, lw=2, label=label)]
    pylab.xlim(-0.5, 100.5)
    if log: pylab.ylim(1e-4, 10); plt.yscale('log')
    #else  : pylab.ylim(0, np.ceil(np.max([np.max(h[0]) for h in histos])))
    else  : pylab.ylim(0, 2.5)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Number of Constituents', fontsize=24)
    plt.ylabel('Distribution'+(' (%)' if normalize else '' ), fontsize=24)
    plt.legend(loc='lower right' if log else 'upper left' , ncol=1, fontsize=18)
    file_name = output_dir+'/'+'n_constituents.png'
    print('Saving constituents distributions to:', file_name); plt.savefig(file_name)


def sample_distributions(sample, sig_label, output_dir, name, bin_sizes={'m':5,'pt':10}, weight_type='None'):
    processes = [mp.Process(target=plot_distributions, args=(sample, sig_label, var, bin_sizes,
                 output_dir, name+'_'+var+'.png', weight_type)) for var in ['m','pt']] #,'m_over_pt']]
    for job in processes: job.start()
    for job in processes: job.join()


def get_rates(y_true, X_loss, weights, metric=None, return_dict=None):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, X_loss, pos_label=0, sample_weight=weights,
                                             drop_intermediate=True)
    fpr, tpr, thresholds = fpr[fpr!=0], tpr[fpr!=0], thresholds[fpr!=0]
    if return_dict is None: return                fpr, tpr, thresholds
    else                  : return_dict[metric] = fpr, tpr, thresholds


def ROC_rates(y_true, X_losses, weights, metrics_list):
    manager      = mp.Manager(); return_dict = manager.dict()
    arguments    = [(y_true, X_losses[metric], weights, metric, return_dict) for metric in metrics_list]
    processes    = [mp.Process(target=get_rates, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    return {metric:return_dict[metric] for metric in metrics_list}


def best_threshold(y_true, positive_rates, weights, cut_type, min_tpr=1):
    fpr, tpr, thresholds = positive_rates
    fpr, tpr, thresholds = fpr[tpr>min_tpr], tpr[tpr>min_tpr], thresholds[tpr>min_tpr]
    if cut_type=='gain':
        cut_values = tpr/fpr
        factor = 1
    if cut_type=='sigma':
        cut_values = tpr/np.sqrt(fpr)
        n_sig  = np.sum(weights[y_true==0])
        n_bkg  = np.sum(weights[y_true==1])
        factor = n_sig/np.sqrt(n_bkg)/10
    cut_index = np.argmax(cut_values)
    return thresholds[cut_index], cut_values[cut_index] * factor


def make_cut(y_true, X_loss, sample, positive_rates, metric, cut_type, bkg_eff=None):
    if bkg_eff is None:
        loss_cut, loss_val = best_threshold(y_true, positive_rates, sample['weights'], cut_type)
        print('Best', metric, 'cut on '+format(cut_type,'4s')+'  --> ', metric, '>=', end= ' ')
        print(format(loss_cut, '.3f')+' / best '+format(cut_type,'4s'), '=', format(loss_val, '>4.2f'))
    else:
        fpr, tpr, thresholds = positive_rates
        cut_idx  = np.argmin(abs(fpr-bkg_eff))
        loss_cut = thresholds[cut_idx]
    return {key:sample[key][X_loss>loss_cut] for key in sample}


def bump_scan(y_true, X_loss, disc, sample, sig_label, output_dir, n_cuts=100, eff_type='bkg'):
    def logit(x, delta=1e-16):
        x = np.clip(np.float64(x), delta, 1-delta)
        return np.log10(x) - np.log10(1-x)
    def inverse_logit(x): return 1/(1+10**(-x))
    font_manager._get_font.cache_clear()
    fpr, tpr, thresholds = get_rates(y_true, X_loss, sample['weights'])
    cut_thresholds = thresholds
    if eff_type == 'sig':
        cut_eff = tpr
        x_min, x_max = 10*np.floor(tpr[0]/0.1), 1
        eff_val = np.linspace(tpr[0], x_max, n_cuts)
    elif eff_type == 'bkg':
        cut_eff = fpr
        x_min, x_max = 10**np.ceil(np.log10(np.min(fpr))), 1
        #eff_val = np.append(100*inverse_logit(np.linspace(logit(x_min/100),-logit(x_min/100),n_cuts)), 100)
        eff_val = np.logspace(np.log10(x_min), np.log10(x_max), num=n_cuts)
    idx = np.minimum(np.searchsorted(cut_eff, eff_val, side='right'), len(cut_eff)-1)
    sample = {key:sample[key] for key in ['JZW','m','pt','weights']}
    global get_sigma
    def get_sigma(sample, X_loss, cut_thresholds, idx):
        cut_sample = {key:sample[key][X_loss>cut_thresholds[idx]] for key in sample}
        return bump_hunter(cut_sample)
        #try   : return bump_hunter(cut_sample)
        #except: return np.nan, np.nan
    with mp.Pool() as pool:
        sigma = np.array(pool.map(partial(get_sigma, sample, X_loss, cut_thresholds), idx))
    loc_sigma, max_sigma = [np.array(array) for array in zip(*sigma)]
    cut_filter = np.logical_and(np.isfinite(loc_sigma), np.isfinite(max_sigma))
    cut_thresholds, cut_eff   = np.take(cut_thresholds,idx), np.take(cut_eff,idx)
    cut_thresholds, cut_eff   = cut_thresholds[cut_filter],   cut_eff[cut_filter]
    loc_sigma     , max_sigma =      loc_sigma[cut_filter], max_sigma[cut_filter]
    if len(cut_thresholds) == 0: return
    """ Printing bkg suppression and bum hunting plots at maximum significance cut """
    opt_max_sigma = np.max(max_sigma)
    loc_sigma, max_sigma = loc_sigma/loc_sigma[-1], max_sigma/max_sigma[-1]
    opt_sigma = loc_sigma
    best_cut = {'cuts':{disc:cut_thresholds[np.argmax(opt_sigma)]}}
    best_cut['sig_eff'] = tpr[np.argmin(np.abs(thresholds-best_cut['cuts'][disc]))]
    best_cut['bkg_eff'] = fpr[np.argmin(np.abs(thresholds-best_cut['cuts'][disc]))]
    #best_cut['bkg_eff'] = cut_eff[np.argmax(opt_sigma)]
    cut_sample = {key:sample[key][X_loss>best_cut['cuts'][disc]] for key in sample}
    arguments  = (cut_eff, eff_type, loc_sigma, max_sigma, output_dir)
    processes  = [mp.Process(target=plot_significance, args=arguments)]
    arguments  = (    sample, output_dir+'/BH_uncut.png', sig_label, opt_max_sigma)
    processes += [mp.Process(target=bump_hunter, args=arguments)]
    arguments  = (cut_sample, output_dir+'/BH_best.png' , sig_label, opt_max_sigma)
    processes += [mp.Process(target=bump_hunter, args=arguments)]
    arguments  = ([sample,cut_sample], sig_label, output_dir, 'BH_bkg_supp', {'m':5,'pt':10})
    processes += [mp.Process(target=sample_distributions, args=arguments)]
    for job in processes: job.start()
    for job in processes: job.join()
    return best_cut


def bump_scan_2d(y_true, X_loss, sample, sig_label, output_dir, n_cuts=100, eff_type='bkg'):
    def get_fpr(y_true, X_loss, weights, n_cuts):
        fpr, _, thresholds = get_rates(y_true, X_loss, weights)
        x_min = np.min(fpr)
        #x_min = 10**np.ceil(np.log10(x_min))
        eff_val = np.logspace(np.log10(x_min), np.log10(1), num=n_cuts)
        idx = np.minimum(np.searchsorted(fpr, eff_val, side='left'), len(fpr)-1)
        return thresholds, idx
    global get_sigma
    def get_sigma(y_true, sample, X_loss_1, X_loss_2, thresholds_1, thresholds_2, idx_tuples, index):
        if (index+1)%10 == 0:
            print('2D Discriminant Search: '+format(index+1,'5.0f')+'/'+str(len(idx_tuples))+' cuts'+'\033[A')
        idx_1, idx_2 = idx_tuples[index]
        cuts = np.logical_and(X_loss_1>=thresholds_1[idx_1], X_loss_2>=thresholds_2[idx_2])
        cut_sample = {key:sample[key][cuts] for key in ['JZW','m','weights']}
        weights    = cut_sample['weights']
        tpr, fpr   = np.sum(weights[y_true[cuts]==0]), np.sum(weights[y_true[cuts]==1])
        #if np.sum(cuts) < 1e3: return tpr, fpr, np.nan, np.nan
        try   : loc_sigma, max_sigma = bump_hunter(cut_sample)
        except: loc_sigma, max_sigma = np.nan, np.nan
        return tpr, fpr, loc_sigma, max_sigma
    def interpolation(tpr, fpr, value):
        idx1 = np.where(fpr<=value)[0][-1]
        idx2 = np.where(fpr>=value)[0][ 0]
        M = (tpr[idx2]-tpr[idx1])/(fpr[idx2]-fpr[idx1]) if fpr[idx2]!=fpr[idx1] else 0
        return tpr[idx1] + M*(value-fpr[idx1])
    def plot_ROC_curves(tpr, fpr, sig_label, output_dir, best_fpr):
        fpr, tpr = fpr[fpr!=0], tpr[fpr!=0]
        fpr, tpr = smoothing(fpr, tpr, sort=True)
        best_cuts = {'sig_eff':interpolation(tpr, fpr, best_fpr), 'bkg_eff':best_fpr}
        ROC_curves(y_true=None, X_loss=None, weights=None, disc=None, sig_label=sig_label,
                   best_cut=best_cuts, output_dir=output_dir, valid_data={'Auto+Disc':(fpr,tpr)})
        return best_cuts
    font_manager._get_font.cache_clear()
    sample = {key:sample[key] for key in ['JZW','m','pt','weights']}

    disc = ['Autoencoder', 'Discriminator']
    X_loss_1, X_loss_2  = [X_loss[name] for name in disc]
    thresholds_1, idx_1 = get_fpr(y_true, X_loss_1, sample['weights'], n_cuts)
    thresholds_2, idx_2 = get_fpr(y_true, X_loss_2, sample['weights'], n_cuts)
    idx_tuples = np.array(list(itertools.product(idx_1,idx_2)))
    with mp.Pool() as pool:
        func_args = (y_true, sample, X_loss_1, X_loss_2, thresholds_1, thresholds_2, idx_tuples)
        sigma = np.array(pool.map(partial(get_sigma, *func_args), np.arange(len(idx_tuples))))
    tpr, fpr, loc_sigma, max_sigma = [np.array(array) for array in zip(*sigma)]
    tpr, fpr = tpr/np.sum(sample['weights'][y_true==0]), fpr/np.sum(sample['weights'][y_true==1])

    best_cuts = plot_ROC_curves(tpr, fpr, sig_label, output_dir, fpr[np.argmax(loc_sigma)])
    cut_filter = np.logical_and(np.isfinite(loc_sigma), np.isfinite(max_sigma))
    loc_sigma, max_sigma = loc_sigma [cut_filter], max_sigma[cut_filter]
    idx_tuples, tpr, fpr = idx_tuples[cut_filter], tpr[cut_filter], fpr[cut_filter]
    if len(idx_tuples) == 0: return
    best_tuple   = np.argmax(loc_sigma) #= np.argmax(max_sigma)
    idx_1, idx_2 = idx_tuples[best_tuple]
    best_cuts['cuts'] = dict(zip(disc, [thresholds_1[idx_1], thresholds_2[idx_2]]))
    cuts = np.logical_and(X_loss_1>=best_cuts['cuts'][disc[0]], X_loss_2>=best_cuts['cuts'][disc[1]])
    cut_sample = {key:sample[key][cuts] for key in sample}
    """ Printing bkg suppression and bum hunting plots at maximum significance cut """
    arguments  = (fpr, eff_type, loc_sigma/loc_sigma[-1], max_sigma/max_sigma[-1], output_dir)
    processes  = [mp.Process(target=plot_significance, args=arguments)]
    arguments  = (    sample, output_dir+'/BH_uncut.png', sig_label, max_sigma[best_tuple])
    processes += [mp.Process(target=bump_hunter, args=arguments)]
    arguments  = (cut_sample, output_dir+'/BH_best.png' , sig_label, max_sigma[best_tuple])
    processes += [mp.Process(target=bump_hunter, args=arguments)]
    arguments  = ([sample,cut_sample], sig_label, output_dir, 'BH_bkg_supp', {'m':5,'pt':10})
    processes += [mp.Process(target=sample_distributions, args=arguments)]
    for job in processes: job.start()
    for job in processes: job.join()
    return best_cuts


def plot_significance(fpr, eff_type, loc_sigma, max_sigma, output_dir):
    font_manager._get_font.cache_clear()
    def smooth_sigma(fpr, sigma):
        idx = np.argmax(sigma)
        fpr_1, sigma_1 = smoothing(fpr[:idx]      , sigma[:idx]      )
        fpr_2, sigma_2 = smoothing(fpr[idx:][::-1], sigma[idx:][::-1])
        return np.concatenate([fpr_1,fpr_2[::-1]]), np.concatenate([sigma_1,sigma_2[::-1]])
    plt.figure(figsize=(12,8)); pylab.grid(False); axes = plt.gca()
    indices = np.argsort(fpr, kind='mergesort')
    loc_sigma, max_sigma, fpr = loc_sigma[indices], max_sigma[indices], fpr[indices]
    #fpr_smooth, sigma_smooth = smooth_sigma(fpr, max_sigma)
    #plt.plot(fpr_smooth, sigma_smooth, label='Max Bin $\sigma$', color='silver', lw=4)
    fpr_smooth, loc_sigma = smooth_sigma(fpr, loc_sigma)
    plt.plot(fpr_smooth, loc_sigma, label='Local $\sigma$', color='gray', lw=4)
    opt_sigma = loc_sigma #np.append(loc_sigma, max_sigma)
    if   np.max(opt_sigma)-np.min(opt_sigma) >= 2: factor, val_pre = (1.0, '.2f')
    elif np.max(opt_sigma)-np.min(opt_sigma) >= 1: factor, val_pre = (0.2, '.2f')
    else                                         : factor, val_pre = (0.2, '.2f')
    x_min, x_max = min(1e-2, 10**np.floor(np.log10(fpr_smooth[np.argmax(loc_sigma)]))), 1
    pylab.xlim(x_min, x_max)
    y_min = factor*np.floor(np.min(opt_sigma[fpr_smooth>=x_min])/factor)
    y_max = factor*np.ceil (np.max(opt_sigma)                   /factor)
    pylab.ylim(y_min, y_max)
    max_val, max_eff = np.max(opt_sigma), fpr_smooth[np.argmax(opt_sigma)]
    plt.text(1.004, (max_val-axes.get_ylim()[0])/np.diff(axes.get_ylim()), format(max_val,val_pre),
             {'color':'black','fontsize':15}, va="center", ha="left", transform=axes.transAxes)
    if max_eff > 1.1*x_min:
        plt.text((np.log10(max_eff)-np.log10(x_min))/(np.log10(x_max)-np.log10(x_min)), 1.018, format(max_eff,'.3f'),
                 {'color':'black','fontsize':15}, va='center', ha='center', transform=axes.transAxes)
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.tick_params(which='minor', direction='in', length=4, width=1, colors='black',
                     bottom=True, top=True, left=True, right=True)
    axes.tick_params(which='major', direction='in', length=7, width=3, colors='black',
                     bottom=True, top=True, left=True, right=True)
    axes.tick_params(axis='x', pad=8, labelsize=22)
    axes.tick_params(axis='y', pad=8, labelsize=22)
    for axis in ['top', 'bottom', 'left', 'right']:
        axes.spines[axis].set_linewidth(3)
        axes.spines[axis].set_color('black')
    if eff_type == 'sig':
        xmin = (max_eff-x_min)/(x_max-x_min)
        plt.xlabel('$\epsilon_{\operatorname{sig}}$', fontsize=32, labelpad=5)
    elif eff_type == 'bkg':
        plt.xlabel('$\epsilon_{\operatorname{bkg}}$', fontsize=32, labelpad=5)
        plt.xscale('log')
        pos    = [10**int(n) for n in range(int(np.log10(x_min)),1)]
        labels = [('$10^{'+str(n)+'}$' if n<=-1 else int(10**n))for n in range(int(np.log10(x_min)),1)]
        plt.xticks(pos, labels)
        xmin = (np.log10(max_eff)-np.log10(x_min))/(np.log10(x_max)-np.log10(x_min))
    axes.axhline(max_val, xmin=xmin, xmax=1, ls='--', linewidth=1.5, color='tab:gray')
    if max_eff > 1.1*x_min:
        axes.axvline(max_eff, ymin=(max_val-y_min)/(y_max-y_min), ymax=1, ls='--', linewidth=1.5, color='tab:gray')
    axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    #plt.ylabel('$\sigma_{\operatorname{cut}}/\sigma_{\operatorname{uncut}}$', fontsize=32, labelpad=6)
    plt.ylabel('$\sigma_{\operatorname{ratio}}$', fontsize=32, labelpad=6)
    plt.legend(loc='best', fontsize=20, facecolor='ghostwhite', frameon=False, framealpha=1)
    plt.subplots_adjust(left=0.11, top=0.97, bottom=0.15, right=0.96)
    file_name = output_dir+'/'+'BH_sigma.png'
    print('Saving BH significance   to:', file_name); plt.savefig(file_name)


def plot_bump(data, data_weights, y_true, bins, bin_sigma, loc_sigma, max_sigma,
              bump_range, m_range, gaussian_par, sig_label, filename, log=False):
    font_manager._get_font.cache_clear()
    log = log or np.mean(bump_range)>400
    def Gaussian(x, A, B, C): return A*np.exp(-(x-B)**2/(2*C**2))
    labels     = {0:'QCD', 1:sig_label}
    color_dict = {labels[0]:'tab:blue', labels[1]:'tab:orange'}
    fig, (ax1,ax2) = plt.subplots(figsize=(12,8), ncols=1, nrows=2, sharex=True,
                                          gridspec_kw={'height_ratios':[3,1]})
    data_weights  = 100*data_weights/(np.sum(data_weights))
    indices       = np.searchsorted(bins, data, side='right')
    data_weights  = data_weights / np.take(np.diff(bins), np.minimum(indices, len(bins)-1)-1)
    bkg_data, bkg_weights = data[y_true==1], data_weights[y_true==1]
    sig_data, sig_weights = data[y_true==0], data_weights[y_true==0]
    samples = [bkg_data   , sig_data   ]
    weights = [bkg_weights, sig_weights]
    ax1.hist(samples, bins, weights=weights, histtype='barstacked', log=log, lw=3, alpha=0.2, clip_on=True,
             label=[labels[0],labels[1]], color=[color_dict[key] for key in [labels[0],labels[1]]], zorder=0)
    for n in [1,0]:
        merged_samples = np.concatenate([samples[n] for n in range(n+1)])
        merged_weights = np.concatenate([weights[n] for n in range(n+1)])
        H = ax1.hist(merged_samples, bins=bins, weights=merged_weights, histtype='step', log=log,
                     lw=3, fill=False, zorder=0, edgecolor=color_dict[labels[n]], alpha=1, clip_on=True)
        if n == 1:
            vlines_y = H[0][np.argmin(np.abs(bump_range[0]-bins))], H[0][np.argmin(np.abs(bump_range[1]-bins))]
    ax1.vlines(bump_range, 0, vlines_y, colors='tab:red', ls=(0,(4,1)), lw=2, label='Bump', zorder=0)
    ax2.hist(bins[:-1], bins, histtype='step', weights=bin_sigma, lw=3, fill=True, clip_on=True, zorder=0,
             edgecolor=mcolors.to_rgba('darkgray')[:-1]+(1,),facecolor=mcolors.to_rgba('gray')[:-1]+(0.2,))
    array = np.linspace(m_range[0], m_range[1], num=1000)
    if gaussian_par is not None:
        A_approx, B_approx, C_approx, height, mean, std = gaussian_par
        ax2.plot(array, A_approx*Gaussian((array-B_approx)/C_approx, height, mean, std),
                 color='dimgray', lw=2, zorder=10)
    ax2.axvline(bump_range[0], 0, 1, color='tab:red', ls=(0,(4,1)), lw=2, zorder=0)
    ax2.axvline(bump_range[1], 0, 1, color='tab:red', ls=(0,(4,1)), lw=2, zorder=0)
    handles, labels = ax1.get_legend_handles_labels()
    handles = [patches.Patch(edgecolor=h.get_facecolor()[:-1]+(1,), facecolor=h.get_facecolor(),
                             fill=True, lw=3) for h in handles[:-1]] + [handles[-1]]
    ax1.legend(handles, labels, loc='upper right', frameon=False, fontsize=20)
    for AX in [ax1, ax2]:
        AX.tick_params(which='minor', direction='in', length=5, width=1.5, colors='black',
                         bottom=True, top=True, left=True, right=True, zorder=20)
        AX.tick_params(which='major', direction='in', length=7, width=3, colors='black',
                         bottom=True, top=True, left=True, right=True, zorder=20)
        AX.tick_params(axis="x", pad=5, labelsize=22)
        AX.tick_params(axis="y", pad=5, labelsize=22)
        for axis in ['top', 'bottom', 'left', 'right']:
            AX.spines[axis].set_linewidth(3)
            AX.spines[axis].set_zorder(20)
    ax1.set_ylabel('Probability Density (%)', loc='center', fontsize=26, labelpad=5)
    ax1.set_xlim(m_range)
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    x_max = min(m_range[1], np.sum(bump_range) )
    x_max = 100*np.ceil(x_max/100)
    ax1.set_xlim(m_range[0],x_max)
    if log:
        y_min = np.floor(np.log10(np.min(H[0])))
        ax1.set_ylim(10**y_min, None)
    else:
        y_max = np.ceil(10*np.max(H[0]))/10
        ax1.set_ylim(0, y_max)
    ax2.set_xlabel('$m\,$(GeV)', loc='center', fontsize=30, labelpad=5)
    ax2.set_ylabel('$\sigma$'  , loc='center', fontsize=30, labelpad=5)
    if max_sigma <= 5:
        factor = 1 ; ax2.set_yticks([0,1,2,3,4,5])
    elif max_sigma <= 10:
        factor = 2 ; ax2.set_yticks([0,2,4,6,8,10])
    elif max_sigma <= 25:
        factor = 5 ; ax2.set_yticks([0,5,10,15,20,25])
    else:
        factor = 10
    y_max = factor*np.ceil(max_sigma/factor)
    ax2.set_ylim(0, y_max)
    plt.text(bump_range[1]/x_max+0.02, np.max(bin_sigma)/y_max-(0.05 if 'best' in filename else -0.20),
             '$\,\sigma_{\operatorname{local}}\,\,=$'+format(loc_sigma,'.1f'),
             {'color':'black','fontsize':14}, va='top', ha='left', transform=ax2.transAxes)
    plt.text(bump_range[1]/x_max+0.02, np.max(bin_sigma)/y_max-(0.22 if 'best' in filename else -0.03),
             '$m_{\operatorname{bump}}\!=$'+format(mean*C_approx+B_approx,'.0f')+'$\,$GeV',
             {'color':'black','fontsize':14}, va='top', ha='left', transform=ax2.transAxes)
    fig.align_labels()
    fig.subplots_adjust(left=0.11, right=0.96, bottom=0.15, top=0.97, hspace=0.08)
    print('Saving bump hunting plot to:', filename)
    plt.savefig(filename, bbox_inches="tight")


def get_JSD(P, Q, delta=1e-32):
    def KLD(P, Q, base=2):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.nan_to_num(P*np.log(P/Q)/np.log(base))
    P, Q = np.maximum(np.float64(P), delta), np.maximum(np.float64(Q), delta)
    P, Q = P/np.sum(P), Q/np.sum(Q)
    M = (P + Q) / 2
    return np.sqrt( np.sum((KLD(P,M)+KLD(Q,M)))/2 )


def get_distance(y_true, X_losses, sample, metric, eff_type, truth, var, distance_dict, n_cuts=200):
    X_loss, X_var, weights = X_losses[metric], sample[var], sample['weights']
    fpr, tpr, thresholds = get_rates(y_true, X_loss, weights)
    if eff_type == 'sig':
        eff = tpr
        x_min, x_max = 10*np.floor(tpr[0]/10), 100
        eff_val = np.linspace(tpr[0], x_max, n_cuts)
    elif eff_type == 'bkg':
        eff = fpr
        x_min, x_max = fpr[0], 100
        eff_val = np.logspace(np.log10(x_min), np.log10(x_max), n_cuts)
    idx = np.minimum(np.searchsorted(eff, eff_val, side='right'), len(eff)-1)
    thresholds, tpr, fpr = np.take(thresholds, idx), np.take(tpr, idx), np.take(fpr, idx)
    X_loss =  X_loss [y_true==truth]
    X_var  =  X_var  [y_true==truth]
    weights = weights[y_true==truth]
    KSD = []; JSD = []; sig_eff = []; bkg_eff = []
    if var == 'm':
        bins, var_range = 100, (0,1000)
    if var == 'pt':
        bins, var_range = 100, (0,2000)
    P = np.histogram(X_var, bins=bins, range=var_range, weights=weights)[0]
    for n in range(len(thresholds)):
        X_var_cut   = X_var  [X_loss>=thresholds[n]]
        weights_cut = weights[X_loss>=thresholds[n]]
        if len(X_var_cut) != 0:
            Q = np.histogram(X_var_cut, bins=bins, range=var_range, weights=weights_cut)[0]
            #KSD += [ KS_distance(masses, masses_cut, weights, weights_cut) ]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                #JSD += [ spatial.distance.jensenshannon(P, Q, base=2) ]
                JSD += [ get_JSD(P, Q) ]
            sig_eff += [ tpr[n] ]
            bkg_eff += [ fpr[n] ]
    distance_dict[(metric,truth,var)] = KSD, JSD, sig_eff, bkg_eff


def plot_correlations(y_true, X_losses, sample, metrics_list, sig_label, best_cut,
                      output_dir, eff_type='bkg', distance_type='JSD', loss_metric=None):
    font_manager._get_font.cache_clear()
    def make_plot(distance_type, loss_metric):
        x_min = 100 ; y_max = 0
        for var_tuple in itertools.product(metrics_list, [1,0], ['m','pt']):
            metric, truth, var = var_tuple
            #label = str(metric) + ' (' + ('sig' if truth==0 else 'bkg') +')'
            label  = sig_label if truth==0 else 'QCD$\;\;\,$'
            label += ' $(m)$' if var=='m' else ' $(p_T)$' if var=='pt' else ''
            color  = 'tab:orange' if truth==0 else 'tab:blue'
            KSD, JSD, sig_eff, bkg_eff = distance_dict[(metric,truth,var)]
            distance = JSD if distance_type=='JSD' else KSD
            ls, alpha = ('-',1) if var=='m' else ('-',0.5)
            if   eff_type == 'sig':
                plt.plot(sig_eff, distance, label=label, color=color_dict[metric],
                         lw=2, ls=ls, zorder=1, alpha=alpha)
            elif eff_type == 'bkg':
                x_min = min(x_min, np.min(bkg_eff))
                plt.plot(bkg_eff, distance, label=label, color=color, lw=4, ls=ls, zorder=1, alpha=alpha)
            y_max = max(y_max, np.max(distance))
            if metric == loss_metric and truth == 1 and eff_type=='sig':
                P = []; labels = []
                for eff, marker in zip([1e-3,1e-2,1e-1,1e0,1e1], ['o','^','v','s','D']):
                    idx = (np.abs(np.array(bkg_eff) - eff)).argmin()
                    labels += ['$\epsilon_{\operatorname{bkg}}$: '+'$10^{'+str(int(np.log10(eff/100)))+'}$']
                    #labels += ['$\!\!\!\epsilon_{\operatorname{bkg}}$: '
                    #           + ('$10^{'+str(int(np.log10(eff)))+'}$' if eff<1 else str(int(eff))) +' %']
                    P += [plt.scatter(sig_eff[idx], distance[idx], s=40, marker=marker,
                                      color=color_dict[metric], zorder=10)]
                L = plt.legend(P, labels, loc='upper left', fontsize=13, ncol=1,
                               facecolor='ghostwhite', framealpha=1)
        plt.xlabel('$\epsilon_{\operatorname{'+eff_type+'}}$ (%)', fontsize=30)
        y_text = r'$\,(\mathcal{P}_{\!\operatorname{cut}}\Vert\mathcal{P}_{\!\operatorname{uncut}})$'
        plt.ylabel(distance_type + y_text, fontsize=26)
        ncol = 1 if len(metrics_list)==1 else 2 if len(metrics_list)<9 else 3

        #for var_tuple in itertools.product(metrics_list, [1,0], ['m','pt']):
        #    metric, truth, var = var_tuple
        #    plt.plot(np.nan, np.nan, '.', ms=0, label='($m$)' if var=='m' else '($p_T$)' if var=='pt' else '')
        plt.legend(loc='upper right', fontsize=20, ncol=ncol, facecolor='ghostwhite',
                   framealpha=1, frameon=False, columnspacing=-2.5, labelspacing=0.5)

        if eff_type=='sig': plt.gca().add_artist(L)
        pylab.grid(False); axes = plt.gca()
        axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
        axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        axes.tick_params(which='minor', direction='out', length=4, width=1, colors='black',
                         bottom=True, top=False, left=True, right=False)
        axes.tick_params(which='major', direction='out', length=7, width=3, colors='black',
                         bottom=True, top=False, left=True, right=False)
        axes.tick_params(axis="both", pad=5, labelsize=22)
        for axis in ['top', 'bottom', 'left', 'right']:
            axes.spines[axis].set_linewidth(3)
            axes.spines[axis].set_color('black')
        if eff_type == 'sig':
            pylab.xlim(0, 100)
        elif eff_type == 'bkg':
            x_min = max(-4, int(np.floor(np.log10(x_min))))
            pylab.xlim(10**x_min, 100)
            plt.xscale('log')
            pos    = [10**int(n) for n in range(x_min,3)]
            labels = [('$10^{'+str(n)+'}$' if n<=-1 else int(10**n))for n in range(x_min,3)]
            plt.xticks(pos, labels)
            if best_cut is not None:
                x_pos = best_cut['bkg_eff']
                plt.axvline(x_pos, ymin=0, ymax=1, ls='--', lw=1.5, color='tab:gray', zorder=20)
                text = '.1e' if np.log10(x_pos)<-1 else '.2f' if np.log(x_pos)<1 else '.1f'
                text = format(x_pos,text)
                if np.log(x_pos)<-1:
                    text = text.split('e')[0] + r'$\!\times\!10^{-' + text.split('e-0')[-1] + '}$'
                plt.text((np.log10(x_pos)-x_min)/(2-x_min), 1.018, text, {'color':'black','fontsize':14},
                         va="center", ha="center", transform=axes.transAxes)
        pylab.ylim(0, np.ceil(10*y_max)/10)
    color_dict = {'MSE' :'tab:orange', 'MAE'   :'tab:brown', 'X-S'   :'tab:purple',
                   'JSD':'tab:cyan'  , 'EMD'   :'tab:green', 'KSD'   :'black'     ,
                   'KLD':'tab:red'   , 'Latent':'tab:blue' , 'Inputs':'gray', 'Inputs_scaled':'black'}
    manager   = mp.Manager(); distance_dict = manager.dict()
    arguments = [(y_true, X_losses, sample, metric, eff_type, truth, var, distance_dict)
                 for metric in metrics_list for truth in [0,1] for var in ['m','pt']]
    processes = [mp.Process(target=get_distance, args=arg) for arg in arguments]
    for job in processes: job.start()
    for job in processes: job.join()
    plt.figure(figsize=(12,8)); make_plot(distance_type, loss_metric)
    #plt.figure(figsize=(11,16));
    #plt.subplot(2, 1, 1); make_plot('KSD', loss_metric)
    #plt.subplot(2, 1, 2); make_plot('JSD', loss_metric)
    plt.subplots_adjust(left=0.11, top=0.97, bottom=0.15, right=0.96)
    file_name = output_dir + '/' + distance_type + '_correlations.png'
    print('Saving ' + distance_type + ' correlations  to:', file_name); plt.savefig(file_name)


def plot_discriminant(y_true, X_loss, weights, output_dir, sig_label=None, best_cuts=None, disc=None, n_bins=200,
                      normalize=True, density=True, log=False, logit_scale=True, base=np.e, rotation=20):
    font_manager._get_font.cache_clear()
    def logit(x, base=10, delta=np.float64(1e-42)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x = np.maximum(np.float64(x), delta)
            return (np.log(x) - np.log(1-x))/np.log(base)
    def inverse_logit(x, base=10):
        return 1/(1+base**(-x))
    def make_xlabels(x, n_pos):
        if n_pos < 7: return x
        if   x > 0.5: return '$1\:\!\!\!-\:\!\!\!10^{'+ str(int(np.round(np.log10(1-x)))) +'}$'
        elif x < 0.5: return   '$10^{'+ str(int(np.round(np.log10(x  )))) +'}$'
        else        : return x
    def minor_ticks(pos, idx):
        if pos[idx] == 0.1: return np.linspace(0.2,0.4,3)
        if pos[idx] == 0.5: return np.linspace(0.6,0.8,3)
        else              : return np.linspace(pos[idx],pos[idx+1],10)[1:-1]
    labels = [sig_label, 'QCD']; colors = ['tab:orange', 'tab:blue']
    plt.figure(figsize=(12,8)); pylab.grid(False); ax = plt.gca()
    if log:
        x_lim = 1e-2, 1e4
        y_lim = 1e-3, 1e4
        bins = np.logspace(np.log10(x_lim[0]), np.log10(x_lim[1]), num=n_bins)
    elif logit_scale:
        #x_min, x_max = -36, 4
        x_min, x_max = -4, 4
        x_min = int(max(np.floor(logit(np.min(X_loss),base=10)), x_min))
        x_max = int(min(np.ceil (logit(np.max(X_loss),base=10)), x_max))
        if x_max-x_min+1 > 10:
            x_min = int(2*np.floor(x_min/2))
            x_max = int(2*np.ceil (x_max/2))
        pos  = [  10**float(n) for n in np.arange(x_min        ,0          )]
        pos += [0.5] if x_min <= 0 and x_max >= 0 else []
        pos += [1-10**float(n) for n in np.arange(-max(1,x_min),-x_max-1,-1)]
        lab = [make_xlabels(x, len(pos)) for x in pos]
        if x_max-x_min+1 > 20:
            lab = lab[::2] ; pos = pos[::2]
            for n in range(len(lab))[::2]: lab[n] = ''
        elif x_max-x_min+1 > 10:
            lab = lab[::2] ; pos = pos[::2]
        minor_pos = np.concatenate([minor_ticks(pos, idx) for idx in range(len(pos)-1)])
        if np.any(logit(X_loss,base) == np.inf):
            pos += [1-10**float(-x_max-1),1-10**float(-x_max-2)]
            minor_pos = np.r_[minor_pos, minor_ticks(pos, len(pos)-3)]
            lab += ['', '']
        pos = logit(np.array(pos), base)
        minor_pos = logit(minor_pos, base)
        X_loss = logit(X_loss, base)
        if np.any(X_loss == np.inf):
            break_width = ((pos[-1]-pos[0]))*6e-3
            break_x1 = (pos[-3] + pos[-2] - break_width)/2
            break_x2 = (pos[-3] + pos[-2] + break_width)/2
            #break_x1 = (pos[-3]) - break_width/2
            #break_x2 = (pos[-3]) + break_width/2
            minor_pos = minor_pos[minor_pos<break_x1]
            bins = np.linspace(pos[0], break_x1, num=n_bins)
            bins = np.r_[bins, break_x2, pos[-2], pos[-1]]
            X_loss[np.logical_and(X_loss >= break_x1, X_loss < np.inf)] = (bins[-3] + bins[-2])/2
            X_loss[X_loss == np.inf] = (bins[-2] + bins[-1])/2
            plt.text(((bins[-2]+bins[-1])/2-bins[0])/(bins[-1]-bins[0]), -0.06, '$\mathcal{D}\!=\!1$',
                     {'color':'black','fontsize':24}, va='center', ha='center',
                     transform=ax.transAxes, rotation=0)
        else:
            bins = np.linspace(pos[0], pos[-1], num=n_bins)
    else:
        x_min, x_max = 0, 1
        bins = np.linspace(x_min, x_max, num=n_bins)
    if np.any(weights) is None: weights = np.array(len(y_true)*[1.])
    for n in [1,0]:
        variable = X_loss[y_true==n]
        hist_weights = weights[y_true==n]
        if normalize:
            hist_weights = hist_weights * 100/np.sum(hist_weights)
        if density:
            indices = np.searchsorted(bins, variable, side='right')
            bin_widths = np.diff(bins)
            hist_weights = hist_weights / np.take(bin_widths, np.minimum(indices, len(bins)-1)-1)
        plt.hist(variable, bins[:-3], histtype='step', weights=hist_weights, label=labels[n],
                 edgecolor=colors[n], facecolor=mcolors.to_rgba(colors[n])[:-1]+(0.1,),
                 lw=3, cumulative=False, zorder=0, clip_on=False, fill=True)
        plt.hist(variable, bins[-3:] if len(bins)>n_bins else bins, histtype='step', weights=hist_weights,
                 edgecolor=colors[n], facecolor=mcolors.to_rgba(colors[n])[:-1]+(0.1,),
                 lw=3, cumulative=False, zorder=0, clip_on=False, fill=True)
    ax.set_xlim(bins[0],bins[-1])
    ax.set_ylim(0,None)
    if best_cuts is not None:
        text = format(best_cuts[disc],'.3f')
        if logit_scale:
            best_cuts[disc] = logit(best_cuts[disc], base)
        plt.axvline(best_cuts[disc], ymin=0, ymax=1, ls='--', lw=1.5, color='tab:gray', zorder=30)
        if log:
            x_pos = (np.log10(best_cuts[disc])-np.log10(x_lim[0]))/(np.log10(x_lim[1])-np.log10(x_lim[0]))
        else:
            x_pos = (best_cuts[disc]-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0])
        plt.text(x_pos, 1.018, text, {'color':'black', 'fontsize':14},
                 va="center", ha="center", transform=ax.transAxes)
    if logit_scale:
        plt.xticks(pos, lab, rotation=rotation)
        if x_max-x_min+1 <= 12: ax.set_xticks(minor_pos, minor=True)
    else:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    if log: plt.xscale('log'); plt.yscale('log')
    ax.tick_params(which='minor', direction='in', length=4, width=1, colors='black',
                     bottom=True, top=True, left=True, right=True)
    ax.tick_params(which='major', direction='in', length=7, width=3, colors='black',
                     bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis="both", pad=5, labelsize=22)
    #ax.tick_params(axis="x", pad=5, labelsize=21 if len(pos)<7 else 22)
    #ax.tick_params(axis="y", pad=5, labelsize=22)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
        ax.spines[axis].set_color('black')
    if len(bins) > n_bins:
        dy = 0.013*np.diff(ax.get_ylim())
        box_1 = patches.Rectangle((break_x1, ax.get_ylim()[0]-dy), break_width, 2*dy,
                                  color='white', clip_on=False, zorder=10)
        box_2 = patches.Rectangle((break_x1, ax.get_ylim()[1]-dy), break_width, 2*dy,
                                  color='white', clip_on=False, zorder=10)
        ax.add_patch(box_1)
        ax.add_patch(box_2)
        ax.axvline(break_x1, ymin=-0.013, ymax=0.013, lw=3, color='black', zorder=20, clip_on=False)
        ax.axvline(break_x2, ymin=-0.013, ymax=0.013, lw=3, color='black', zorder=20, clip_on=False)
        ax.axvline(break_x1, ymin =0.987, ymax=1.013, lw=3, color='black', zorder=20, clip_on=False)
        ax.axvline(break_x2, ymin= 0.987, ymax=1.013, lw=3, color='black', zorder=20, clip_on=False)
        #plt.subplots_adjust(left=0.09, top=0.97, bottom=0.15, right=0.99)
        plt.subplots_adjust(left=0.11, top=0.97, bottom=0.15, right=0.96)
    else:
        #plt.subplots_adjust(left=0.10, top=0.97, bottom=0.14, right=0.96)
        plt.subplots_adjust(left=0.11, top=0.97, bottom=0.15, right=0.96)
    plt.xlabel(r'$\mathcal{D}$' if logit_scale else r'$\mathcal{D}$', fontsize=32, labelpad=-4)
    plt.ylabel('Probability Density (%)', fontsize=26, labelpad=6)
    plt.legend(loc='best', fontsize=20, facecolor='ghostwhite', frameon=False)
    output_dir += '/discriminants'
    try   : os.mkdir(output_dir)
    except: pass
    file_name = output_dir+'/'+disc.lower()+'.png'
    print('Saving discriminant      to:', file_name); plt.savefig(file_name)


def plot_distributions(samples, sig_label, plot_var, bin_sizes, output_dir, file_name='', weight_type='None',
                       normalize=True, density=True, log=True, plot_weights=False, logspace=True):
    font_manager._get_font.cache_clear()
    labels = {0:sig_label, 1:'QCD'}
    colors = ['tab:orange', 'tab:blue', 'tab:brown']; alphas = [1, 0.5]
    xlabel = {'pt':'$p_T$', 'm':'$m$', 'm_over_pt':'$m\,/\,{p_T}$',
              'rljet_n_constituents':'Number of constituents'}[plot_var]
    #if   plot_var == 'm' : x_min=  0; x_max= 800; y_min=1e-8; y_max=1.2e0
    #elif plot_var == 'pt': x_min=400; x_max=1750; y_min=1e-8; y_max=  1e0
    if   plot_var == 'm' : x_min=  0; x_max=1000; y_min=1e-8; y_max=1e0
    elif plot_var == 'pt': x_min=400; x_max=3000; y_min=1e-8; y_max=1e0
    plt.figure(figsize=(12,8)); pylab.grid(False); axes = plt.gca()
    if not isinstance(samples, list): samples = [samples]
    for m in [1,0]:
        variable = samples[0][plot_var]
        min_val, max_val = max(x_min, np.min(variable)), min(x_max,np.max(variable))
        bins = get_idx(max_val, bin_size=bin_sizes[plot_var], min_val=min_val, integer=False, tuples=False)
        for n in range(len(samples)):
            sample = samples[n]
            condition = sample['JZW']==-1 if m==0 else sample['JZW']>= 0 if m==1 else sample['JZW']>=-2
            if not np.any(condition): continue
            variable = sample[plot_var][condition]
            bins = get_bins(variable, bins, min_bin_count=20)
        for n in range(len(samples)):
            sample = samples[n]
            condition = sample['JZW']==-1 if m==0 else sample['JZW']>= 0 if m==1 else sample['JZW']>=-2
            if not np.any(condition): continue
            variable = sample[plot_var][condition]
            weights = sample['weights'][condition]
            if normalize:
                if weight_type == 'None': weights = 100*weights/(np.sum(samples[0]['weights']))
                else                    : weights = 100*weights/(np.sum(sample    ['weights']))
            if density:
                try:
                    indices = np.searchsorted(bins, variable, side='right')
                    weights = weights / np.take(np.diff(bins), np.minimum(indices, len(bins)-1)-1)
                except: return
            pylab.hist(variable, bins, histtype='step', weights=weights, label=labels[m],
                       edgecolor=mcolors.to_rgba(colors[m])[:-1]+(alphas[n],), lw=3, log=log, fill=True,
                       facecolor=mcolors.to_rgba(colors[m])[:-1]+(0.1 if n==0 else 0.1,))
    if plot_var == 'pt':
        axes.xaxis.set_major_locator(ticker.MultipleLocator(250))
    pylab.xlim(x_min, x_max); pylab.ylim(y_min, y_max)
    y_min, y_max = np.ceil(np.log10(y_min)), np.floor(np.log10(y_max))
    y_pos = [10**float(n) for n in np.arange(y_min,y_max+1)]
    y_lab = ['$10^{'+str(n)+'}$' if n<0 else str(int(10**n)) for n in np.arange(int(y_min),int(y_max+1))]
    plt.yticks(y_pos, y_lab)
    plt.xlabel(xlabel+('$\,$(GeV)' if plot_var!='m_over_pt' else ''), fontsize=30)
    plt.ylabel('Probability Density (%)', fontsize=26, labelpad=-2 if plt.yticks()[0][0]<=1e-10 else None)
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    if not log: axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    axes.tick_params(which='minor', direction='in', length=5, width=1.5, colors='black',
                     bottom=True, top=True, left=True, right=True)
    axes.tick_params(which='major', direction='in', length=7, width=3, colors='black',
                     bottom=True, top=True, left=True, right=True)
    axes.tick_params(axis="both", pad=5, labelsize=22)
    for axis in ['top', 'bottom', 'left', 'right']:
        axes.spines[axis].set_linewidth(3)
        axes.spines[axis].set_color('black')
    plt.subplots_adjust(left=0.11, top=0.97, bottom=0.15, right=0.96)
    if len(samples) == 1:
        plt.legend(loc='upper right', ncol=1, columnspacing=-2.5, fontsize=20, frameon=False)
    else:
        for m in [1,0]:
            for n in range(len(samples)):
                plt.plot(np.nan, np.nan, '.', ms=0, label='(uncut)' if n==0 else '(cut)')
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [4,5,6,7,0,1,2,3]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                   loc='upper right', ncol=2, columnspacing=-2.5, fontsize=20, frameon=False)
    if file_name == '': file_name = (plot_var if plot_var=='pt' else 'mass')+'_dist.png'
    file_name = output_dir+'/'+file_name
    print('Saving', format(plot_var, '>2s'), 'distributions  to:', file_name); plt.savefig(file_name)


def ROC_curves(y_true, X_loss, weights, disc, sig_label, best_cut, output_dir, valid_data=None):
    font_manager._get_font.cache_clear()
    color_dict = {'MSE' :'tab:orange', 'MAE'   :'tab:gray', 'X-S'   :'tab:purple',
                   'JSD':'tab:cyan'  , 'EMD'   :'tab:green', 'KSD'   :'black'     ,
                   'KLD':'tab:red'   , 'Latent':'tab:blue' , 'Inputs':'gray', 'Inputs_scaled':'black',
                   'Autoencoder':'dimgray', 'Discriminator':'silver', 'Auto+Disc':'tab:blue'}
    if   valid_data is None  : metrics_dict = ROC_rates(y_true, X_loss, weights, [disc])
    elif len(valid_data) == 1: metrics_dict = {key:val+(None,) for key,val in valid_data.items()}
    else                     : metrics_dict = {key:val+(None,) for key,val in binary_dics_eff(valid_data).items()}

    """ Background rejection plot """
    plt.figure(figsize=(12,8)); pylab.grid(False); ax = plt.gca()
    bkg_rej_max = 0; sig_eff_min = 1
    for metric in metrics_dict:
        fpr, tpr, _ = metrics_dict[metric]
        bkg_rej_max = max(bkg_rej_max, np.max(1/fpr))
        sig_eff_min = min(sig_eff_min, np.min(tpr))
        if len(metrics_dict) > 1:
            plt.plot(tpr, 1/fpr, label=metric, lw=4, color=color_dict[metric],
                     zorder=1 if metric=='Auto+Disc' else 2)
        else:
            sig_label = 'QCD vs '+ sig_label + ' (AUC:$\,$'+format(metrics.auc(fpr,tpr), '.3f')+')'
            plt.plot(tpr, 1/fpr, label=sig_label, lw=4, color='tab:gray')

    if best_cut is not None:
        plt.scatter(best_cut['sig_eff'], 1/best_cut['bkg_eff'], color='tab:blue',
                    marker='o', s=80, zorder=30, label='Best BH $\sigma$')
    if valid_data is not None:
        for metric in metrics_dict:
            fpr, tpr, _ = metrics_dict[metric]
            plt.plot(np.nan, np.nan, '.', ms=0, label='(AUC:$\,$'+format(metrics.auc(fpr,tpr), '.3f')+')')
        plt.plot(np.nan, np.nan, '.', ms=0, label=' ')

    y_max = 10**(np.ceil(np.log10(bkg_rej_max)))
    x_min = (np.floor(10*sig_eff_min))/10
    pylab.xlim(x_min,1); pylab.ylim(1,y_max)
    plt.yscale('log')
    ax.xaxis.set_major_locator(ticker.MultipleLocator (0.1 if x_min>=0.5 or x_min%0.2 else  0.2))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5   if x_min>=0.5 or x_min%0.2 else 10  ))

    step = 0.1 if x_min>=0.5 or x_min%0.2 else 0.2
    pos = np.arange(x_min, 1+step, step)
    plt.xticks(pos, [format(n,'.0f' if n==0 or n==1 else '.1f') for n in pos])

    pos    = [10**int(n) for n in np.arange(0,int(np.log10(ax.get_ylim()[1]))+1)]
    labels = ['1'] + ['$10^{'+str(n)+'}$' for n in np.arange(1,int(np.log10(ax.get_ylim()[1]))+1)]
    plt.yticks(pos, labels)
    plt.subplots_adjust(left=0.11, top=0.97, bottom=0.15, right=0.96)
    plt.xlabel('$\epsilon_{\operatorname{sig}}$'  , fontsize=32)
    plt.ylabel('$1/\epsilon_{\operatorname{bkg}}$', fontsize=32)
    ax.tick_params(which='minor', direction='in', length=4, width=1, colors='black',
                     bottom=True, top=True, left=True, right=True, zorder=20)
    ax.tick_params(which='major', direction='in', length=7, width=3, colors='black',
                     bottom=True, top=True, left=True, right=True, zorder=20)
    ax.tick_params(axis="both", pad=8, labelsize=22)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
        ax.spines[axis].set_color('black')
    ax.tick_params(axis='both', which='major', labelsize=22)

    if valid_data is None or len(valid_data)==1:
        plt.legend(loc='upper right', fontsize=20, ncol=1 if valid_data is None else 2, columnspacing=-2.5,
                   facecolor='ghostwhite', frameon=False, framealpha=1).set_zorder(30)
    else:
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0,1,2,7,3,4,5,6]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                   loc='upper right', fontsize=20, ncol=1 if valid_data is None else 2, columnspacing=-2.5,
                   facecolor='ghostwhite', frameon=False, framealpha=1).set_zorder(30)

    file_name = output_dir + '/' + 'bkg_rejection.png'
    print('Saving bkg rejection     to:', file_name); plt.savefig(file_name)

    """ Signal gain plot """
    plt.figure(figsize=(13,8)); pylab.grid(True); ax = plt.gca()
    max_ROC_var = 0
    for metric in metrics_dict:
        fpr, tpr, _ = metrics_dict[metric]
        ROC_var  = tpr/fpr
        max_ROC_var = max(max_ROC_var, np.max(ROC_var[tpr>=0.01]))
        label = metric if metric!='Inputs_scaled' else 'Inputs (scaled)'
        plt.plot(tpr, ROC_var, label=label, lw=2, color=color_dict[metric])
        #plt.plot(tpr, ROC_var, label=metric, lw=2, color=color_dict[metric], ls=ls_dict[metric])
    #pylab.xlim(1,100)
    #pylab.ylim(0,np.ceil(max_ROC_var))
    #plt.xscale('log')
    #plt.xticks([1,10,100], ['1','10', '100'])
    #ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    pylab.xlim(0,100)
    pylab.ylim(1,1e6)
    plt.yscale('log')
    location = 'upper right'
    plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=28)
    plt.ylabel('$G_{S/B}=\epsilon_{\operatorname{sig}}/\epsilon_{\operatorname{bkg}}$', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(loc='best', fontsize=15, ncol=1, facecolor='ghostwhite', framealpha=1).set_zorder(10)
    file_name = output_dir + '/' + 'signal_gain.png'
    print('Saving signal gain       to:', file_name); plt.savefig(file_name)

    '''
    """ Significance plot """
    plt.figure(figsize=(13,8)); pylab.grid(True); ax = plt.gca()
    max_ROC_var = 0
    for metric in metrics_dict:
        fpr, tpr, _ = metrics_dict[metric]
        n_sig = np.sum(weights[y_true==0])
        n_bkg = np.sum(weights[y_true==1])
        ROC_var = n_sig*tpr/np.sqrt(n_bkg*fpr)/10
        max_ROC_var = max(max_ROC_var, np.max(ROC_var[tpr>=1]))
        label = metric if metric!='Inputs_scaled' else 'Inputs (scaled)'
        plt.plot(tpr, ROC_var, label=label, lw=2, color=color_dict[metric])
    val = ROC_var[-1]
    #ax.axhline(val, xmin=0, xmax=1, ls='--', linewidth=1., color='dimgray')
    plt.text(100.4, val, format(val,'.1f'), {'color':'dimgray', 'fontsize':14}, va="center", ha="left")
    pylab.xlim(1,100)
    pylab.ylim(0,np.ceil(max_ROC_var))
    plt.xscale('log')
    plt.xticks([1,10,100], ['1','10', '100'])
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    location = 'upper right'
    plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.ylabel('$\sigma=n_{\operatorname{sig}}\epsilon_{\operatorname{sig}}$/'
               +'$\sqrt{n_{\operatorname{bkg}}\epsilon_{\operatorname{bkg}}}$', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(loc='upper left', fontsize=15, ncol=2 if len(metrics_list)<9 else 3)
    file_name = output_dir + '/' + 'significance.png'
    print('Saving significance      to:', file_name); plt.savefig(file_name)
    '''


def plot_history(hist_file, output_dir, first_epoch=0, x_step=10):
    print('PLOTTING TRAINING HISTORY:')
    losses = pickle.load(open(hist_file, 'rb'))
    plt.figure(figsize=(13,8)); pylab.grid(True); axes = plt.gca()
    epochs = np.arange(1+first_epoch, len(list(losses.values())[0])+1)
    if len(epochs) <= 1: return
    for key, loss in losses.items():
        plt.plot(epochs, loss[first_epoch:], label=key, lw=2)
    pylab.xlim(1, epochs[-1])
    plt.xticks(np.append(1, np.arange(x_step, epochs[-1]+x_step, x_step)))
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    pylab.ylim(0, min(50, np.max(losses['Train loss'][1:])))
    plt.xlabel('Epoch', fontsize=25)
    plt.ylabel('Loss' , fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='upper right', fontsize=18)
    file_name = output_dir+ '/'+ 'train_history.png'
    print('Saving training history  to:', file_name ); plt.savefig(file_name)


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


def bin_meshgrid(beta_val, lamb_val, Z_val, file_name, vmin=None, vmax=None, color='black', prec=2):
    beta_val, lamb_val = [[int(n) if int(n)==n else format(n,'.1f') for n in array]
                          for array in [beta_val, lamb_val]]
    beta_idx = np.arange(0, len(beta_val)+1) - 0.5
    lamb_idx = np.arange(0, len(lamb_val)+1) - 0.5
    plt.figure(figsize=(11,7.5)); ax = plt.gca()
    if vmin is None: vmin = np.min(Z_val[Z_val!=-1])
    if vmax is None: vmax = np.max(Z_val[Z_val!=-1])
    plt.pcolormesh(beta_idx, lamb_idx, Z_val, cmap="Blues"  , edgecolors='black', vmin=vmin, vmax=vmax)
    #plt.pcolormesh(beta_idx, lamb_idx, Z_val, cmap="Blues_r", edgecolors='black', vmin=vmin, vmax=vmax)
    plt.xticks(np.arange(len(beta_val)), beta_val)
    plt.yticks(np.arange(len(lamb_val)), lamb_val)
    for x in range(len(beta_val)):
        for y in range(len(lamb_val)):
            text = 'Ind' if Z_val[y,x]==-1 else format(Z_val[y,x],'.'+str(prec)+'f')
            plt.text(x, y, text, {'color':color, 'fontsize':18}, ha='center', va='center')
    #ax.set_xticks(beta_idx, minor=True)
    #ax.set_yticks(lamb_idx, minor=True)
    #plt.grid(True, color='black', lw=1, alpha=1, which='minor')
    #for val in beta_idx: ax.axvline(val, ymin=0, ymax=1, ls='-', linewidth=1, color='black', zorder=100)
    #for val in lamb_idx: ax.axhline(val, xmin=0, xmax=1, ls='-', linewidth=1, color='black', zorder=100)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Beta'  , fontsize=25)
    plt.ylabel('Lambda', fontsize=25)
    cbar = plt.colorbar(fraction=0.04, pad=0.02)
    ticks = [val for val in cbar.get_ticks() if min(abs(val-vmin),abs(val-vmax))>0.02*(vmax-vmin)
             and round(val,1)!=round(vmin,1) and round(val,1)!=round(vmax,1)]
    ticks = [vmin] + ticks + [vmax]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([format(n,'.1f') for n in ticks])
    cbar.ax.tick_params(labelsize=12)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=color)
    plt.tight_layout()
    print('Saving meshgrid to:', file_name); plt.savefig(file_name)




def deco_example(output_dir):
    def polynom_coeff():
        x0, y0 = (0. , 0.  )
        x1, y1 = (1. , 0.  )
        d1, d2 = (0.2, 0.75)
        f_coeff_1  = [   x0**4,   x0**3,   x0**2, x0 , 1]
        f_coeff_2  = [   x1**4,   x1**3,   x1**2, x1 , 1]
        df_coeff_1 = [ 4*d1**3, 3*d1**2, 2*d1   ,  1 , 0]
        df_coeff_2 = [12*d2**2, 6*d2   ,    2   ,  0 , 0]
        int_coeff  = [     1/5,     1/4,  1/3   , 1/2, 1]
        a = np.array([f_coeff_1, f_coeff_2, df_coeff_1, df_coeff_2, int_coeff])
        b = np.array([y0, y1, 0, 0, 1])
        return np.linalg.solve(a,b)
    def poly_pdf(x, coeff):
        a, b, c, d, e = coeff
        return a*x**4 + b*x**3 + c*x**2 + d*x +e
    def poly_cdf(x, coeff):
        a, b, c, d, e = coeff
        return a*x**5/5 + b*x**4/4 + c*x**3/3 + d*x**2/2 +e*x
    def Gaussian(x, A, B, C):
        return A*np.exp(-(x-B)**2/(2*C**2))
    def rectircle(x, a, b, r):
        return b * (1-((np.abs(x)/a)**(2*a/r)))**(r/(2*b))
    def Maxwell_pdf(x, a):
        return np.sqrt(2/np.pi) * (x**2/a**3) * np.exp(-x**2/(2*a**2))
    def Maxwell_cdf(x, a):
        return special.erf(x/(np.sqrt(2)*a)) - (np.sqrt(2/np.pi) * (x/a) * np.exp(-x**2/(2*a**2)))
    def Maxwell_inv_pdf(x, a):
        return Maxwell_pdf(1-x, a)
    def Maxwell_inv_cdf(x, a):
        return 1 - Maxwell_cdf(1-x, a)
    def best_significance(bkg_rej, sig_rej, score):
        selection = np.logical_and(bkg_rej<1, sig_rej<1)
        bkg_eff = 1 - bkg_rej[selection]
        sig_eff = 1 - sig_rej[selection]
        sigma = sig_eff/np.sqrt(bkg_eff)
        idx   = np.argmax(sigma)
        return {'best_cut':score[idx], 'best_sigma':sigma[idx]}
    def deco_plots(x, new_x, hist_bkg, hist_sig, plot_number, base=np.e):
        fig = plt.figure(figsize=(12,8)) ; pylab.grid(False) ; axes = plt.gca()
        colors = {'bkg':'tab:blue', 'sig':'tab:orange', 'QCD':'darkgray'}
        disc_symbol = 'x' #\mathcal{D}
        axes.tick_params(which='minor', direction='in', length=5, width=2, colors='black',
                         bottom=True, top=True, left=True, right=True)
        axes.tick_params(which='major', direction='out', length=10, width=4, colors='black',
                         bottom=True, top=False, left=True, right=False)
        axes.tick_params(axis="both", pad=8, labelsize=30)
        for axis in ['top', 'bottom', 'left', 'right']:
            axes.spines[axis].set_linewidth(3)
            axes.spines[axis].set_color('black')
            axes.spines[axis].set_zorder(30)
        for axis in ['right','top']:
            axes.spines[axis].set_visible(False)
        if plot_number == 'AUC':
            x = np.linspace(0, 1, int(1e5)+1)
            y = rectircle(x, a=1, b=1, r=0.4)
            plt.plot(x, y, color='darkgray', lw=4, clip_on=False)
            plt.fill_between(x, y, y2=0, alpha=0.15, color='darkgray', clip_on=False)
            plt.text(0.895, 0.79, 'AUC$=$'+format(np.trapz(y,x),'.2f'), {'color':'gray', 'fontsize':14},
                     rotation=-41, va='center', ha='center', transform=axes.transAxes, zorder=20)
            y = rectircle(x, a=1, b=1, r=0.605)
            plt.plot(x, y, color='darkgray', lw=4, clip_on=False)
            plt.text(0.855, 0.71, 'AUC$=$'+format(np.trapz(y,x),'.2f'), {'color':'gray', 'fontsize':14},
                     rotation=-41, va='center', ha='center', transform=axes.transAxes, zorder=20)
            y = rectircle(x, a=1, b=1, r=0.785)
            plt.plot(x, y, color='darkgray', lw=4, clip_on=False)
            plt.text(0.8225, 0.645, 'AUC$=$'+format(np.trapz(y,x),'.2f'), {'color':'gray', 'fontsize':14},
                     rotation=-41, va='center', ha='center', transform=axes.transAxes, zorder=20)
            plt.plot(x, 1-x, color='tab:blue', lw=4, ls='-', clip_on=False)
            plt.text(0.17, 0.72, 'AUC$=$0.50', {'color':'tab:blue', 'fontsize':16},
                     rotation=0, va='center', ha='center', transform=axes.transAxes, zorder=20)
            plt.text(0.17, 0.68, '(random classifier)', {'color':'tab:blue', 'fontsize':16},
                     rotation=0, va='center', ha='center', transform=axes.transAxes, zorder=20)
            x = np.linspace(0, 0.5, int(1e5)+1)
            y = rectircle(x, a=0.5, b=1, r=0.05)
            y[-1] = 0.5
            plt.plot(x, y, color='tab:blue', lw=4, ls=(0,(4, 1)), clip_on=False)

            x_fill = np.append(x,[0.5,1])
            y_fill = np.append(y,[0.5,0])
            plt.fill_between(x_fill, y_fill, y2=0, alpha=0.10, color='tab:blue', clip_on=False)

            y = 1-rectircle(x-0.5, a=0.5, b=1, r=0.024)
            y[0] = 0.5
            plt.plot(x+0.5, y, color='tab:blue', lw=4, ls=(0,(4, 1)), clip_on=False, zorder=20)
            plt.text(0.34, 0.27, 'AUC$=$0.50', {'color':'tab:blue', 'fontsize':16},
                     rotation=0, va='center', ha='center', transform=axes.transAxes, zorder=20)
            plt.text(0.34, 0.23, '(quasi-perfect classifier)', {'color':'tab:blue', 'fontsize':16},
                     rotation=0, va='center', ha='center', transform=axes.transAxes, zorder=20)
            plt.xlim([0,1.0])
            plt.ylim([0,1.0])
            plt.xticks([0,0.5,1], labels=[0,0.5,1])
            plt.yticks([0,1], labels=[0,1])
            plot_arrows(fig, axes)
            axes.axvline(1, ymin=0, ymax=1, ls=':', linewidth=2, color='tab:gray', clip_on=False)
            axes.axhline(1, xmin=0, xmax=1, ls=':', linewidth=2, color='tab:gray', clip_on=False)
            plt.text(1.1, -0.01, '$ϵ_{\operatorname{sig}}$', {'color':'black', 'fontsize':38},
                     va='center', ha='left', transform=axes.transAxes, zorder=20)
            plt.text(0.02, 1.16, '$1\!-\!ϵ_{\operatorname{bkg}}$', {'color':'black', 'fontsize':38},
                     va='center', ha='center', transform=axes.transAxes, zorder=20)
        if plot_number == 'uncut' or plot_number == 'cut':
            x = np.linspace(0, 1, int(1e5)+1)
            if plot_number == 'uncut': QCD = Maxwell_pdf(x/2.5+0.2, a=0.2)+1
            if plot_number ==   'cut': QCD = Maxwell_pdf(x/2.5+0.2, a=0.2)-1
            bump = Gaussian(x, A=1, B=0.5, C=0.03)
            plt.plot(x, np.log(np.exp(QCD)+np.exp(bump)), color=colors['QCD'], lw=4, clip_on=False, label='QCD')
            plt.fill_between(x, np.log(np.exp(QCD)+np.exp(bump)), y2=0, alpha=0.15, color=colors['QCD'], clip_on=False)
            bump_range = np.logical_and(x>=0.4,x<=0.6)
            plt.plot(x[bump_range], bump[bump_range], color=colors['sig'], lw=4, clip_on=False, label='Signal')
            plt.fill_between(x[bump_range], bump[bump_range], y2=0, alpha=0.15, color=colors['sig'])
            plt.xlim([0,1.0])
            plt.ylim([0,4.0])
            plt.xticks([])
            plt.yticks([])
            plot_arrows(fig, axes)
            plt.text(1.1, 0, '$m$', {'color':'black', 'fontsize':42},
                     va='center', ha='left', transform=axes.transAxes, zorder=20)
            plt.text(0, 1.17, '$\mathcal{P}$', {'color':'black', 'fontsize':42},
                     va='center', ha='center', transform=axes.transAxes, zorder=20)
            if plot_number == 'uncut':
                kw = dict(arrowstyle='Simple, head_width=6, head_length=10', color="k")
                curved_arrow = patches.FancyArrowPatch((0.5, 3.35), (0.55, 3.50), lw=2, zorder=10,
                                                       connectionstyle='arc3,rad=-.1', **kw)
                axes.add_patch(curved_arrow)
                plt.text(0.555, 3.6/4, 'Weak', {'color':'black', 'fontsize':18},
                         va='center', ha='left', transform=axes.transAxes, zorder=20)
                plt.text(0.555, 3.4/4, 'Signal', {'color':'black', 'fontsize':18},
                         va='center', ha='left', transform=axes.transAxes, zorder=20)
            if plot_number == 'cut':
                kw = dict(arrowstyle='Simple, head_width=6, head_length=10', color="k")
                curved_arrow = patches.FancyArrowPatch((0.5, 1.85), (0.55, 2.), lw=2, zorder=10,
                                                       connectionstyle='arc3,rad=-.1', **kw)
                axes.add_patch(curved_arrow)
                plt.text(0.555, 2.1/4, 'Strong', {'color':'black', 'fontsize':18},
                         va='center', ha='left', transform=axes.transAxes, zorder=20)
                plt.text(0.555, 1.9/4, 'Signal', {'color':'black', 'fontsize':18},
                         va='center', ha='left', transform=axes.transAxes, zorder=20)
            handles, labels = axes.get_legend_handles_labels()
            new_handles = [patches.Patch(edgecolor=colors[n], facecolor=mcolors.to_rgba(colors[n])[:-1]+(0.15,),
                                         fill=True, lw=3) for n in ['QCD','sig']]
            plt.legend(handles=new_handles, loc='upper right', labels=labels,
                       fontsize=22, facecolor='ghostwhite', frameon=False, ncol=1)

        if plot_number == 'distributions':
            x = np.linspace(0, 1.07, int(1e5)+1)
            func_bkg = Maxwell_pdf(x, a=0.16)
            func_sig = poly_pdf(1-x, coeff=polynom_coeff()) - 0.1
            plt.plot(x, func_bkg, color=colors['bkg'], lw=4, label='Background', zorder=2, clip_on=False)
            plt.plot(x, func_sig, color=colors['sig'], lw=4, label='Signal'    , zorder=1, clip_on=True)
            plt.fill_between(x, func_bkg, y2=0, alpha=0.1, color=colors['bkg'])
            plt.fill_between(x, func_sig, y2=0, alpha=0.1, color=colors['sig'])
            matplotlib.rcParams['hatch.linewidth'] = 2.0
            x_cut = 0.4
            plt.fill_between(x[x>=x_cut], func_bkg[x>=x_cut], y2=0, color="none", edgecolor=colors['bkg'], hatch='//',
                             zorder=3)
            plt.fill_between(x[x<=x_cut], func_sig[x<=x_cut], y2=0, color="none", edgecolor=colors['sig'], hatch='\\\\')
            axes.axvline(x_cut, ymin=0, ymax=1.1, ls='-', linewidth=5, color='dimgray', clip_on=False, zorder=4)
            plt.scatter( [x_cut,x_cut], [0,4.4], s=160, marker='o', zorder=40, clip_on=False, color='dimgray')
            plt.xlim([0,1])
            plt.ylim([0,4])
            plt.xticks([])
            plt.yticks([])
            plot_arrows(fig, axes)
            axes.arrow(0.023,3.85,-0.02,0, lw=2, color='dimgray', head_width=0.07,
                       head_length=0.02, length_includes_head=True, zorder=30)
            axes.arrow(x_cut-0.023,3.85,0.02,0, lw=2, color='dimgray', head_width=0.07,
                       head_length=0.02, length_includes_head=True, zorder=30)
            axes.axhline(3.85, xmin=0, xmax=x_cut, ls='--', linewidth=3, color='dimgray', clip_on=False)
            plt.text(0.2, 3.97, 'Predicted Background', {'color':'dimgray', 'fontsize':18},
                     va='center', ha='center', zorder=20)
            axes.arrow(x_cut+0.023,3.85,-0.02,0, lw=2, color='dimgray', head_width=0.07,
                       head_length=0.02, length_includes_head=True, zorder=30)
            axes.arrow(1-0.023,3.85,0.02,0, lw=2, color='dimgray', head_width=0.07,
                       head_length=0.02, length_includes_head=True, zorder=30)
            axes.axhline(3.85, xmin=x_cut, xmax=1, ls='--', linewidth=3, color='dimgray', clip_on=False)
            plt.text(0.7, 3.97, 'Predicted Signal', {'color':'dimgray', 'fontsize':18},
                     va='center', ha='center', zorder=20)
            plt.text(x_cut, -0.2, 'Variable Threshold', {'color':'dimgray', 'fontsize':18},
                     va='center', ha='center', zorder=20)
            axes.axvline(1, ymin=0, ymax=3.85/4, ls=':', linewidth=2, color='tab:gray', clip_on=False)
            kw = dict(arrowstyle='Simple, head_width=6, head_length=10', color='dimgray')
            curved_arrow = patches.FancyArrowPatch((0.3, 0.22), (0.5, 2), lw=2, zorder=10,
                                                   connectionstyle='arc3,rad=-.3', **kw)
            axes.add_patch(curved_arrow)
            kw = dict(arrowstyle='Simple, head_width=6, head_length=10', color='dimgray')
            curved_arrow = patches.FancyArrowPatch((0.42, 0.67), (0.5, 1.96), lw=2, zorder=10,
                                                   connectionstyle='arc3,rad=-.1', **kw)
            axes.add_patch(curved_arrow)
            plt.text(0.505, 1.97, 'Wrong Predictions', {'color':'dimgray','fontsize':18},
                     va='center', ha='left', zorder=20)
            plt.text(1.09, 0, '$\mathcal{D}$', {'color':'black', 'fontsize':42},
                     va='center', ha='left', transform=axes.transAxes, zorder=20)
            #plt.text(0, 1.185, '$\mathcal{P}(\mathcal{D})$', {'color':'black', 'fontsize':42},
            plt.text(0, 1.17, '$\mathcal{P}$', {'color':'black', 'fontsize':42},
                     va='center', ha='center', transform=axes.transAxes, zorder=20)
            plt.text(0.22, 0.21, 'FN', {'color':colors['sig'],'fontsize':32}, va='center', ha='center',
                     zorder=20, fontweight='bold')
            plt.text(0.45, 0.21, 'FP', {'color':colors['bkg'],'fontsize':32}, va='center', ha='center',
                     zorder=20, fontweight='bold')
            plt.text(0.22, 1.50, 'TN', {'color':colors['bkg'],'fontsize':32}, va='center', ha='center',
                     zorder=20, fontweight='bold')
            plt.text(0.80, 0.80, 'TP', {'color':colors['sig'],'fontsize':32}, va='center', ha='center',
                     zorder=20, fontweight='bold')
            handles, labels = axes.get_legend_handles_labels()
            new_handles = [patches.Patch(edgecolor=colors[n], facecolor=mcolors.to_rgba(colors[n])[:-1]+(0.1,),
                                         fill=True, lw=3) for n in ['bkg','sig']]
            plt.legend(handles=new_handles, loc='upper right', labels=labels,
                       fontsize=22, facecolor='ghostwhite', frameon=False, bbox_to_anchor=(1,0.95))

        if plot_number == 'ROC_curve':
            x = np.linspace(0, 1, int(1e5)+1)
            e_bkg = Maxwell_cdf(1, a=0.16) - Maxwell_cdf(x, a=0.16)
            e_sig = -poly_cdf(0, coeff=polynom_coeff()) + poly_cdf(1-x, coeff=polynom_coeff())
            plt.plot(e_sig, 1-e_bkg, color='gray', lw=4, zorder=1, clip_on=False,
                     label='AUC$=$'+format(-np.trapz(1-e_bkg,e_sig),'.2f'))
            #plt.plot(e_sig[e_bkg!=0], e_sig[e_bkg!=0]/e_bkg[e_bkg!=0], color='darkgray', lw=4, zorder=1, clip_on=False)
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.xticks([0,1], labels=[0,1])
            plt.yticks([0,1], labels=[0,1])
            plot_arrows(fig, axes)
            r_sig = 0.5
            accuracy = e_sig*r_sig + (1-e_bkg)*(1-r_sig)
            best_idx = np.argmax(accuracy)
            plt.scatter( e_sig[best_idx], (1-e_bkg)[best_idx], s=120, marker='o', zorder=40,
                         label='Best Accuracy ('+format(100*accuracy[best_idx],'.0f')+'%)', color='black')
            #print(x[best_idx])
            plt.text(1.1, -0.01, '$ϵ_{\operatorname{sig}}$', {'color':'black', 'fontsize':38},
                     va='center', ha='left', transform=axes.transAxes, zorder=20)
            plt.text(0.02, 1.16, '$1\!-\!ϵ_{\operatorname{bkg}}$', {'color':'black', 'fontsize':38},
                     va='center', ha='center', transform=axes.transAxes, zorder=20)
            plt.legend(loc='upper left', fontsize=22, facecolor='ghostwhite',
                       frameon=False, bbox_to_anchor=(0,0.95))

        if plot_number == 'gain_curve':
            x = np.linspace(0, 1, int(1e5)+1)
            e_bkg = Maxwell_cdf(1, a=0.16) - Maxwell_cdf(x, a=0.16)
            e_sig = -poly_cdf(0, coeff=polynom_coeff()) + poly_cdf(1-x, coeff=polynom_coeff())
            #plt.plot(e_sig[e_bkg!=0], 1/e_bkg[e_bkg!=0], color='darkgray', lw=4, zorder=1, clip_on=False)
            plt.plot(e_sig[e_bkg!=0], e_sig[e_bkg!=0]/e_bkg[e_bkg!=0], color='darkgray', lw=4, zorder=1, clip_on=False)
            #print( e_sig[e_bkg!=0][np.argmax(e_sig[e_bkg!=0]/e_bkg[e_bkg!=0])] )
            plt.xlim([0,1])
            plt.ylim([1,3e5])
            plt.xticks([0,1], labels=[0,1])
            plt.yticks([1,1e5,2e5,3e5], rotation=0, labels=[1,'1e5','2e5','3e5'])
            plot_arrows(fig, axes, y_origin=1)
            plt.text(1.1, -0.01, '$ϵ_{\operatorname{sig}}$', {'color':'black', 'fontsize':38},
                     va='center', ha='left', transform=axes.transAxes, zorder=20)
            #plt.text(0.0, 1.184, '$G_{S/B}/10^5$',
            #         {'color':'black', 'fontsize':36}, va='center', ha='center', transform=axes.transAxes, zorder=20)
            plt.text(0.0, 1.185, '$G_{\operatorname{s/b}}$',
                     {'color':'black', 'fontsize':38}, va='center', ha='center', transform=axes.transAxes, zorder=20)
            axes.tick_params(axis='y', pad=4, labelsize=30)

        if plot_number == 'sigma_curve':
            x = np.linspace(0, 1, int(1e5)+1)
            e_bkg = Maxwell_cdf(1, a=0.16) - Maxwell_cdf(x, a=0.16)
            e_sig = -poly_cdf(0, coeff=polynom_coeff()) + poly_cdf(1-x, coeff=polynom_coeff())
            plt.plot(e_sig[e_bkg!=0], e_sig[e_bkg!=0]/np.sqrt(e_bkg[e_bkg!=0]), color='darkgray', lw=4, zorder=1, clip_on=False)
            plt.xlim([0,1])
            plt.ylim([1,120])
            plt.xticks([0,1], labels=[0,1])
            plt.yticks([1,40,80,120], rotation=0, labels=[1,40,80,120])
            plot_arrows(fig, axes, y_origin=1)
            plt.text(1.1, -0.01, '$ϵ_{\operatorname{sig}}$', {'color':'black', 'fontsize':38},
                     va='center', ha='left', transform=axes.transAxes, zorder=20)
            plt.text(0.0, 1.195, '$\sigma_{\operatorname{ratio}}$',
                     {'color':'black', 'fontsize':38}, va='center', ha='center', transform=axes.transAxes, zorder=20)
            axes.tick_params(axis='y', pad=4, labelsize=30)

        if plot_number == '0':
            x = np.linspace(0, 1.07, int(1e5)+1)
            plt.plot(x, Maxwell_pdf(x+0.35, a=0.32), color='darkgray', lw=4, clip_on=False, label='QCD')
            plt.fill_between(x, Maxwell_pdf(x+0.35,a=0.32), y2=0, alpha=0.1, color='gray')
            x = np.linspace(0.15, 0.20, 100)
            plt.fill_between(x, Maxwell_pdf(x+0.35,a=0.32), y2=0, alpha=0.3, color='dimgray')
            plt.fill_between(x, Maxwell_pdf(x+0.35,a=0.32), facecolor="none", hatch="//", edgecolor="dimgray",
                             linewidth=3, ls='-', zorder=10)
            plt.xlim([0,1.0])
            plt.ylim([0,1.9])
            plt.xticks([])
            plt.yticks([])
            plot_arrows(fig, axes)
            #plt.text(1.02, 0.0, 'm$\,;p_T$', {'color':'black', 'fontsize':40},
            #         va='center', ha='left', transform=axes.transAxes, zorder=20)
            plt.text(1.1, 0.05, '$m$', {'color':'black', 'fontsize':42},
                     va='center', ha='left', transform=axes.transAxes, zorder=20)
            plt.text(1.1, -0.05, '$p_T$', {'color':'black', 'fontsize':42},
                     va='center', ha='left', transform=axes.transAxes, zorder=20)
            plt.text(0, 1.17, '$\mathcal{P}$', {'color':'black', 'fontsize':42},
                     va='center', ha='center', transform=axes.transAxes, zorder=20)
            kw = dict(arrowstyle='Simple, head_width=6, head_length=10', color="k")
            curved_arrow = patches.FancyArrowPatch((0.175, 1), (0.4, 1.25), lw=2, zorder=10,
                                                   connectionstyle='arc3,rad=-.2', **kw)
            axes.add_patch(curved_arrow)
            plt.text(0.4, 1.25/1.9, 'Bin', {'color':'black', 'fontsize':22},
                     va='center', ha='left', transform=axes.transAxes, zorder=20)
            handles, labels = axes.get_legend_handles_labels()
            new_handles = [patches.Patch(edgecolor='darkgray', facecolor=mcolors.to_rgba('gray')[:-1]+(0.1,),
                                         fill=True, lw=3)]
            plt.legend(handles=new_handles, loc='upper left', labels=labels,
                       fontsize=22, facecolor='ghostwhite', frameon=False, bbox_to_anchor=(0, 1.1))
        if plot_number == '1a' or plot_number == '2a':
            plt.xticks([0,1], labels=[0,1])
            plt.yticks([0,1,2,3,4,5], labels=[0,1,2,3,4,5])
            plt.plot(x, f_bkg(x), color=colors['bkg'], lw=4, label='Background', zorder=2)
            plt.plot(x, f_sig(x), color=colors['sig'], lw=4, label='Signal', zorder=1)
            plt.fill_between(x, f_bkg(x), y2=0, alpha=0.1, color=colors['bkg'])
            plt.fill_between(x, f_sig(x), y2=0, alpha=0.1, color=colors['sig'])
            if plot_number == '1a' and False:
                x_bin = np.linspace(0.28, 0.32, 100)
                plt.fill_between(x_bin, f_bkg(x_bin), y2=0, alpha=0.25, color=colors['bkg'])
                plt.fill_between(x_bin, f_bkg(x_bin), facecolor="none", hatch=None, edgecolor=colors['bkg'],
                                 linewidth=3, ls='-', zorder=10)
                plt.text(0.3, -0.05, '$\Delta$$'+disc_symbol+'$', {'color':'black', 'fontsize':26},
                         va='center', ha='center', transform=axes.transAxes, zorder=20)
            plt.xlim([0,1])
            plt.ylim([0,5])
            plot_arrows(fig, axes)
            plt.text(1.1, 0, '$'+disc_symbol+'$', {'color':'black', 'fontsize':42},
                     va='center', ha='left', transform=axes.transAxes, zorder=20)
            plt.text(0, 1.18, '$f('+disc_symbol+')$', {'color':'black', 'fontsize':42},
                     va='center', ha='center', transform=axes.transAxes, zorder=20)
            cut = best_significance(F_bkg(x), F_sig(x), x)['best_cut']
            ymax = max(f_bkg(cut), f_sig(cut))/axes.get_ylim()[1]
            axes.axvline(cut, ymin=0, ymax=ymax, ls='--', linewidth=2, color='tab:gray')
            #plt.legend(loc='upper left', fontsize=22, facecolor='ghostwhite', frameon=False)
            handles, labels = axes.get_legend_handles_labels()
            new_handles = [patches.Patch(edgecolor=colors[n], facecolor=mcolors.to_rgba(colors[n])[:-1]+(0.1,),
                                         fill=True, lw=3) for n in ['bkg','sig']]
            plt.legend(handles=new_handles, loc='upper left', labels=labels,
                       fontsize=22, facecolor='ghostwhite', frameon=False)
        if plot_number == '1b' or plot_number == '2b':
            plt.xticks([0,1], labels=[0,1])
            plt.yticks([0,1], labels=[0,1])
            plt.plot(x, F_bkg(x), color=colors['bkg'], lw=4, label='Background', clip_on=False, zorder=0)
            x_bin = np.linspace(0.28, 0.32, 100)
            if plot_number == '1b':
                plt.fill_between(x_bin, F_bkg(x_bin), y2=0, alpha=0.25, color=colors['bkg'])
                plt.fill_between(x_bin, F_bkg(x_bin), facecolor="none", hatch=None, edgecolor=colors['bkg'],
                                 linewidth=3, ls='-', zorder=0)
                plt.fill_betweenx(F_bkg(x_bin), x_bin, x2=0, alpha=0.25, color=colors['bkg'])
                plt.fill_betweenx(F_bkg(x_bin), x_bin, facecolor="none", hatch=None, edgecolor=colors['bkg'],
                                  linewidth=3, ls='-', zorder=0)
                plt.text(0.3, -0.05, '$\Delta$$'+disc_symbol+'$', {'color':'black', 'fontsize':26},
                         va='center', ha='center', transform=axes.transAxes, zorder=20)
                plt.text(-0.035, 0.45, '$\Delta$$F$', {'color':'black', 'fontsize':26},
                         va='center', ha='center', transform=axes.transAxes, zorder=20)
                #plt.text(0.3, -0.05, '$'+disc_symbol+'$', {'color':'black', 'fontsize':26},
                #         va='center', ha='center', transform=axes.transAxes, zorder=20)
                #plt.text(-0.025, 0.45, '$F$', {'color':'black', 'fontsize':26},
                #         va='center', ha='center', transform=axes.transAxes, zorder=20)
                plt.text(0.79, 0.76, '$F('+disc_symbol+')$$={\int}^{\!'+disc_symbol+'}_{\!\!0}$$\!f(x)dx$',
                         {'color':'black', 'fontsize':28},
                         va='center', ha='center', transform=axes.transAxes, zorder=20)
                axes.arrow(0.3,0,0,F_bkg(0.3), lw=2, ls='-', fc='k', ec='k',
                           head_width=0.01, head_length=0.03, overhang=0.,
                           length_includes_head=True, clip_on=False, zorder=30)
                axes.arrow(0.3,F_bkg(0.3),-0.3,0, lw=2, ls='-', fc='k', ec='k',
                           head_width=0.015, head_length=0.02, overhang=0.,
                           length_includes_head=True, clip_on=False, zorder=30)
            if plot_number == '2b':
                plt.text(0.3, -0.05, '$'+disc_symbol+'$', {'color':'black', 'fontsize':26},
                         va='center', ha='center', transform=axes.transAxes, zorder=20)
                plt.text(-0.025, 0.41, '$F$', {'color':'black', 'fontsize':26},
                         va='center', ha='center', transform=axes.transAxes, zorder=20)
                plt.text(0.79, 0.90, '$F('+disc_symbol+')$$={\int}^{\!'+disc_symbol+'}_{\!\!0}$$\!f(x)dx$',
                         {'color':'black', 'fontsize':28},
                         va='center', ha='center', transform=axes.transAxes, zorder=20)
                axes.arrow(0.3,0,0,F_bkg(0.3), lw=2, ls='-', fc='k', ec='k',
                           head_width=0.01, head_length=0.03, overhang=0.,
                           length_includes_head=True, clip_on=False, zorder=30)
                axes.arrow(0.3,F_bkg(0.3),-0.3,0, lw=2, ls='-', fc='k', ec='k',
                           head_width=0.015, head_length=0.02, overhang=0.,
                           length_includes_head=True, clip_on=False, zorder=30)
            plt.xlim([0,1])
            plt.ylim([0,1])
            plot_arrows(fig, axes)
            plt.text(1.1, 0, '$'+disc_symbol+'$', {'color':'black', 'fontsize':42},
                     va='center', ha='left', transform=axes.transAxes, zorder=20)
            plt.text(0, 1.18, '$F('+disc_symbol+')$', {'color':'black', 'fontsize':42},
                     va='center', ha='center', transform=axes.transAxes, zorder=20)
            axes.axvline(1, ymin=0, ymax=1, ls=':', linewidth=2, color='tab:gray', clip_on=False)
            axes.axhline(1, xmin=0, xmax=1, ls=':', linewidth=2, color='tab:gray', clip_on=False)
            plt.legend(loc='upper left', fontsize=22, facecolor='ghostwhite', frameon=False)
        if plot_number == '1c' or plot_number == '2c':
            plt.plot((new_x[:-1]+new_x[1:])/2, hist_bkg/np.diff(new_x) + (1 if plot_number == '2c' else 0),
                     color=colors['bkg'], lw=4, label='Background', clip_on=False, zorder=2)
            plt.plot(np.append((new_x[:-1]+new_x[1:])/2,[1]), np.append(hist_sig/np.diff(new_x),0),
                     color=colors['sig'], lw=4, label='Signal'    , clip_on=False, zorder=1)
            plt.fill_between((new_x[:-1]+new_x[1:])/2, hist_bkg/np.diff(new_x),
                             y2=0, alpha=0.1, color=colors['bkg'])
            plt.fill_between(np.append((new_x[:-1]+new_x[1:])/2,[1]), np.append(hist_sig/np.diff(new_x),0),
                             y2=0, alpha=0.1, color=colors['sig'])
            plt.xlim([0,1])
            plt.xticks([0,1], labels=[0,1])
            if plot_number == '1c':
                plt.ylim([0,8])
                plt.yticks([0,2,4,6,8], labels=[0,2,4,6,8])
            if plot_number == '2c':
                plt.ylim([0,None])
                plt.yticks([0,100,200,300], labels=[0,100,200,300])
                axes.tick_params(axis="y", pad=4, labelsize=30)
            plot_arrows(fig, axes)
            plt.text(1.1, 0, '$F$', {'color':'black', 'fontsize':42},
                     va='center', ha='left', transform=axes.transAxes, zorder=20)
            plt.text(0, 1.18, '$g(F)$', {'color':'black', 'fontsize':42},
                     va='center', ha='center', transform=axes.transAxes, zorder=20)
            cut = best_significance(np.cumsum(hist_bkg), np.cumsum(hist_sig), new_x[1:])['best_cut']
            idx = np.argmin(np.abs(cut-(new_x[:-1]+new_x[1:])/2))
            ymax = (hist_sig/np.diff(new_x))[idx]/axes.get_ylim()[1]
            axes.axvline(cut, ymin=0, ymax=ymax, ls='--', linewidth=2, color='tab:gray')
            #plt.legend(loc='upper left', fontsize=22, facecolor='ghostwhite', frameon=False)
            handles, labels = axes.get_legend_handles_labels()
            new_handles = [patches.Patch(edgecolor=colors[n], facecolor=mcolors.to_rgba(colors[n])[:-1]+(0.1,),
                                         fill=True, lw=3) for n in ['bkg','sig']]
            plt.legend(handles=new_handles, loc='upper left', labels=labels,
                       fontsize=22, facecolor='ghostwhite', frameon=False)
        if plot_number == '1d' or plot_number == '2d':
            def logit        (x, base=10): return (np.log(x) - np.log(1-x))/np.log(base)
            def inverse_logit(x, base=10): return 1/(1+base**(-x))
            if plot_number == '1d':
                x_min, x_max = -3, 3 # base-10 exponent range
            if plot_number == '2d':
                x_min, x_max = -3, 4.1 # base-10 exponent range
            pos  = [  10**float(n) for n in np.arange(int(np.floor(x_min)),0)       ] + [0.5]
            pos += [1-10**float(n) for n in np.arange(-1,-int(np.floor(x_max))-1,-1)]
            lab  = ['0.'+n*'0'+'1' for n in np.arange(int(np.floor(x_min))+5,-1,-1) ] + [0.5]
            lab += ['0.9'+n*'9'    for n in np.arange(0,int(np.floor(x_max)))       ]
            pos  = logit(np.array(pos), base)
            x_min, x_max = np.log(10**(x_min))/np.log(base), np.log(10**(x_max))/np.log(base)
            x = np.linspace(1.5*x_min, 1.5*x_max, int(1e6)+1)
            x = inverse_logit(x, base)
            x_map = [F_bkg(a)            for a   in    (x[:-1]+x[1:])/2]
            n_bkg = [F_bkg(b) - F_bkg(a) for a,b in zip(x[:-1],x[1:])  ]
            n_sig = [F_sig(b) - F_sig(a) for a,b in zip(x[:-1],x[1:])  ]
            new_x = np.linspace(x_min, x_max*1.1, int(1e3)+1)
            bins  = inverse_logit(new_x, base)
            hist_bkg = np.histogram(x_map, bins=bins, weights=n_bkg, density=False)[0]
            hist_sig = np.histogram(x_map, bins=bins, weights=n_sig, density=False)[0]
            hist_bkg /= np.sum(hist_bkg)
            hist_sig /= np.sum(hist_sig)
            plt.plot((new_x[:-1]+new_x[1:])/2, hist_bkg/np.diff(new_x),
                     color=colors['bkg']  , lw=4, label='Background', clip_on=False, zorder=2)
            plt.plot((new_x[:-1]+new_x[1:])/2, hist_sig/np.diff(new_x),
                     color=colors['sig'], lw=4, label='Signal'    , clip_on=True, zorder=1)
            #plt.plot(new_x, np.log(base)/(base**(-new_x/2)+base**(new_x/2))**2,
            #         color='green'  , lw=4, label='Background', clip_on=False, zorder=2)
            plt.fill_between((new_x[:-1]+new_x[1:])/2, hist_bkg/np.diff(new_x), y2=0, alpha=0.1, color=colors['bkg'])
            plt.fill_between((new_x[:-1]+new_x[1:])/2, hist_sig/np.diff(new_x), y2=0, alpha=0.1, color=colors['sig'])
            plt.xlim([x_min,x_max])
            plt.xticks(pos, lab, rotation=20)
            if plot_number == '1d':
                plt.ylim([0,0.65])
                plt.yticks([0,0.2,0.4,0.6], labels=[0,0.2,0.4,0.6])
            if plot_number == '2d':
                plt.ylim([0,0.3])
                plt.yticks([0,0.1,0.2,0.3], labels=[0,0.1,0.2,0.3])
            axes.tick_params(axis="x", pad=1, labelsize=28)
            plot_arrows(fig, axes, x_origin=x_min)
            #x_label, y_label = '$t(F)$', '$g(t)$'
            x_label, y_label = '$F$', '$g(t)$'
            plt.text(1.1, 0, x_label, {'color':'black', 'fontsize':42},
                     va='center', ha='left', transform=axes.transAxes, zorder=20)
            plt.text(0, 1.18, y_label, {'color':'black', 'fontsize':42},
                     va='center', ha='center', transform=axes.transAxes, zorder=20)
            cut = best_significance(np.cumsum(hist_bkg), np.cumsum(hist_sig), new_x[1:])['best_cut']
            idx = np.argmin(np.abs(cut-(new_x[:-1]+new_x[1:])/2))
            ymax = (hist_sig/np.diff(new_x))[idx]/axes.get_ylim()[1]
            axes.axvline(cut, ymin=0, ymax=ymax, ls='--', linewidth=2, color='tab:gray')
            hist_bkg_max = np.max(hist_bkg/np.diff(new_x))
            axes.axhline(hist_bkg_max, xmin=0, xmax=(-x_min)/(x_max-x_min),
                         ls=':', linewidth=2, color='tab:gray')
            plt.text(-0.015, hist_bkg_max/axes.get_ylim()[1], r'$\frac{1}{4}$',
                     {'color':'black', 'fontsize':26},
                     va='center', ha='right', transform=axes.transAxes, zorder=20)
            #plt.legend(loc='upper left', fontsize=22, facecolor='ghostwhite', frameon=False,
            #           bbox_to_anchor=(0, 1.05))
            handles, labels = axes.get_legend_handles_labels()
            new_handles = [patches.Patch(edgecolor=colors[n], facecolor=mcolors.to_rgba(colors[n])[:-1]+(0.1,),
                                         fill=True, lw=3) for n in ['bkg','sig']]
            plt.legend(handles=new_handles, loc='upper left', labels=labels,
                       fontsize=22, facecolor='ghostwhite', frameon=False, bbox_to_anchor=(0, 1.05))
        #fig.subplots_adjust(left=0.06, top=0.83, bottom=0.10, right=0.84)
        fig.subplots_adjust(left=0.08, top=0.83, bottom=0.11, right=0.84)
        #fig.subplots_adjust(left=0.12, top=0.83, bottom=0.11, right=0.84)
        file_name = output_dir+'/'+'deco_'+str(plot_number)+'.png'
        print('Printing:', file_name)
        plt.savefig(file_name); plt.close()
    def plot_arrows(fig, axes, x_origin=0, y_origin=0):
        xmin, xmax = axes.get_xlim()
        ymin, ymax = axes.get_ylim()
        # get width and height of axes object to compute matching arrowhead length and width
        dps = fig.dpi_scale_trans.inverted()
        bbox = axes.get_window_extent().transformed(dps)
        width, height = bbox.width, bbox.height
        # manual arrowhead width and length
        hw = 1./30.*(ymax-ymin)
        hl = 1./30.*(xmax-xmin)
        lw = 3. # axis line width
        ohg = 0.2 # arrow overhang
        # compute matching arrowhead length and width
        yhw = hw/(ymax-ymin) * (xmax-xmin) * height/width
        yhl = hl/(xmax-xmin) * (ymax-ymin) * width/height
        # draw x and y axis
        axes.arrow(xmin, y_origin, (xmax-xmin)*1.08, 0., fc='k', ec='k', lw=lw,
                   head_width=hw, head_length=hl, overhang = ohg,
                   length_includes_head=True, clip_on=False, zorder=10)
        axes.arrow(x_origin, ymin, 0., (ymax-ymin)*1.12, fc='k', ec='k', lw=lw,
                   head_width=yhw, head_length=yhl, overhang = ohg,
                   length_includes_head=True, clip_on=False)
    def get_hist():
        x = np.linspace(0, 1, int(1e5)+1)
        x_map = [F_bkg(a)            for a   in    (x[:-1]+x[1:])/2]
        n_bkg = [F_bkg(b) - F_bkg(a) for a,b in zip(x[:-1],x[1:])  ]
        n_sig = [F_sig(b) - F_sig(a) for a,b in zip(x[:-1],x[1:])  ]
        new_x = np.linspace(np.min(x_map), np.max(x_map), int(1e3)+1)
        hist_bkg = np.histogram(x_map, bins=new_x, weights=n_bkg, density=False)[0]
        hist_sig = np.histogram(x_map, bins=new_x, weights=n_sig, density=False)[0]
        hist_bkg /= np.sum(hist_bkg)
        hist_sig /= np.sum(hist_sig)
        return x, new_x, hist_bkg, hist_sig

    #deco_plots(x=None, new_x=None, hist_bkg=None, hist_sig=None, plot_number='AUC')
    #deco_plots(x=None, new_x=None, hist_bkg=None, hist_sig=None, plot_number='uncut')
    #deco_plots(x=None, new_x=None, hist_bkg=None, hist_sig=None, plot_number='cut')
    #deco_plots(x=None, new_x=None, hist_bkg=None, hist_sig=None, plot_number='distributions')
    #deco_plots(x=None, new_x=None, hist_bkg=None, hist_sig=None, plot_number='ROC_curve')
    #deco_plots(x=None, new_x=None, hist_bkg=None, hist_sig=None, plot_number='gain_curve')
    #deco_plots(x=None, new_x=None, hist_bkg=None, hist_sig=None, plot_number='sigma_curve')
    #sys.exit()

    f_bkg = partial(poly_pdf, coeff=polynom_coeff())
    F_bkg = partial(poly_cdf, coeff=polynom_coeff())
    f_sig = partial(Maxwell_inv_pdf, a=0.12)
    F_sig = partial(Maxwell_inv_cdf, a=0.12)
    x, new_x, hist_bkg, hist_sig = get_hist()
    for plot_number in ['0','1a','1b','1c']: deco_plots(x, new_x, hist_bkg, hist_sig, plot_number)
    deco_plots(x=None, new_x=None, hist_bkg=None, hist_sig=None, plot_number='1d', base=np.e)
    sys.exit()

    f_bkg = partial(Maxwell_pdf    , a=0.215)
    F_bkg = partial(Maxwell_cdf    , a=0.215)
    f_sig = partial(Maxwell_inv_pdf, a=0.12)
    F_sig = partial(Maxwell_inv_cdf, a=0.12)
    x, new_x, hist_bkg, hist_sig = get_hist()
    for plot_number in ['2a','2b','2c']: deco_plots(x, new_x, hist_bkg, hist_sig, plot_number)
    deco_plots(x=None, new_x=None, hist_bkg=None, hist_sig=None, plot_number='2d', base=np.e)
    sys.exit()
