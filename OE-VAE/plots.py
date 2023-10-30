import numpy             as np
import multiprocessing   as mp
import matplotlib.pyplot as plt
import os, sys, h5py, time, pickle, warnings
from matplotlib import pylab, ticker, patches, colors as mcolors, font_manager
from sklearn    import metrics
from scipy      import spatial, interpolate
from functools  import partial
from utils      import loss_function, latent_loss, inverse_scaler, get_4v, bump_hunter
from utils      import n_constituents, jets_pt, get_idx


def plot_results(y_true, X_true, X_pred, sample, n_dims, model, metrics, loss_metric,
                 sig_data, output_dir, apply_cuts, normal_losses, decorrelation):
    def loss_mapping(x):
        if   np.all(np.logical_and(x >= 0, x <= 1)): return  x
        elif np.all(np.logical_and(x >=-1, x <= 0)): return  x + 1
        elif np.all(x >= 0)                        : return  x/(np.abs(x)+1)
        elif np.all(x <= 0)                        : return  x/(np.abs(x)+1) + 1
        else                                       : return (x/(np.abs(x)+1) + 1)/2
    print('\nPLOTTING PERFORMANCE RESULTS:')
    manager = mp.Manager(); X_losses = manager.dict()
    arguments = [(X_true, X_pred, n_dims, metric, X_losses) for metric in set(metrics)-set(['Inputs','Latent'])]
    if 'Inputs' in metrics:
        arguments += [(sample['constituents'], X_pred, n_dims, 'Inputs'       , X_losses)]
        arguments += [(X_true                , X_pred, n_dims, 'Inputs_scaled', X_losses)]
    processes = [mp.Process(target=loss_function, args=arg) for arg in arguments]
    for job in processes: job.start()
    for job in processes: job.join()
    """ Adding latent space KLD metric and loss """
    if 'Latent' in metrics:
        X_losses['Latent'] = latent_loss(X_true, model)
    metrics = list(X_losses.keys())
    if normal_losses == 'ON' or decorrelation == 'ON':
        X_losses = {key:loss_mapping(val) for key,val in X_losses.items()}
    if decorrelation == 'ON':
        #X_losses[loss_metric] = mass_deco(y_true, sample, X_losses[loss_metric], deco='m')
        #X_losses[loss_metric] = mass_deco(y_true, sample, X_losses[loss_metric], deco='pt')
        X_losses[loss_metric] = mass_deco(y_true, sample, X_losses[loss_metric], deco='2d')
    best_loss  = bump_scan(y_true, X_losses[loss_metric], loss_metric, sample, sig_data, output_dir)
    #processes  = [mp.Process(target=ROC_curves, args=(y_true, X_losses, sample['weights'], metrics, output_dir))]
    #arguments  = [(y_true, X_losses, sample['m'], sample['weights'], metrics, loss_metric, output_dir)]
    #processes += [mp.Process(target=mass_correlation, args=arg) for arg in arguments]
    #arguments  = [(y_true, X_losses[metric], sample['weights'], metric, output_dir, best_loss) for metric in metrics]
    #processes += [mp.Process(target=loss_distributions, args=arg) for arg in arguments]
    #processes += [mp.Process(target=tSNE, args=(y_true,  X_true, model, output_dir))]
    #for job in processes: job.start()
    #for job in processes: job.join()
    if apply_cuts == 'ON':
        generate_cuts(y_true, sample, X_losses[loss_metric], loss_metric, sig_data, output_dir)
    print()


def get_bins(var, deco, max_bins=100, min_bin_count=2, logspace=True):
    if not deco: return [np.min(var), np.max(var)]
    if logspace: m_bins = np.logspace(np.log10(np.min(var)), np.log10(np.max(var)), num=max_bins)
    else       : m_bins = np.linspace(         np.min(var) ,          np.max(var) , num=max_bins)
    while True:
        m_idx = np.clip(np.digitize(var, m_bins), 1, len(m_bins)-1) - 1
        for idx in range(len(m_bins)-1)[::-1]:
            if np.sum(m_idx==idx) < max(2, min_bin_count):
                m_bins = np.delete(m_bins, idx)
                break
        if idx == 0: return m_bins
def cum_distribution(x):
    values, counts = [np.insert(array, 0, 0) for array in np.unique(x, return_counts=True)]
    return interpolate.interp1d(values, np.cumsum(counts)/len(x), fill_value=(0,1), bounds_error=False)
def mass_deco(y_true, sample, X_loss, deco='2d'):
    def get_pt_bins(mass, pt, m_range, deco):
        return get_bins(pt[np.logical_and(mass>=m_range[0], mass<m_range[1])], False if deco=='m' else True)
    mass, pt, loss = sample['m'][y_true==1], sample['pt'][y_true==1], X_loss[y_true==1]
    m_bins  =  get_bins(mass, False if deco=='pt' else True)
    pt_bins = [get_pt_bins(mass, pt, m_range, deco) for m_range in zip(m_bins[:-1], m_bins[1:])]
    m_idx  =  np.clip(np.digitize(mass, m_bins), 1, len(m_bins)-1)-1
    pt_idx = [np.clip(np.digitize(pt  ,   bins), 1, len(  bins)-1)-1 for bins in pt_bins]
    cdf_list = [[cum_distribution(loss[np.logical_and(m_idx==m, pt_idx[m]==n)])
                 for n in range(np.max(pt_idx[m])+1)] for m in range(len(pt_idx))]
    mass, pt, loss = sample['m'], sample['pt'], X_loss
    m_idx  =  np.clip(np.digitize(mass, m_bins), 1, len(m_bins)-1)-1
    pt_idx = [np.clip(np.digitize(pt  ,   bins), 1, len(  bins)-1)-1 for bins in pt_bins]
    for m in range(len(pt_idx)):
        for n in range(np.max(pt_idx[m])+1):
            selections = np.logical_and(m_idx==m, pt_idx[m]==n)
            loss[selections] = cdf_list[m][n](loss[selections])
    return loss


def generate_cuts(y_true, sample, X_loss, loss_metric, sig_data, output_dir, cut_types=['bkg_eff','gain']):
    print('\nAPPLYING CUTS ON SAMPLE:')
    def plot_suppression(sample, sig_data, X_loss, positive_rates, bkg_eff=None, file_name=None):
        cut_sample = make_cut(y_true, X_loss, sample, positive_rates, loss_metric, cut_type, bkg_eff)
        if file_name is None: file_name  = 'bkg_suppression/bkg_eff_' + format(bkg_eff,'1.0e')
        sample_distributions([sample,cut_sample], sig_data, output_dir, file_name)
    if not os.path.isdir(output_dir+'/bkg_suppression'): os.mkdir(output_dir+'/bkg_suppression')
    positive_rates = get_rates(y_true, X_loss, sample['weights'])
    for cut_type in cut_types:
        if cut_type in ['bkg_eff']:
            processes = [mp.Process(target=plot_suppression, args=(sample, sig_data, X_loss, positive_rates,
                         bkg_eff)) for bkg_eff in [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]]
            for job in processes: job.start()
            for job in processes: job.join()
        if cut_type in ['gain','sigma']:
            file_name = 'bkg_suppression/best_' + cut_type ; print()
            plot_suppression(sample, sig_data, X_loss, positive_rates, file_name=file_name)


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
        if normalize: weights *= 100/np.sum(weights)
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
        if normalize: weights *= 100/np.sum(weights)
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


def sample_distributions(sample, OoD_data, output_dir, name, weight_type='None', bin_sizes={'m':2.5,'pt':10}):
    processes = [mp.Process(target=plot_distributions, args=(sample, OoD_data, var, bin_sizes,
                 output_dir, name+'_'+var+'.png', weight_type)) for var in ['m','pt']] #,'m_over_pt']]
    for job in processes: job.start()
    for job in processes: job.join()


def get_rates(y_true, X_loss, weights, metric=None, return_dict=None):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, X_loss, pos_label=0, sample_weight=weights)
    fpr, tpr, thresholds = 100*fpr[fpr!=0], 100*tpr[fpr!=0], thresholds[fpr!=0]
    if return_dict is None: return                fpr, tpr, thresholds
    else                  : return_dict[metric] = fpr, tpr, thresholds


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


def bump_scan(y_true, X_loss, loss_metric, sample, sig_data, output_dir, n_cuts=100, eff_type='bkg'):
    def logit(x, delta=1e-16): return np.log10(x) - np.log10(1-x)
    def inverse_logit(x)     : return 1/(1+10**(-x))
    fpr, tpr, thresholds = get_rates(y_true, X_loss, sample['weights'])
    if eff_type == 'sig':
        eff = tpr
        x_min, x_max = 10*np.floor(tpr[0]/10), 100
        eff_val = np.linspace(tpr[0], x_max, n_cuts)
    elif eff_type == 'bkg':
        eff = fpr
        x_min, x_max = 10**np.ceil(np.log10(np.min(fpr))), 100
        #eff_val = np.logspace(np.log10(x_min), np.log10(x_max), n_cuts)
        eff_val = np.append(100*inverse_logit(np.linspace(logit(x_min/100),-logit(x_min/100),n_cuts)), 100)
    idx = np.minimum(np.searchsorted(eff, eff_val, side='right'), len(eff)-1)
    sample = {key:sample[key] for key in ['JZW','m','pt','weights']}
    global get_sigma
    def get_sigma(sample, X_loss, thresholds, idx):
        cut_sample = {key:sample[key][X_loss>thresholds[idx]] for key in sample}
        try   : return bump_hunter(cut_sample)
        except: return None, None
    with mp.Pool() as pool:
        sigma = np.array(pool.map(partial(get_sigma, sample, X_loss, thresholds), idx))
    sigma, _ = [np.array(array) for array in zip(*sigma)]
    none_filter = sigma != np.array(None)
    thresholds, eff = np.take(thresholds, idx), np.take(eff, idx)
    thresholds, eff, sigma = thresholds[none_filter], eff[none_filter], sigma[none_filter]
    if len(sigma) == 0: return None
    plt.figure(figsize=(13,8)); pylab.grid(True); axes = plt.gca()
    plt.plot(eff, sigma, label='', color='tab:blue', lw=2, zorder=1)
    factor  = np.floor(np.log10(np.max(sigma)))
    val_pre = '.'+str(int(max(1,-factor+1)))+'f'
    factor = 10**factor
    pylab.xlim(x_min, x_max)
    pylab.ylim(factor*np.floor(np.min(sigma/factor)), factor*np.ceil(np.max(sigma/factor)))
    end_val = sigma[-1]
    max_val = np.max(sigma)
    max_eff = eff[np.argmax(sigma)]
    plt.text(1.0025, (end_val-axes.get_ylim()[0])/np.diff(axes.get_ylim()), format(end_val,val_pre),
             {'color':'dimgray','fontsize':14}, va="center", ha="left", transform=axes.transAxes)
    plt.text(1.0025, (max_val-axes.get_ylim()[0])/np.diff(axes.get_ylim()), format(max_val,val_pre),
             {'color':'dimgray','fontsize':14}, va="center", ha="left", transform=axes.transAxes)
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.tick_params(axis='both', which='major', labelsize=15)
    if eff_type == 'sig':
        xmin = (max_eff-x_min)/(x_max-x_min)
        plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
    elif eff_type == 'bkg':
        plt.xlabel('$\epsilon_{\operatorname{bkg}}$ (%)', fontsize=25)
        plt.xscale('log')
        pos    = [10**int(n) for n in range(int(np.log10(x_min)),3)]
        labels = [('$10^{'+str(n)+'}$' if n<=-1 else int(10**n))for n in range(int(np.log10(x_min)),3)]
        plt.xticks(pos, labels)
        xmin = (np.log10(max_eff)-np.log10(x_min))/(np.log10(x_max)-np.log10(x_min))
    axes.axhline(max_val, xmin=xmin, xmax=1, ls='--', linewidth=1., color='dimgray')
    plt.ylabel('Significance', fontsize=25)
    file_name = output_dir+'/'+'BH_sigma.png'
    print('Saving max significance  to:', file_name); plt.savefig(file_name)
    """ Printing bkg suppression and bum hunting plots at maximum significance cut """
    best_loss = {'metric':loss_metric, 'eff':eff[np.argmax(sigma)], 'loss':thresholds[np.argmax(sigma)]}
    cut_sample = {key:sample[key][X_loss>best_loss['loss']] for key in sample}
    bump_hunter(cut_sample, sig_label=None, max_sigma=None, filename=output_dir+'/BH_best.png' , print_info=False)
    sample_distributions([sample,cut_sample], sig_data, output_dir, 'BH_bkg_supp', bin_sizes={'m':2.5,'pt':10})
    return best_loss

'''
def bump_scan(y_true, X_loss, loss_metric, sample, sig_data, output_dir, n_cuts=100, eff_type='bkg'):
    font_manager._get_font.cache_clear()
    if   'top'  in sig_data: sig_label = 'Top'
    elif 'VZ'   in sig_data: sig_label = 'VZ'
    elif 'BSM'  in sig_data: sig_label = 'BSM'
    elif 'OoD'  in sig_data: sig_label = 'OoD'
    elif '2HDM' in sig_data: sig_label = '2HDM'
    else                   : sig_label = 'N.A.'
    def logit(x, delta=1e-16):
        x = np.clip(np.float64(x), delta, 1-delta)
        return np.log10(x) - np.log10(1-x)
    def inverse_logit(x)     : return 1/(1+10**(-x))
    fpr, tpr, thresholds = get_rates(y_true, X_loss, sample['weights'])
    cut_thresholds = thresholds
    if eff_type == 'sig':
        cut_eff = tpr
        x_min, x_max = 10*np.floor(tpr[0]/10), 100
        eff_val = np.linspace(tpr[0], x_max, n_cuts)
    elif eff_type == 'bkg':
        cut_eff = fpr
        x_min, x_max = 10**np.ceil(np.log10(np.min(fpr))), 100
        eff_val = np.append(100*inverse_logit(np.linspace(logit(x_min/100),-logit(x_min/100),n_cuts)), 100)
    idx = np.minimum(np.searchsorted(cut_eff, eff_val, side='right'), len(cut_eff)-1)
    sample = {key:sample[key] for key in ['JZW','m','pt','weights']}
    global get_sigma
    def get_sigma(sample, X_loss, cut_thresholds, idx):
        cut_sample = {key:sample[key][X_loss>cut_thresholds[idx]] for key in sample}
        try   : return bump_hunter(cut_sample)
        except: return None, None
    with mp.Pool() as pool:
        sigma = np.array(pool.map(partial(get_sigma, sample, X_loss, cut_thresholds), idx))
    loc_sigma, max_sigma = [np.array(array) for array in zip(*sigma)]
    none_filter = np.logical_and(loc_sigma!=np.array(None), max_sigma!=np.array(None))
    cut_thresholds, cut_eff   = np.take(cut_thresholds,idx), np.take(cut_eff,idx)
    cut_thresholds, cut_eff   = cut_thresholds[none_filter],   cut_eff[none_filter]
    loc_sigma     , max_sigma =      loc_sigma[none_filter], max_sigma[none_filter]
    if len(cut_thresholds) == 0:
        print('FAILED')
        return
    """ Printing bkg suppression and bum hunting plots at maximum significance cut """
    opt_max_sigma = np.max(max_sigma)
    loc_sigma, max_sigma = loc_sigma/loc_sigma[-1], max_sigma/max_sigma[-1]
    opt_sigma = loc_sigma if np.max(loc_sigma)>=np.max(max_sigma) else max_sigma
    best_cut = {'metric':loss_metric, 'cut':cut_thresholds[np.argmax(opt_sigma)]}
    best_cut['eff'    ] = cut_eff[np.argmax(opt_sigma)]
    best_cut['bkg_eff'] = fpr[np.argmin(np.abs(thresholds-best_cut['cut']))]
    best_cut['sig_eff'] = tpr[np.argmin(np.abs(thresholds-best_cut['cut']))]
    cut_sample = {key:sample[key][X_loss>best_cut['cut']] for key in sample}
    #plot_significance(cut_eff, eff_type, loc_sigma, max_sigma, opt_sigma, x_min, x_max, output_dir)
    #bump_hunter(    sample, sig_label, opt_max_sigma, filename=output_dir+'/BH_uncut.png', print_info=False)
    #bump_hunter(cut_sample, sig_label, opt_max_sigma, filename=output_dir+'/BH_best.png' , print_info=False)
    #sample_distributions([sample,cut_sample], sig_label, output_dir, 'BH_bkg_supp', bin_sizes={'m':5,'pt':10})
    arguments  = (cut_eff, eff_type, loc_sigma, max_sigma, opt_sigma, x_min, x_max, output_dir)
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
'''

def plot_significance(cut_eff, eff_type, loc_sigma, max_sigma, opt_sigma, x_min, x_max, output_dir):
    plt.figure(figsize=(12,8)); pylab.grid(False); axes = plt.gca()
    plt.plot(cut_eff, max_sigma, label='Best Bin $\sigma$', color='silver', lw=4)
    plt.plot(cut_eff, loc_sigma, label='Local $\sigma$'   , color='gray'  , lw=4)
    all_sigma = np.append(loc_sigma, max_sigma)
    if   np.max(all_sigma)-np.min(all_sigma) >= 10: factor, val_pre = (10  ,'.0f')
    elif np.max(all_sigma)-np.min(all_sigma) >=  1: factor, val_pre = ( 1  ,'.1f')
    else                                          : factor, val_pre = ( 0.1,'.2f')
    x_min = min(1,x_min)
    y_min, y_max = factor*np.floor(np.min(all_sigma)/factor), factor*np.ceil(np.max(all_sigma)/factor)
    pylab.xlim(x_min, x_max)
    pylab.ylim(y_min, y_max)
    max_val, end_val, max_eff = np.max(opt_sigma), opt_sigma[-1], cut_eff[np.argmax(opt_sigma)]
    #plt.text(1.004, (end_val-axes.get_ylim()[0])/np.diff(axes.get_ylim()), format(end_val,'.0f'),
    #         {'color':'black','fontsize':14}, va="center", ha="left", transform=axes.transAxes)
    plt.text(1.004, (max_val-axes.get_ylim()[0])/np.diff(axes.get_ylim()), format(max_val,val_pre),
             {'color':'black','fontsize':14}, va="center", ha="left", transform=axes.transAxes)
    plt.text((np.log10(max_eff)-np.log10(x_min))/(np.log10(x_max)-np.log10(x_min)), 1.018, format(max_eff,'.1f'),
             {'color':'black','fontsize':14}, va='center', ha='center', transform=axes.transAxes)
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    axes.tick_params(which='minor', direction='in', length=5, width=1.5, colors='black',
                     bottom=True, top=True, left=True, right=True)
    axes.tick_params(which='major', direction='in', length=7, width=3, colors='black',
                     bottom=True, top=True, left=True, right=True)
    axes.tick_params(axis="both", pad=5, labelsize=22)
    for axis in ['top', 'bottom', 'left', 'right']:
        axes.spines[axis].set_linewidth(3)
        axes.spines[axis].set_color('black')
    if eff_type == 'sig':
        xmin = (max_eff-x_min)/(x_max-x_min)
        plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=30, labelpad=5)
    elif eff_type == 'bkg':
        plt.xlabel('$\epsilon_{\operatorname{bkg}}$ (%)', fontsize=30, labelpad=5)
        plt.xscale('log')
        pos    = [10**int(n) for n in range(int(np.log10(x_min)),3)]
        labels = [('$10^{'+str(n)+'}$' if n<=-1 else int(10**n))for n in range(int(np.log10(x_min)),3)]
        plt.xticks(pos, labels)
        xmin = (np.log10(max_eff)-np.log10(x_min))/(np.log10(x_max)-np.log10(x_min))
    axes.axhline(max_val, xmin=xmin, xmax=1, ls='--', linewidth=1.5, color='tab:gray')
    axes.axvline(max_eff, ymin=(max_val-y_min)/(y_max-y_min), ymax=1, ls='--', linewidth=1.5, color='tab:gray')
    #plt.ylabel('BH Significance', fontsize=26)
    plt.ylabel('$\sigma_{\operatorname{cut}}/\sigma_{\operatorname{uncut}}$', fontsize=30, labelpad=5)
    plt.legend(loc='best', fontsize=20, facecolor='ghostwhite', frameon=False, framealpha=1)
    plt.subplots_adjust(left=0.11, top=0.97, bottom=0.15, right=0.96)
    file_name = output_dir+'/'+'BH_sigma.png'
    print('Saving BH significance   to:', file_name); plt.savefig(file_name)


def plot_bump(data, data_weights, y_true, bins, bin_sigma, loc_sigma, max_sigma,
              bump_range, m_range, gaussian_par, sig_label, filename, log=False):
    def Gaussian(x, A, B, C): return A*np.exp(-(x-B)**2/(2*C**2))
    labels     = {0:'QCD', 1:sig_label}
    color_dict = {labels[0]:'tab:blue', labels[1]:'tab:orange'}
    fig, (ax1,ax2) = plt.subplots(figsize=(12,8), ncols=1, nrows=2, sharex=True,
                                          gridspec_kw={'height_ratios':[3,1]})
    data_weights  = 100*data_weights/(np.sum(data_weights))
    indices       = np.searchsorted(bins, data, side='right')
    data_weights /= np.take(np.diff(bins), np.minimum(indices, len(bins)-1)-1)
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
    x_max = min(m_range[1], bins[np.where(bin_sigma>max_sigma/50)[0][-1]] )
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
    plt.text(bump_range[1]/x_max+0.02, np.max(bin_sigma)/y_max-0.05,
             '$\,\sigma_{\operatorname{local}}\,\,=$'+format(loc_sigma,'.1f'),
             {'color':'black','fontsize':14}, va='top', ha='left', transform=ax2.transAxes)
    plt.text(bump_range[1]/x_max+0.02, np.max(bin_sigma)/y_max-0.22,
             '$m_{\operatorname{bump}}\!=$'+format(mean*C_approx+B_approx,'.1f')+'$\,$GeV',
             {'color':'black','fontsize':14}, va='top', ha='left', transform=ax2.transAxes)
    fig.align_labels()
    fig.subplots_adjust(left=0.11, right=0.96, bottom=0.15, top=0.97, hspace=0.08)
    print('Saving bump hunting plot to:', filename)
    plt.savefig(filename, bbox_inches="tight")


def mass_distances(y_true, X_losses, X_mass, weights, metric, eff_type, truth, distance_dict, n_cuts=100):
    X_loss = X_losses[metric]
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
    losses  =  X_loss[y_true==truth]
    masses  =  X_mass[y_true==truth]
    weights = weights[y_true==truth]
    KSD = []; JSD = []; sig_eff = []; bkg_eff = []
    P = np.histogram(masses, bins=100, range=(0,500), weights=weights)[0]
    for n in range(len(thresholds)):
        masses_cut  = masses [losses>=thresholds[n]]
        weights_cut = weights[losses>=thresholds[n]]
        if len(masses_cut) != 0:
            Q = np.histogram(masses_cut, bins=100, range=(0,500), weights=weights_cut)[0]
            #KSD += [ KS_distance(masses, masses_cut, weights, weights_cut) ]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                JSD += [ spatial.distance.jensenshannon(P, Q) ]
            sig_eff += [ tpr[n] ]
            bkg_eff += [ fpr[n] ]
    distance_dict[(metric,truth)] = KSD, JSD, sig_eff, bkg_eff


def mass_correlation(y_true, X_losses, X_mass, weights, metrics_list, loss_metric, output_dir, eff_type='bkg'):
    def make_plot(distance_type, loss_metric):
        for metric in metrics_list:
            for truth in [1,0]:
                label = str(metric) + ' (' + ('sig' if truth==0 else 'bkg') +')'
                KSD, JSD, sig_eff, bkg_eff = distance_dict[(metric,truth)]
                distance = JSD if distance_type=='JSD' else KSD
                ls, alpha = ('-',1) if truth==1 else ('-',0.5)
                if eff_type ==  'sig':
                    plt.plot(sig_eff, distance, label=label, color=color_dict[metric],
                             lw=2, ls=ls, zorder=1, alpha=alpha)
                elif eff_type  == 'bkg':
                    plt.plot(bkg_eff, distance, label=label, color=color_dict[metric],
                             lw=2, ls=ls, zorder=1, alpha=alpha)
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
        plt.xlabel('$\epsilon_{\operatorname{'+eff_type+'}}$ (%)', fontsize=25)
        plt.ylabel(distance_type, fontsize=25)
        ncol = 1 if len(metrics_list)==1 else 2 if len(metrics_list)<9 else 3
        plt.legend(loc='upper center', fontsize=15, ncol=ncol, facecolor='ghostwhite', framealpha=1).set_zorder(10)
        if eff_type=='sig': plt.gca().add_artist(L)
        pylab.grid(True); axes = plt.gca()
        axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
        axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
        axes.tick_params(axis='both', which='major', labelsize=15)
        if eff_type == 'sig':
            pylab.xlim(0, 100)
        elif eff_type == 'bkg':
            pylab.xlim(1e-4, 100)
            plt.xscale('log')
            pos    = [10**int(n) for n in range(-4,3)]
            labels = [('$10^{'+str(n)+'}$' if n<=-1 else int(10**n))for n in range(-4,3)]
            plt.xticks(pos, labels)
        pylab.ylim(0, 1.0)
    color_dict = {'MSE' :'tab:orange', 'MAE'   :'tab:brown', 'X-S'   :'tab:purple',
                   'JSD':'tab:cyan'  , 'EMD'   :'tab:green', 'KSD'   :'black'     ,
                   'KLD':'tab:red'   , 'Latent':'tab:blue' , 'Inputs':'gray', 'Inputs_scaled':'black'}
    manager   = mp.Manager(); distance_dict = manager.dict()
    arguments = [(y_true, X_losses, X_mass, weights, metric, eff_type, truth, distance_dict)
                 for metric in metrics_list for truth in [0,1]]
    processes = [mp.Process(target=mass_distances, args=arg) for arg in arguments]
    for job in processes: job.start()
    for job in processes: job.join()
    plt.figure(figsize=(13,8)); make_plot('JSD', loss_metric)
    #plt.figure(figsize=(11,16));
    #plt.subplot(2, 1, 1); make_plot('KSD', loss_metric)
    #plt.subplot(2, 1, 2); make_plot('JSD', loss_metric)
    file_name = output_dir + '/' + 'mass_correlation.png'
    print('Saving mass sculpting    to:', file_name); plt.savefig(file_name)


def loss_distributions(y_true, X_loss, weights, metric, output_dir, best_loss=None, n_bins=100,
                       normalize=True, density=True, log=False):
    if log:
        x_lim = 1e-2, 1e4
        y_lim = 1e-3, 1e4
        bins = np.logspace(np.log10(x_lim[0]), np.log10(x_lim[1]), num=n_bins)
    else:
        min_loss, max_loss = 0, 1
        bins = np.linspace(min_loss, max_loss, num=n_bins)
    labels = [r'$t\bar{t}$', 'QCD']; colors = ['tab:orange', 'tab:blue']
    if np.any(weights) == None: weights = np.array(len(y_true)*[1.])
    plt.figure(figsize=(13,8)); pylab.grid(True); ax = plt.gca()
    for n in set(y_true):
        variable     = X_loss [y_true==n]
        hist_weights = weights[y_true==n]
        if normalize:
            hist_weights *= 100/np.sum(hist_weights)
        if density:
            indices       = np.searchsorted(bins, variable, side='right')
            hist_weights /= np.take(np.diff(bins), np.minimum(indices, len(bins)-1)-1)
        pylab.hist(variable, bins, histtype='step', weights=hist_weights,
                   label=labels[n], color=colors[n], lw=2, cumulative=False)
    if log: pylab.xlim(x_lim); pylab.ylim(y_lim)
    else  : pylab.xlim(min_loss, max_loss)
    if best_loss is not None and metric == best_loss['metric']:
        ax.axvline(best_loss['loss'], ymin=0, ymax=1, ls='--', linewidth=1., color='black')
        if log:
            x_pos = (np.log10(best_loss['loss'])-np.log10(x_lim[0]))/(np.log10(x_lim[1])-np.log10(x_lim[0]))
        else:
            x_pos = (best_loss['loss']-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0])
        plt.text(x_pos, 1.01, format(best_loss['loss'],'.2f'), {'color':'black', 'fontsize':10},
                 va="center", ha="center", transform=ax.transAxes)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    if log: plt.xscale('log'); plt.yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=15)
    if metric == 'Latent'       : label = 'KLD Latent Loss'
    if metric == 'Inputs'       : label = 'Inputs'
    if metric == 'Inputs_scaled': label = 'Inputs (scaled)'
    if metric not in ['Latent','Inputs','Inputs_scaled']: label = metric + ' Reconstruction Loss'
    plt.xlabel(label, fontsize=24)
    plt.ylabel('Distribution Density (%)', fontsize=24)
    plt.legend(loc='upper left', fontsize=18)
    output_dir += '/metrics_losses'
    try   : os.mkdir(output_dir)
    except: pass
    file_name = output_dir+'/'+metric+'_loss.png'
    print('Saving metric loss       to:', file_name); plt.savefig(file_name)


def plot_distributions(samples, sig_data, plot_var, bin_sizes, output_dir, file_name='', weight_type='None',
                       normalize=True, density=True, log=True, plot_weights=False):
    if   'top'  in sig_data: tag = r'$t\bar{t}$'
    elif 'VZ'   in sig_data: tag = r'$t\bar{t}$'
    elif 'BSM'  in sig_data: tag = 'BSM'
    elif 'OoD'  in sig_data: tag = 'OoD'
    elif '2HDM' in sig_data: tag = '2HDM'
    else                   : tag = 'N.A.'
    if 'OoD' in sig_data: labels = {0:[tag,'QCD'], 1:[tag+' (weighted)','QCD (weighted)']}
    else: labels = {0:[tag,'QCD'], 1:[tag+' (cut)','QCD (cut)']}
    colors = ['tab:orange', 'tab:blue', 'tab:brown']; alphas = [1, 0.5]
    xlabel = {'pt':'$p_t$', 'm':'$m$', 'm_over_pt':'$m\,/\,{p_t}$',
              'rljet_n_constituents':'Number of constituents'}[plot_var]
    plt.figure(figsize=(13,8)); pylab.grid(True); axes = plt.gca()
    if not isinstance(samples, list): samples = [samples]
    for m in [0,1]:
        for n in range(len(samples)):
            sample = samples[n]
            condition = sample['JZW']==-1 if m==0 else sample['JZW']>= 0 if m==1 else sample['JZW']>=-2
            if not np.any(condition): continue
            if plot_var == 'm_over_pt':
                variable = np.float32(sample['m']/sample['pt'])[condition]
                bin_sizes[plot_var] = 0.01
            else:
                variable = np.float32(sample[plot_var][condition])
            weights = sample['weights'][condition]
            if plot_var == 'm_over_pt': min_val, max_val = max(0, np.min(variable))        , np.max(variable        )
            elif 'flat' in weight_type: min_val, max_val = max(0, np.min(variable))        , np.max(variable        )
            else                      : min_val, max_val = max(0, np.min(sample[plot_var])), np.max(sample[plot_var])
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
                try:
                    indices  = np.searchsorted(bins, variable, side='right')
                    weights /= np.take(np.diff(bins), np.minimum(indices, len(bins)-1)-1)
                except: return
            pylab.hist(variable, bins, histtype='step', weights=weights, color=colors[m],
                       lw=2, log=log, alpha=alphas[n], label=labels[n][m])
    if not density or plot_weights:
        pass
    elif 'OoD' in sig_data:
        if   plot_var == 'm' : pylab.xlim(0, 1200); pylab.ylim(1e0, 1e5)
        elif plot_var == 'pt': pylab.xlim(0, 3000); pylab.ylim(1e0, 1e5)
    elif 'Geneva' in sig_data:
        if   plot_var == 'm' : pylab.xlim(0,  800); pylab.ylim(1e-2, 1e5)
        elif plot_var == 'pt': pylab.xlim(0, 2000); pylab.ylim(1e-2, 1e5)
        elif plot_var == 'm_over_pt': pylab.xlim(0,0.8); pylab.ylim(1e-2, 1e5)
    else:
        if   plot_var == 'm' : pylab.xlim(0,  500); pylab.ylim(1e0, 1e6)
        elif plot_var == 'pt': pylab.xlim(0, 2000); pylab.ylim(1e0, 1e6)
    if normalize:
        if   plot_var == 'm' : pylab.ylim(1e-6, 1e0)
        elif plot_var == 'pt': pylab.ylim(1e-7, 1e0)
        elif plot_var == 'm_over_pt': pylab.ylim(1e-4, 1e3)
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    if not log: axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    plt.xlabel(xlabel+(' (GeV)' if plot_var!='m_over_pt' else ''), fontsize=24)
    y_label = ' density' if density else ''
    if normalize: y_label += ' (%)'
    elif sig_data in ['top-UFO','BSM']: y_label += ' ('+r'58.5 fb$^{-1}$'+')'
    plt.ylabel('Distribution' + y_label, fontsize=24)
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='upper right', ncol=1 if len(samples)==1 else 2, fontsize=18)
    if file_name == '':
        file_name = (plot_var if plot_var=='pt' else 'mass')+'_dist.png'
    file_name = output_dir+'/'+file_name
    print('Saving', format(plot_var, '>2s'), 'distributions  to:', file_name); plt.savefig(file_name)


def combine_ROC_curves(output_dir):
    def plot_ROC_curve(file_path):
        fpr, tpr = pickle.load(open(file_path, 'rb')).values()
        fpr_0 = np.sum(fpr==0)
        return fpr[fpr_0:], tpr[fpr_0:]
    data_path  = '/lcg/storage20/atlas/godin/ATLAS-VAE/jet-ID/outputs/Delphes_new'
    #file_list  = ['top_10const', 'top_20const', 'top_30const', 'top_40const', 'top_50const',
    #              'top_60const', 'top_70const', 'top_80const', 'top_90const', 'top_100const']
    #label_list = [n+ ' const' for n in ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100']]
    file_list  = ['top_20const', 'top_40const', 'top_60const', 'top_80const', 'top_100const']
    label_list = [n+ ' const' for n in ['20', '40', '60', '80', '100']]
    #data_path  = '/lcg/storage20/atlas/godin/ATLAS-VAE/jet-ID/outputs/Atlas'
    #file_list  = ['top_50const', 'top_100const', 'top_150const', 'top_196const', 'top_50const',
    #              'top_60const', 'top_70const', 'top_80const', 'top_90const', 'top_100const']
    #label_list = [n+ ' const' for n in ['50', '100', '150', '200']]
    file_list  = [data_path+'/'+pkl_file+'/class_0_vs_bkg/pos_rates.pkl' for pkl_file in file_list]
    pos_rates  = {label:plot_ROC_curve(file_path) for label,file_path in zip(label_list, file_list)}
    #color_list = 2*['tab:orange', 'tab:blue', 'tab:green', 'tab:green', 'tab:green']
    plt.figure(figsize=(13,8));  axes = plt.gca()
    pylab.grid(True, which="both", ls="--", color='tab:blue', alpha=0.2)
    #for n_zip in list(zip(label_list, color_list))[:5]:
    #    label, color = n_zip
    for label in label_list[:5]:
        fpr, tpr = pos_rates[label]
        plt.plot(100*tpr, 1/fpr, label=label, lw=2)
    #Ps = []; labels = []
    #for n_zip in list(zip(label_list, color_list))[5:]:
    #    label, color = n_zip
    #for label in label_list[5:]:
    #    fpr, tpr = pos_rates[label]
    #    P, = plt.plot(100*tpr, 1/fpr, label=label, lw=2)
    #    Ps += [P]; labels += [label]
    for fpr,tpr in list(pos_rates.values())[:5]:
        plt.plot(np.nan, np.nan, '.', ms=0, label='(AUC: '+format(metrics.auc(fpr,tpr), '.4f')+')')
    #for fpr,tpr in list(pos_rates.values())[5:]:
    #    P, = plt.plot(np.nan, np.nan, '.', ms=0, label='(AUC: '+format(metrics.auc(fpr,tpr), '.4f')+')')
    #    Ps += [P]; labels += ['(AUC: '+format(metrics.auc(fpr,tpr), '.4f')+')']
    #L = plt.legend(Ps, labels, loc='lower left', bbox_to_anchor=(0.29,0), fontsize=13, ncol=2,
    #               columnspacing=-2, facecolor='ghostwhite', framealpha=1)
    pylab.xlim(0,100); pylab.ylim(1,1e5)
    plt.yscale('log')
    plt.xticks(np.linspace(0,100,11))
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    pos    = [10**int(n) for n in np.arange(0,int(np.log10(axes.get_ylim()[1]))+1)]
    labels = ['1'] + ['$10^{'+str(n)+'}$' for n in np.arange(1,int(np.log10(axes.get_ylim()[1]))+1)]
    plt.yticks(pos, labels)
    plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.ylabel('$1/\epsilon_{\operatorname{bkg}}$', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=15)
    handles, labels = plt.gca().get_legend_handles_labels()
    #order = [0,1,2,3,4,10,11,12,13,14]
    #order = [0,1,5,6]
    #plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
    #           bbox_to_anchor=(0,0), loc='lower left', fontsize=13, ncol=2,
    #           columnspacing=-2, facecolor='ghostwhite', framealpha=1).set_zorder(10)
    #plt.gca().add_artist(L)
    plt.legend(loc='best', fontsize=14, ncol=2, columnspacing=-2,
               facecolor='ghostwhite', framealpha=1).set_zorder(10)
    file_name = output_dir + '/' + 'ROC_curves.png'
    print('Saving ROC curves to:', file_name); plt.savefig(file_name); sys.exit()


def ROC_curves(y_true, X_losses, weights, metrics_list, output_dir, wps=[1,10]):
    color_dict = {'MSE' :'tab:orange', 'MAE'   :'tab:brown', 'X-S'   :'tab:purple',
                   'JSD':'tab:cyan'  , 'EMD'   :'tab:green', 'KSD'   :'black'     ,
                   'KLD':'tab:red'   , 'Latent':'tab:blue' , 'Inputs':'gray', 'Inputs_scaled':'black'}
    metrics_dict = ROC_rates(y_true, X_losses, weights, metrics_list)

    if False:
        #fpr, tpr, _ = metrics_dict['Latent']
        #pickle.dump({'fpr':fpr, 'tpr':tpr}, open(output_dir+'/'+'pos_rates.pkl','wb'), protocol=4)

        file_path  = '/lcg/storage20/atlas/godin/ATLAS-VAE/e-ID_framework'
        file_path += '/outputs/top_100const/class_0_vs_bkg/pos_rates.pkl'
        fpr, tpr = pickle.load(open(file_path, 'rb')).values()
        fpr_0 = np.sum(fpr==0)
        color_dict['Top (supervised)'] = 'tab:blue'
        metrics_dict['Top (supervised)'] = 100*fpr[fpr_0:], 100*tpr[fpr_0:], None

        file_path  = '/lcg/storage20/atlas/godin/ATLAS-VAE/OE-VAE'
        file_path += '/outputs/Delphes/KLD-beta1_lamb100_X-S_8m_100const/plots/pos_rates.pkl'
        fpr, tpr = pickle.load(open(file_path, 'rb')).values()
        fpr_0 = np.sum(fpr==0)
        color_dict['Top (OE-VAE)'] = 'tab:blue'
        metrics_dict['Top (OE-VAE)'] = fpr[fpr_0:], tpr[fpr_0:], None

        file_path  = '/lcg/storage20/atlas/godin/ATLAS-VAE/e-ID_framework'
        file_path += '/outputs/2HDM_100const/class_0_vs_bkg/pos_rates.pkl'
        fpr, tpr = pickle.load(open(file_path, 'rb')).values()
        fpr_0 = np.sum(fpr==0)
        color_dict['2HDM (supervised)'] = 'tab:cyan'
        metrics_dict['2HDM (supervised)'] = 100*fpr[fpr_0:], 100*tpr[fpr_0:], None

        color_dict['2HDM (OE-VAE)'] = color_dict.pop('Latent')
        color_dict['2HDM (OE-VAE)'] = 'tab:cyan'
        metrics_dict['2HDM (OE-VAE)'] = metrics_dict.pop('Latent')

        ls_dict = {'Top (supervised)':'--', 'Top (OE-VAE)':'-', '2HDM (supervised)':'--', '2HDM (OE-VAE)':'-'}

    """ Background rejection plot """
    plt.figure(figsize=(13,8)); pylab.grid(True); axes = plt.gca()
    bkg_rej_max = 0
    for metric in metrics_dict:
        fpr, tpr, _ = metrics_dict[metric]
        bkg_rej_max = max(bkg_rej_max, np.max(100/fpr))
        #plt.text(0.98, 0.65-4*metrics_list.index(metric)/100, 'AUC: '+format(metrics.auc(fpr,tpr)/1e4, '.3f'),
        #         {'color':color_dict[metric], 'fontsize':14}, va='center', ha='right', transform=axes.transAxes)
        label = metric if metric!='Inputs_scaled' else 'Inputs (scaled)'
        plt.plot(tpr, 100/fpr, label=label, lw=2, color=color_dict[metric])
    for metric in metrics_dict:
        fpr, tpr, _ = metrics_dict[metric]
        plt.plot(np.nan, np.nan, '.', ms=0, label='(AUC: '+format(metrics.auc(fpr,tpr)/1e4, '.3f')+')')
    metrics_scores = [metrics_dict[metric][:2] for metric in metrics_dict]
    y_max = 10**(np.ceil(np.log10(bkg_rej_max)))
    pylab.xlim(0,100); pylab.ylim(1,y_max)
    for wp in wps:
        fpr_list = np.array([fpr[np.argwhere(tpr >= wp)[0]] for fpr,tpr in metrics_scores])
        if np.all(100/fpr_list < bkg_rej_max):
            score = 100/np.min(fpr_list)
            color = color_dict[metrics_list[np.argmin(fpr_list)]]
            axes.axhline(score, xmin=wp/100, xmax=1, ls='--', linewidth=1., color='dimgray')
            plt.text(100.4, score, str(int(score)), {'color':color, 'fontsize':14}, va="center", ha="left")
            axes.axvline(wp, ymin=0, ymax=np.log(score)/np.log(axes.get_ylim()[1]),
                         ls='--', linewidth=1., color='dimgray')
    plt.yscale('log')
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    pos    = [10**int(n) for n in np.arange(0,int(np.log10(axes.get_ylim()[1]))+1)]
    labels = ['1'] + ['$10^{'+str(n)+'}$' for n in np.arange(1,int(np.log10(axes.get_ylim()[1]))+1)]
    plt.yticks(pos, labels)
    plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.ylabel('$1/\epsilon_{\operatorname{bkg}}$', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(loc='upper right', fontsize=15, ncol=2, columnspacing=-2,
               facecolor='ghostwhite', framealpha=1).set_zorder(10)
    file_name = output_dir + '/' + 'bkg_rejection.png'
    print('Saving bkg rejection     to:', file_name); plt.savefig(file_name)
    """ Signal gain plot """
    plt.figure(figsize=(13,8)); pylab.grid(True); axes = plt.gca()
    max_ROC_var = 0
    for metric in metrics_dict:
        fpr, tpr, _ = metrics_dict[metric]
        ROC_var  = tpr/fpr
        max_ROC_var = max(max_ROC_var, np.max(ROC_var[tpr>=1]))
        label = metric if metric!='Inputs_scaled' else 'Inputs (scaled)'
        plt.plot(tpr, ROC_var, label=label, lw=2, color=color_dict[metric])
        #plt.plot(tpr, ROC_var, label=metric, lw=2, color=color_dict[metric], ls=ls_dict[metric])
    #pylab.xlim(1,100)
    #pylab.ylim(0,np.ceil(max_ROC_var))
    #plt.xscale('log')
    #plt.xticks([1,10,100], ['1','10', '100'])
    #axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    pylab.xlim(0,100)
    pylab.ylim(1,1e6)
    plt.yscale('log')
    location = 'upper right'
    plt.xlabel('$\epsilon_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.ylabel('$G_{S/B}=\epsilon_{\operatorname{sig}}/\epsilon_{\operatorname{bkg}}$', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(loc='best', fontsize=15, ncol=1, facecolor='ghostwhite', framealpha=1).set_zorder(10)
    file_name = output_dir + '/' + 'signal_gain.png'
    print('Saving signal gain       to:', file_name); plt.savefig(file_name)
    """ Significance plot """
    plt.figure(figsize=(13,8)); pylab.grid(True); axes = plt.gca()
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
    axes.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(loc='upper left', fontsize=15, ncol=2 if len(metrics_list)<9 else 3)
    file_name = output_dir + '/' + 'significance.png'
    print('Saving significance      to:', file_name); plt.savefig(file_name)


def ROC_rates(y_true, X_losses, weights, metrics_list):
    manager      = mp.Manager(); return_dict = manager.dict()
    arguments    = [(y_true, X_losses[metric], weights, metric, return_dict) for metric in metrics_list]
    processes    = [mp.Process(target=get_rates, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    return {metric:return_dict[metric] for metric in metrics_list}


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


'''
def combined_plots(n_test, n_top, output_dir, plot_var, n_dims=4, n_constituents=20):
    sample_topo = make_sample(n_constituents, 'QCD-topo', 'top-topo', n_test, n_top, n_dims, adjust_weights=True)
    sample_UFO  = make_sample(n_constituents, 'QCD-UFO' , 'BSM'     , n_test, n_top, n_dims, adjust_weights=True)
    samples = [sample_UFO, sample_topo]
    plot_distributions(samples, output_dir, plot_var, sig_bins=200, bkg_bins=400, sig_tag='top')
    sys.exit()
'''

'''
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
'''
