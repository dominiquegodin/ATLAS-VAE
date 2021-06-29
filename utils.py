import numpy           as np
import multiprocessing as mp
import sys, h5py, time, pickle, warnings
from   sklearn       import preprocessing, metrics
from   scipy.spatial import distance
from   energyflow    import emd
from   functools import partial


def make_sample(bkg_idx, sig_idx, n_dims, metrics, train_var, n_jets, bkg_cuts, sig_cuts, bkg, sig):
    data_path  = '/opt/tmp/godin/AD_data'
    data_files = {'qcd':'Atlas_MC_dijet.h5', 'W':'resamples_oe_w.h5', 'top':'Atlas_MC_ttbar.h5'}
    bkg_file   = data_path + '/' + data_files[bkg]
    sig_file   = data_path + '/' + data_files[sig]
    if np.isscalar(bkg_idx): bkg_idx = (0, bkg_idx)
    if np.isscalar(sig_idx): sig_idx = (0, sig_idx)
    if sig_idx[0] != sig_idx[1]:
        sig_sample = load_data(sig_file, sig, sig_idx, metrics, train_var, n_jets, cuts=sig_cuts)
    if bkg_idx[0] != bkg_idx[1]:
        bkg_sample = load_data(bkg_file, bkg, bkg_idx, metrics, train_var, n_jets, cuts=bkg_cuts)
    if   'sig_sample' not in locals():
        sample = bkg_sample
    elif 'bkg_sample' not in locals():
        sample = sig_sample
    else:
        if sig=='W' and False: sig_sample = upsampling(sig_sample, len(list(bkg_sample.values())[0]))
        sample = {key:np.concatenate([sig_sample[key], bkg_sample[key]])
                  for key in set(sig_sample.keys()) & set(bkg_sample.keys())}
    if 'jets' in sample:
        #sample['jets'] /= sample['pt'][:,np.newaxis] #pt_scaling
        if n_dims == 3:
            shape = sample['jets'].shape
            sample['jets'] = np.reshape(sample['jets']        , (-1,shape[1]//4,4))
            sample['jets'] = np.reshape(sample['jets'][...,1:], (shape[0],-1)     )
    return sample


def load_data(data_file, sample_type, idx, metrics, train_var, n_jets, cuts=''):
    print('Loading', format(sample_type,'^3s'), 'sample', end=' ', flush=True)
    data     = h5py.File(data_file,"r"); start_time = time.time()
    data_var = set(train_var+['pt','M','weights','JZW']) & set(data)
    sample   = {key:data[key][idx[0]:idx[1]] for key in data_var if key!='jets'}
    if 'jets' in data_var: sample['jets'] = data['jets'][idx[0]:idx[1],:4*n_jets]
    if sample_type == 'W':
        sample['jets'] = data['constituents'][idx[0]:idx[1],:4*n_jets]
        sample.update({key:val for key,val in jets_4v(sample['jets']).items()})
    #sample['jets'] = data['jets'][idx[0]:idx[1],:4*n_jets]
    #sample.update({key:val for key,val in jets_4v(sample['jets']).items()})
    if 'DNN' in metrics and sample_type != 'W':
        sample.update({'DNN':data['rljet_topTag_DNN19_qqb_score'][idx[0]:idx[1]]})
    if 'FCN' in metrics and sample_type != 'W':
        if sample_type == 'top': file_name = 'FCN_tagger_ttbar.pkl'
        if sample_type == 'qcd': file_name = 'FCN_tagger_dijet.pkl'
        probs = pickle.load(open('/opt/tmp/godin/AD_data'+'/'+file_name,'rb'))
        sample.update({'FCN':probs[idx[0]:idx[1]]})
    if 'weights' not in sample: sample['weights'] = np.full_like(sample['pt'], 1)
    if   'JZW'   not in sample: sample[  'JZW'  ] = np.full_like(sample['pt'],-1)
    sample['weights'] = sample['weights']*weights_factors(sample['JZW'], data_file)
    cut_list = [np.full_like(sample['weights'], True)]
    for cut in cuts:
        try: cut_list.append(eval(cut))
        except KeyError: pass
    cuts = np.logical_and.reduce(cut_list)
    if not np.all(cuts): sample = {key:sample[key][cuts] for key in sample}
    if len(set(train_var)-{'jets'}) != 0:
        sample['scalars'] = np.hstack([sample.pop(key)[:,np.newaxis] for key in train_var if key!='jets'])
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return sample


def upsampling(sample, target_size):
    sample_size = len(list(sample.values())[0])
    indices = np.random.choice(np.arange(sample_size), target_size, replace=sample_size<target_size)
    return {key: np.take(sample[key], indices, axis=0) for key in sample}


def reweight_sample(sample, sig_bins, bkg_bins, weight_type='X-S', sig_frac=0.1, density=True):
    sig, bkg = sample['JZW']==-1, sample['JZW']>=0
    if weight_type == None:
        sample['weights'] = np.ones_like(sample['weights'])
    if weight_type == 'flat_pt':
        if np.any(sig): sample['weights'][sig] = pt_weighting(sample['pt'][sig], sig_bins, density=density)
        if np.any(bkg): sample['weights'][bkg] = pt_weighting(sample['pt'][bkg], bkg_bins, density=density)
    if np.any(sig) and np.any(bkg):
        sample['weights'][sig] *= np.sum(sample['weights'][bkg])/np.sum(sample['weights'][sig])
        sample['weights'][sig] *= sig_frac/(1-sig_frac)
    return sample


def pt_weighting(pt, n_bins, density=True):
    pt        = np.float32(pt)
    bin_width = (np.max(pt) - np.min(pt)) / n_bins
    pt_bins   = [np.min(pt) + k*bin_width for k in np.arange(n_bins+1)]
    pt_idx    = np.minimum(np.digitize(pt, pt_bins, right=False), len(pt_bins)-1) -1
    hist_pt   = np.histogram(pt, pt_bins, density=density)[0]
    weights   = 1/hist_pt[pt_idx]
    return weights*len(weights)/np.sum(weights)


def weights_factors(JZW, data_file):
    if np.any(JZW == -1):
        factors = len(list(h5py.File(data_file,"r").values())[0])/len(JZW)
    else:
        n_JZW = [  35596, 13406964, 15909276, 17831457, 15981239,
                15997303, 13913843, 13983297, 15946135, 15993849]
        factors = np.ones_like(JZW)
        for n in np.arange(len(n_JZW)):
            if np.sum(JZW==n) != 0: factors[JZW==n] = n_JZW[n]/np.sum(JZW==n)
    return factors


def jets_4v(sample):
    idx_tuples = get_idx(len(sample)); manager = mp.Manager(); return_dict = manager.dict()
    processes  = [mp.Process(target=get_4v, args=(sample, idx, return_dict)) for idx in idx_tuples]
    for task in processes: task.start()
    for task in processes: task.join()
    return {key:np.concatenate([return_dict[idx][key] for idx in idx_tuples]) for key in ['E','pt','M']}
def get_4v(sample, idx=(0,None), return_dict=None):
    sample = np.float32(sample[idx[0]:idx[1]])
    sample = np.sum(np.reshape(sample, (-1,int(sample.shape[1]/4),4)), axis=1)
    E, px, py, pz = [sample[:,n] for n in np.arange(sample.shape[-1])]
    pt = np.sqrt(px**2 + py**2)
    M  = np.sqrt(np.maximum(0, E**2 - px**2 - py**2 - pz**2))
    if idx == (0,None): return {'E':E, 'pt':pt, 'M':M}
    else: return_dict[idx]  =  {'E':E, 'pt':pt, 'M':M}


def JSD(P, Q, idx, n_dims, return_dict, reshape=False):
    if reshape:
        P, Q = np.reshape(P, (-1,int(P.shape[1]/n_dims),n_dims)), np.reshape(Q, (-1,int(Q.shape[1]/n_dims),n_dims))
        return_dict[idx] = [np.mean([distance.jensenshannon(P[n,:,m], Q[n,:,m]) for m in np.arange(n_dims)])
                            for n in np.arange(idx[0],idx[1])]
    else: return_dict[idx] = [distance.jensenshannon(P[n,:], Q[n,:]) for n in np.arange(idx[0],idx[1])]


def EMD(P, Q, idx, n_dims, return_dict, n_iter=int(1e7)):
    jets_3v_pairs    = zip(jets_3v(P[idx[0]:idx[1]],n_dims), jets_3v(Q[idx[0]:idx[1]],n_dims))
    return_dict[idx] = [emd.emd_pot(P, Q, return_flow=False, n_iter_max=n_iter) for P, Q in jets_3v_pairs]
def jets_3v(sample, n_dims, idx=[0,None]):
    sample = np.float32(sample[idx[0]:idx[1]])
    sample = np.reshape(sample, (-1,int(sample.shape[1]/n_dims),n_dims))
    if n_dims == 3:
        px, py, pz = [sample[...,n] for n in np.arange(n_dims)]
        E  = np.sqrt(px**2 + py**2 + pz**2)
    else:
        E, px, py, pz = [sample[...,n] for n in np.arange(n_dims)]
    pt = np.sqrt(px**2 + py**2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        y = np.nan_to_num(np.log((E+pz)/(E-pz))/2, nan=0)
    phi = np.arctan2(py, px)
    return np.concatenate([n[...,np.newaxis] for n in [pt, y, phi]], axis=2)


def loss_function(P, Q, n_dims, metric, X_losses=None, delta=1e-16, multiloss=True):
    if metric in ['KLD', 'X-S', 'JSD']:
        P, Q = np.maximum(np.float64(P), delta), np.maximum(np.float64(Q), delta)
    if metric in ['B-1', 'B-2']: loss = np.mean(P, axis=1)
    if metric in ['DNN', 'FCN']: loss = P
    if metric in ['JSD', 'EMD']:
        idx_tuples = get_idx(len(P), n_sets=mp.cpu_count())
        target     = JSD if metric == 'JSD' else EMD
        manager    = mp.Manager(); return_dict = manager.dict()
        processes  = [mp.Process(target=target, args=(P, Q, idx, n_dims, return_dict)) for idx in idx_tuples]
        for task in processes: task.start()
        for task in processes: task.join()
        loss = np.concatenate([return_dict[idx] for idx in idx_tuples])
    if metric == 'MSE': loss = np.mean(      (P - Q)**2, axis=1)
    if metric == 'MAE': loss = np.mean(np.abs(P - Q)   , axis=1)
    if metric == 'KLD': loss = np.mean(P*np.log(P/Q)   , axis=1)
    if metric == 'X-S': loss = np.mean(P*np.log(1/Q)   , axis=1)
    if multiloss: X_losses[metric] = loss
    else: return loss


def fit_scaler(sample, n_dims, scaler_out, reshape=True):
    print('Fitting quantile transform', end=' ', flush=True); start_time = time.time()
    if reshape: sample = np.reshape(sample, (-1,n_dims))
    #scaler = preprocessing.QuantileTransformer(output_distribution='uniform', n_quantiles=10000, random_state=0)
    scaler = preprocessing.MaxAbsScaler()
    scaler.fit(sample)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    print('Saving quantile transform to', scaler_out)
    pickle.dump(scaler, open(scaler_out, 'wb'))
    return scaler
def s_scaling(sample, scaler):
    print('Applying scalars quantile transform', end=' ', flush=True); start_time = time.time()
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    sample = scaler.transform(sample)
    return sample
def apply_t_scaler(sample, n_dims, scaler, idx=(0,None), return_dict=None, reshape=True):
    if idx == (0,None):
        print('Applying topo-clusters quantile transform', end=' ', flush=True); start_time = time.time()
    shape = sample.shape
    if reshape: sample = np.reshape(sample, (-1,n_dims))
    sample = scaler.transform(sample)
    sample = np.reshape(sample, shape)
    if idx == (0,None):
        print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
        return sample
    else: return_dict[idx] = sample
def t_scaling(sample, n_dims, scaler, reshape=True):
    print('Applying topo-clusters quantile transform', end=' ', flush=True); start_time = time.time()
    idx_tuples = get_idx(len(sample), start_val=0, n_sets=mp.cpu_count())
    manager    = mp.Manager(); return_dict = manager.dict()
    arguments  = [(sample[idx[0]:idx[1]], n_dims, scaler, idx, return_dict, reshape) for idx in idx_tuples]
    processes  = [mp.Process(target=apply_t_scaler, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    sample = np.concatenate([return_dict[idx] for idx in idx_tuples])
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return sample
'''
def apply_scaling(sample, scaler, reshape=True):
    from functools import partial; from multiprocessing import sharedctypes, Pool
    global fill_sample
    def fill_sample(scaler, reshape, idx):
        tmp = np.ctypeslib.as_array(shared_array)
        scaled = sample[idx[0]:idx[1],:]
        shape  = scaled.shape
        if reshape: scaled = np.reshape(scaled, (-1,4))
        scaled = scaler.transform(scaled)
        tmp[idx[0]:idx[1],:] = np.reshape(scaled, shape)
    print('Applying quantile transform', end=' ', flush=True); start_time = time.time()
    scaled_sample = np.ctypeslib.as_ctypes(np.empty(sample.shape, dtype=np.float32))
    shared_array  = sharedctypes.RawArray(scaled_sample._type_, scaled_sample)
    idx_tuples    = get_idx(len(sample), start_val=0, n_sets=mp.cpu_count())
    pool = Pool(); pool.map(partial(fill_sample, scaler, reshape), idx_tuples)
    scaled_sample = np.ctypeslib.as_array(shared_array)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return scaled_sample
'''
def inverse_scaler(sample, n_dims, scaler, reshape=True):
    print('Inversing quantile transform', end=' ', flush=True); start_time = time.time()
    shape = sample.shape
    if reshape: sample = np.reshape(sample, (-1,n_dims))
    sample = scaler.inverse_transform(sample)
    sample = np.reshape(sample, shape)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return sample


def best_threshold(y_true, X_loss, X_mass, weights, cut_type='gain'):
    #cuts = '(X_mass >= 150) & (X_mass <= 190)'
    #    y_true  = y_true [eval(cuts)]
    #    X_loss  = X_loss [eval(cuts)]
    #    weights = weights[eval(cuts)]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, X_loss, pos_label=0, sample_weight=weights)
    len_0      = np.sum(fpr==0)
    thresholds = thresholds[len_0:][tpr[len_0:]>0.01]
    if cut_type=='gain' : cut_values = (tpr[len_0:]/fpr[len_0:])         [tpr[len_0:]>0.01]
    if cut_type=='sigma': cut_values = (tpr[len_0:]/np.sqrt(fpr[len_0:]))[tpr[len_0:]>0.01]
    cut_index = np.argmax(cut_values)
    if cut_type=='sigma':
        n_sig = np.sum(weights[y_true==0])
        n_bkg = np.sum(weights[y_true==1])
        cut_values *= n_sig/np.sqrt(n_bkg)
    return thresholds[cut_index], cut_values[cut_index]


def apply_best_cut(y_true, X_true, X_pred, sample, n_dims, metric, cut_type='gain'):
    if metric == 'DNN': X_true = sample['DNN']
    if metric == 'FCN': X_true = sample['FCN']
    X_loss   = loss_function(X_true, X_pred, n_dims, metric, multiloss=False)
    loss_cut, loss_val = best_threshold(y_true, X_loss, sample['M'], sample['weights'], cut_type)
    print('Best', metric, 'cut:', metric, '>=', format(loss_cut, '.3f'), end= ' ; ')
    print(cut_type, '=', format(loss_val, '.2f'), '\n')
    sample = {key:sample[key][X_loss > loss_cut] for key in sample}
    return sample


def bump_hunter(y_true, sample, output_dir, m_range=[120,200]):
    import pyBumpHunter as BH #from BumpHunter.BumpHunter.bumphunter_1dim import BumpHunter1D
    data, data_weights   = sample['M']    , sample['weights']
    bkg , bkg_weights    = data[y_true==1], data_weights[y_true==1]
    data_hist, bin_edges = np.histogram(data, bins=50, range=m_range, weights=data_weights)
    bkg_hist , bin_edges = np.histogram(bkg , bins=50, range=m_range, weights= bkg_weights)
    start_time = time.time()
    hunter = BH.BumpHunter1D(rang=m_range, width_min=2, width_max=6, width_step=1, scan_step=1,
                             npe=10000, nworker=1, seed=0, bins=bin_edges) #BumpHunter1D class instance
    hunter.bump_scan(data_hist, bkg_hist, is_hist=True)
    hunter.plot_bump(data_hist, bkg_hist, is_hist=True, filename=output_dir+'/'+'bump_mc_weighted.png')
    hunter.print_bump_info()
    hunter.print_bump_true(data_hist, bkg_hist, is_hist=True)
    print(format(time.time() - start_time, '2.1f'), '\b'+' s')


def get_idx(size, start_val=0, n_sets=8):
    n_sets   = min(size, n_sets)
    idx_list = [start_val + n*(size//n_sets) for n in np.arange(n_sets)] + [start_val+size]
    return list(zip(idx_list[:-1], idx_list[1:]))
