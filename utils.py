import numpy           as np
import multiprocessing as mp
import sys, h5py, time, pickle, warnings
from   sklearn         import preprocessing, metrics
from   scipy.spatial   import distance
from   energyflow      import emd


def make_sample(bkg_idx, sig_idx, bkg_cuts, sig_cuts, bkg, sig, normalization=True):
    data_path  = '/opt/tmp/godin/AD_data'
    data_files = {'qcd':'AtlasMCdijet36.h5', 'W':'resamples_oe_w.h5', 'top':'AtlasMCttbar.h5'}
    bkg_file   = data_path + '/' + data_files[bkg]
    sig_file   = data_path + '/' + data_files[sig]
    if np.isscalar(bkg_idx): bkg_idx = (0, bkg_idx)
    if np.isscalar(sig_idx): sig_idx = (0, sig_idx)
    if sig_idx[0] != sig_idx[1]: sig_sample = load_data(sig_file, 'constituents', sig_idx, cuts=sig_cuts)
    if bkg_idx[0] != bkg_idx[1]: bkg_sample = load_data(bkg_file, 'constituents', bkg_idx, cuts=bkg_cuts)
    if   'sig_sample' not in locals(): sample = bkg_sample
    elif 'bkg_sample' not in locals(): sample = sig_sample
    else: sample = {key:np.concatenate([sig_sample[key], bkg_sample[key]]) for key in bkg_sample}
    if normalization: sample['jets'] /= sample['pt'][:,np.newaxis]
    #for key in sample: print(key, sample[key].shape, sample[key].dtype)
    return sample


def load_data(data_file, data_key, idx, n_constituents=20, cuts='', multiprocess=True):
    idx = (int(idx[0]),int(idx[1]))
    if multiprocess:
        def get_data(data_file, data_key, idx, n_constituents, return_dict):
            return_dict[idx] = h5py.File(data_file,"r")[data_key][idx[0]:idx[1],:4*n_constituents]
        size       = min(len(h5py.File(data_file,"r")[data_key]),idx[1]-idx[0])
        idx_tuples = get_idx(size, start_val=idx[0], n_sets=mp.cpu_count())
        manager    = mp.Manager(); return_dict = manager.dict()
        arguments  = [(data_file, data_key, idx, n_constituents, return_dict) for idx in idx_tuples]
        processes  = [mp.Process(target=get_data, args=arg) for arg in arguments]
        for task in processes: task.start()
        for task in processes: task.join()
        sample = np.concatenate([return_dict[idx] for idx in idx_tuples])
    else:
        sample = h5py.File(data_file,"r")[data_key][idx[0]:idx[1],:4*n_constituents]
    sample = {'jets':sample, **{key:val for key,val in jets_4v(sample).items()}}
    data   = h5py.File(data_file,"r")
    if 'weights' in data: sample['weights'] = data['weights'][idx[0]:idx[1]]
    else                : sample['weights'] =         np.full(len(sample['jets']), 1)
    if   'JZW'   in data: sample[  'JZW'  ] = data[  'JZW'  ][idx[0]:idx[1]]
    else                : sample[  'JZW'  ] =         np.full(len(sample['jets']),-1)
    sample['weights'] = sample['weights']*weights_factors(sample['JZW'], data_file)
    if cuts != '': sample = {key:sample[key][eval(cuts)] for key in sample}
    return sample


def reweight_sample(sample, sig_bins, bkg_bins, weights='X-section', sig_frac=0.1, density=True):
    sig, bkg = sample['JZW']==-1, sample['JZW']>=0
    if weights == None:
        sample['weights'] = np.ones(len(sample['weights']))
    if weights == 'flat_pt' and np.any(sig):
        sample['weights'][sig]  = pt_weighting(sample['pt'][sample['JZW']==-1], sig_bins, density=density)
    if weights == 'flat_pt' and np.any(bkg):
        sample['weights'][bkg]  = pt_weighting(sample['pt'][sample['JZW']>= 0], bkg_bins, density=density)
    if np.any(sig) and np.any(bkg):
        sample['weights'][sig] *= np.sum(sample['weights'][bkg])/np.sum(sample['weights'][sig])
        sample['weights'][sig] *= sig_frac/(1-sig_frac)
    #print( np.sum(np.isfinite(sample['weights'])==False), np.sum(sample['weights']<=0) )
    #print( np.min(sample['weights']), np.max(sample['weights']) )
    return sample


def pt_weighting(pt, n_bins, density=True):
    bin_width = (np.max(pt) - np.min(pt)) / n_bins
    pt_bins   = list(np.arange(np.min(pt), np.max(pt), bin_width)) + [np.max(pt)+1e-3]
    pt_idx    = np.digitize (pt, pt_bins, right=False) -1
    hist_pt   = np.histogram(pt, pt_bins, density=density)[0]
    weights   = 1/hist_pt[pt_idx]
    return weights*len(weights)/np.sum(weights)


def weights_factors(JZW, data_file, lum=36):
    if np.any(JZW == -1):
        factors = len(h5py.File(data_file,"r")['constituents'])/len(JZW)
    else:
        #data_path  = '/opt/tmp/godin/AD_data/16-bit'
        #data_files = sorted([data_path+'/'+h5_file for h5_file in os.listdir(data_path) if '.h5' in h5_file])
        #n_JZW      = [len(h5py.File(data_files[n],"r")['constituents']) for n in np.arange(len(data_files))]
        n_JZW = [  35596, 13406964, 15909276, 17831457, 15981239,
                15997303, 13913843, 13983297, 15946135, 15993849]
        factors = np.ones(len(JZW))
        for n in np.arange(len(n_JZW)): factors = np.where(JZW==n, lum*n_JZW[n]/np.sum(JZW==n), factors)
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
    E  = np.float16(E); pt = np.float16(pt); M = np.float16(M)
    if idx == (0,None): return {'E':E, 'pt':pt, 'M':M}
    else: return_dict[idx]  =  {'E':E, 'pt':pt, 'M':M}


def JSD(P, Q, idx, return_dict, reshape=False):
    if reshape:
        P, Q = np.reshape(P, (-1,int(P.shape[1]/4),4)), np.reshape(Q, (-1,int(Q.shape[1]/4),4))
        return_dict[idx] = [np.mean([distance.jensenshannon(P[n,:,m], Q[n,:,m]) for m in np.arange(4)])
                            for n in np.arange(idx[0],idx[1])]
    else: return_dict[idx] = [distance.jensenshannon(P[n,:], Q[n,:]) for n in np.arange(idx[0],idx[1])]
def EMD(P, Q, idx, return_dict, n_iter=int(1e7)):
    jets_3v_pairs    = zip(jets_3v(P[idx[0]:idx[1]]), jets_3v(Q[idx[0]:idx[1]]))
    return_dict[idx] = [emd.emd(P, Q, return_flow=False, n_iter_max=n_iter) for P, Q in jets_3v_pairs]
def jets_3v(sample, idx=[0,None]):
    sample = np.float32(sample[idx[0]:idx[1]])
    sample = np.reshape(sample, (-1,int(sample.shape[1]/4),4))
    px, py, pz, E = [sample[...,n] for n in np.arange(4)]
    pt = np.sqrt(px**2 + py**2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        y = np.nan_to_num(np.log((E+pz)/(E-pz))/2, nan=0)
    phi = np.arctan2(py, px)
    return np.concatenate([n[...,np.newaxis] for n in [pt, y, phi]], axis=2)


def loss_function(P, Q, metric, delta=1e-16):
    P, Q = np.maximum(np.float64(P), delta), np.maximum(np.float64(Q), delta)
    if metric == 'JSD' or metric == 'EMD':
        idx_tuples = get_idx(len(P), n_sets=mp.cpu_count())
        target     = JSD if metric == 'JSD' else EMD
        manager    = mp.Manager(); return_dict = manager.dict()
        processes  = [mp.Process(target=target, args=(P, Q, idx, return_dict)) for idx in idx_tuples]
        for task in processes: task.start()
        for task in processes: task.join()
        loss = np.concatenate([return_dict[idx] for idx in idx_tuples])
    if metric == 'MSE'   : loss = np.mean(      (P - Q)**2, axis=1)
    if metric == 'MAE'   : loss = np.mean(np.abs(P - Q)   , axis=1)
    if metric == 'KLD'   : loss = np.mean(P*np.log(P/Q)   , axis=1)
    if metric == 'X-S'   : loss = np.mean(Q*np.log(1/P)   , axis=1)
    if metric == 'DeltaE': loss = jets_4v(P)['E'] - jets_4v(Q)['E']
    return loss


def fit_scaler(sample, scaler_out, reshape=True):
    print('Fitting quantile transform', end=' --> ', flush=True); start_time = time.time()
    if reshape: sample = np.reshape(sample, (-1,4))
    scaler = preprocessing.QuantileTransformer(n_quantiles=10000).fit(sample)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    print('Saving scaler file in', scaler_out)
    pickle.dump(scaler, open(scaler_out, 'wb'))
    return scaler
def apply_scaler(sample, scaler, reshape=True):
    print('Applying quantile transform', end=' --> ', flush=True); start_time = time.time()
    shape = sample.shape
    if reshape: sample = np.reshape(sample, (-1,4))
    sample = scaler.transform(sample)
    sample = np.reshape(sample, shape)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return sample
def inverse_scaler(sample, scaler, reshape=True):
    print('\nInversing quantile transform', end=' --> ', flush=True); start_time = time.time()
    shape = sample.shape
    if reshape: sample = np.reshape(sample, (-1,4))
    sample = scaler.inverse_transform(sample)
    sample = np.reshape(sample, shape)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return sample


def best_cut(y_true, X_loss, X_mass, weights, cuts=''):
    cuts = '(X_mass >= 140) & (X_mass <= 200)'
    if cuts != '':
        y_true  = y_true [eval(cuts)]
        X_loss  = X_loss [eval(cuts)]
        weights = weights[eval(cuts)]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, X_loss, pos_label=0, sample_weight=weights)
    len_0      = np.sum(fpr==0)
    thresholds = thresholds[len_0:]       [tpr[len_0:]>0.01]
    #ratios     = (tpr[len_0:]/fpr[len_0:])[tpr[len_0:]>0.01]
    ratios     = (tpr[len_0:]/np.sqrt(fpr[len_0:]))[tpr[len_0:]>0.01]
    return thresholds[np.argmax(ratios)]


def apply_cut(y_true, X_true, X_pred, sample, metric):
    X_loss   = loss_function(X_true, X_pred, metric=metric)
    loss_cut = best_cut(y_true, X_loss, sample['M'], sample['weights'])
    print('Best', metric, 'cut:', metric, '>=', format(loss_cut, '.3f'))
    sample = {key:sample[key][X_loss > loss_cut] for key in sample}
    y_true = y_true[X_loss > loss_cut]
    return sample, y_true


def get_idx(size, start_val=0, n_sets=8):
    n_sets   = min(size, n_sets)
    idx_list = [start_val + n*(size//n_sets) for n in np.arange(n_sets)] + [start_val+size]
    return list(zip(idx_list[:-1], idx_list[1:]))
