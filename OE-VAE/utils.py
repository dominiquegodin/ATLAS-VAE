import numpy           as np
import tensorflow      as tf
import multiprocessing as mp
import sys, h5py, time, pickle, warnings
from   sklearn       import preprocessing, metrics
from   scipy.spatial import distance
from   scipy         import stats
from   energyflow    import emd


def make_datasets(train_sample, train_sample_OE, valid_sample, batch_size=1):
    train_X       = (train_sample   ['constituents'], np.float32(train_sample   ['weights']))
    train_X_OE    = (train_sample_OE['constituents'], np.float32(train_sample_OE['weights']))
    train_dataset = tf.data.Dataset.from_tensor_slices( train_X + train_X_OE )
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    valid_X       = (valid_sample   ['constituents'], np.float32(valid_sample   ['weights']))
    valid_dataset = tf.data.Dataset.from_tensor_slices( valid_X )
    valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset, valid_dataset


def make_sample(n_dims, n_constituents, bkg, sig, bkg_idx, sig_idx, bkg_cuts='', sig_cuts=''):
    data_path  = '/opt/tmp/godin/AD_data'
    data_files = {'qcd':'Atlas_MC_dijet.h5', 'W':'resamples_oe_w.h5', 'top':'Atlas_MC_ttbar.h5'}
    bkg_file   = data_path + '/' + data_files[bkg]
    sig_file   = data_path + '/' + data_files[sig]
    if np.isscalar(bkg_idx): bkg_idx = (0, bkg_idx)
    if np.isscalar(sig_idx): sig_idx = (0, sig_idx)
    if sig_idx[0] != sig_idx[1]:
        sig_sample = load_data(sig_file, sig, sig_idx, n_constituents, cuts=sig_cuts)
    if bkg_idx[0] != bkg_idx[1]:
        bkg_sample = load_data(bkg_file, bkg, bkg_idx, n_constituents, cuts=bkg_cuts)
    if   'sig_sample' not in locals():
        sample = bkg_sample
    elif 'bkg_sample' not in locals():
        sample = sig_sample
    else:
        if sig == 'W': sig_sample = upsampling(sig_sample, len(list(bkg_sample.values())[0]))
        sample = {key:np.concatenate([sig_sample[key], bkg_sample[key]])
                  for key in set(sig_sample) & set(bkg_sample)}
    # performing pt_scaling
    if 'constituents' in sample and False:
        sample['constituents'] = sample['constituents']/sample['pt'][:,np.newaxis]
    if 'constituents' in sample and n_dims == 3:
        shape = sample['constituents'].shape
        sample['constituents'] = np.reshape(sample['constituents']        , (-1,shape[1]//4,4))
        sample['constituents'] = np.reshape(sample['constituents'][...,1:], (shape[0],-1)     )
    return sample


def load_data(data_file, sample_type, idx, n_constituents, cuts):
    print('Loading', format(sample_type,'^3s'), 'sample', end=' ', flush=True)
    data     = h5py.File(data_file,"r"); start_time = time.time()
    #data_var = set(['pt','M','weights','JZW']) & set(data)
    sample   = {key:data[key][idx[0]:idx[1]] for key in data if key!='constituents'}
    sample['constituents'] = np.float32(data['constituents'][idx[0]:idx[1],:4*n_constituents])
    #if sample_type == 'W':
    #print()
    #print( sample_type, set(sample) & {'pt','M'}=={} )

    if len(set(sample) & {'pt','M'}) == 0:
        sample.update({key:val for key,val in jets_4v(sample['constituents']).items()})
    if   'JZW'   not in sample: sample[  'JZW'  ] = np.full(len(sample['constituents']),-1)
    if 'weights' not in sample: sample['weights'] = np.full(len(sample['constituents']), 1)
    cut_list = [np.full_like(sample['weights'], True)]
    for cut in cuts:
        try: cut_list.append(eval(cut))
        except KeyError: pass
    cuts = np.logical_and.reduce(cut_list)
    sample['weights'] = sample['weights']*weights_factors(sample['JZW'], data_file)
    if not np.all(cuts):
        sample = {key:sample[key][cuts] for key in sample}
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return sample


def upsampling(sample, target_size):
    source_size = len(list(sample.values())[0])
    indices = np.random.choice(source_size, target_size, replace=source_size<target_size)
    sample['weights'] = sample['weights']*source_size/target_size
    return {key:np.take(sample[key], indices, axis=0) for key in sample}


def reweight_sample(sample, sig_bins, bkg_bins, weight_type='X-S'):
    sig, bkg = sample['JZW']==-1, sample['JZW']>=0
    if weight_type == None or weight_type == 'none':
        if np.any(sig):
            sig_weights = sample['weights'][sig]
            sample['weights'][sig] = np.full(len(sig_weights), np.sum(sig_weights)/len(sig_weights))
        if np.any(bkg):
            bkg_weights = sample['weights'][bkg]
            sample['weights'][bkg] = np.full(len(bkg_weights), np.sum(bkg_weights)/len(bkg_weights))
    if weight_type == 'flat_pt':
        if np.any(sig):
            sample['weights'][sig] = pt_weighting(sample['pt'][sig], sample['weights'][sig], sig_bins)
        if np.any(bkg):
            sample['weights'][bkg] = pt_weighting(sample['pt'][bkg], sample['weights'][bkg], bkg_bins)
    return sample


def pt_weighting(pt, xs_weights, n_bins, density=True):
    pt        = np.float32(pt)
    bin_width = (np.max(pt) - np.min(pt)) / n_bins
    pt_bins   = [np.min(pt) + k*bin_width for k in np.arange(n_bins+1)]
    pt_idx    = np.minimum(np.digitize(pt, pt_bins, right=False), len(pt_bins)-1) -1
    hist_pt   = np.histogram(pt, pt_bins, density=density)[0]
    weights   = 1/hist_pt[pt_idx]
    return weights*np.sum(xs_weights)/np.sum(weights)


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


def KST(P, Q, idx, n_dims, return_dict, reshape=False):
    if reshape:
        P, Q = np.reshape(P, (-1,int(P.shape[1]/n_dims),n_dims)), np.reshape(Q, (-1,int(Q.shape[1]/n_dims),n_dims))
        return_dict[idx] = [np.mean([stats.ks_2samp(P[n,:,m], Q[n,:,m])[0] for m in np.arange(n_dims)])
                            for n in np.arange(idx[0],idx[1])]
    else: return_dict[idx] = [stats.ks_2samp(P[n,:], Q[n,:])[0] for n in np.arange(idx[0],idx[1])]


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
    if metric in ['Inputs']: loss = np.mean(P, axis=1)
    if metric in ['DNN', 'FCN']: loss = P
    if metric in ['JSD', 'KSD', 'EMD']:
        idx_tuples = get_idx(len(P), n_sets=mp.cpu_count()//3)
        target     = JSD if metric=='JSD' else KST if metric=='KSD' else EMD
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


def fit_scaler(sample, n_dims, scaler_out, reshape=True, scaler_type='MaxAbsScaler'):
    print('\nFitting', scaler_type, 'scaler', end=' ', flush=True); start_time = time.time()
    if reshape: sample = np.reshape(sample, (-1,n_dims))
    if scaler_type == 'QuantileTransformer':
        scaler = preprocessing.QuantileTransformer(output_distribution='uniform', n_quantiles=10000, random_state=0)
    if scaler_type == 'MaxAbsScaler':
        scaler = preprocessing.MaxAbsScaler()
    if scaler_type == 'RobustScaler':
        scaler = preprocessing.RobustScaler()
    scaler.fit(sample)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    print('Saving scaling to', scaler_out)
    pickle.dump(scaler, open(scaler_out, 'wb'))
    return scaler
def apply_scaler(sample, n_dims, scaler, idx=(0,None), return_dict=None, reshape=True):
    if idx == (0,None):
        print('Applying scaler', end=' ', flush=True); start_time = time.time()
    shape = sample.shape
    if reshape: sample = np.reshape(sample, (-1,n_dims))
    sample = scaler.transform(sample)
    sample = np.reshape(sample, shape)
    if idx == (0,None):
        print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
        return sample
    else: return_dict[idx] = sample
def scaling(sample, n_dims, scaler, reshape=True):
    print('Applying scaler', end=' ', flush=True); start_time = time.time()
    idx_tuples = get_idx(len(sample), start_val=0, n_sets=mp.cpu_count())
    manager    = mp.Manager(); return_dict = manager.dict()
    arguments  = [(sample[idx[0]:idx[1]], n_dims, scaler, idx, return_dict, reshape) for idx in idx_tuples]
    processes  = [mp.Process(target=apply_scaler, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    sample = np.concatenate([return_dict[idx] for idx in idx_tuples])
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return sample
def inverse_scaler(sample, n_dims, scaler, reshape=True):
    print('Applying inverse scaler', end=' ', flush=True); start_time = time.time()
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
    #BumpHunter1D class instance
    hunter = BH.BumpHunter1D(rang=m_range, width_min=2, width_max=6, width_step=1, scan_step=1,
                             npe=10000, nworker=1, seed=0, bins=bin_edges)
    hunter.bump_scan(data_hist, bkg_hist, is_hist=True)
    hunter.plot_bump(data_hist, bkg_hist, is_hist=True, filename=output_dir+'/'+'bump_mc_weighted.png')
    hunter.print_bump_info()
    hunter.print_bump_true(data_hist, bkg_hist, is_hist=True)
    print(format(time.time() - start_time, '2.1f'), '\b'+' s')


def get_idx(size, start_val=0, n_sets=8):
    n_sets   = min(size, n_sets)
    idx_list = [start_val + n*(size//n_sets) for n in np.arange(n_sets)] + [start_val+size]
    return list(zip(idx_list[:-1], idx_list[1:]))
