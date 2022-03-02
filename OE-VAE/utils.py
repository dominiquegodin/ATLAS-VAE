import numpy           as np
import tensorflow      as tf
import multiprocessing as mp
import sys, h5py, time, pickle, warnings
from   sklearn       import preprocessing, metrics, utils
from   scipy.spatial import distance
from   scipy         import stats
from   energyflow    import emd


def make_datasets(sample, sample_OE, batch_size=1):
    X    = (sample   ['constituents'], sample   ['weights'])
    X_OE = (sample_OE['constituents'], sample_OE['weights'])
    dataset = tf.data.Dataset.from_tensor_slices( X + X_OE )
    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def make_sample(n_constituents, bkg_data, sig_data, bkg_idx=1, sig_idx=1, bkg_cuts='', sig_cuts='',
                n_dims=4, adjust_weights=False, DSIDs=[]):
    data_path  = '/opt/tmp/godin/AD_data'
    data_files = {'qcd-Geneva' :'formatted_converted_20210629_QCDjj_pT_450_1200_nevents_10M.h5'      ,
                  'top-Geneva' :'formatted_converted_20210430_ttbar_allhad_pT_450_1200_nevents_1M.h5',
                  'qcd-Delphes':'Delphes_dijet.h5'   ,
                  'top-Delphes':'Delphes_ttbar.h5'   ,
                  'qcd-topo'   :'Atlas_topo-dijet.h5',
                  'top-topo'   :'Atlas_topo-ttbar.h5',
                  'qcd-UFO'    :'Atlas_UFO-dijet.h5' ,
                  'top-UFO'    :'Atlas_UFO-ttbar.h5' ,
                  'BSM'        :'Atlas_BSM.h5'       ,
                  'W-OoD'      :'resamples_oe_w.h5'  ,
                  'H-OoD'      :'H_HpHm_generation_mh2_merged_mhc_merged_merged.h5'}
    bkg_file = data_path + '/' + data_files[bkg_data]
    sig_file = data_path + '/' + data_files[sig_data]
    if np.isscalar(bkg_idx): bkg_idx = (0, bkg_idx)
    if np.isscalar(sig_idx): sig_idx = (0, sig_idx)
    if sig_idx[0] != sig_idx[1]:
        sig_sample = load_data(sig_file, sig_data, sig_idx, n_constituents, adjust_weights, sig_cuts, DSIDs)
        if 'OoD' in sig_data and False:
            """ Merging W-OoD and H-OoD"""
            extra_sig    = 'W-OoD' if sig_data=='H-OoD' else 'H-OoD'
            extra_file   = data_path + '/' + data_files[extra_sig]
            extra_sample = load_data(extra_file, extra_sig, sig_idx, n_constituents, adjust_weights, sig_cuts)
            sig_sample   = {key:utils.shuffle(np.concatenate([sig_sample[key], extra_sample[key]]), random_state=0)
                            for key in set(sig_sample) & set(extra_sample)}
    if bkg_idx[0] != bkg_idx[1]:
        bkg_sample = load_data(bkg_file, bkg_data, bkg_idx, n_constituents, adjust_weights, bkg_cuts)
    if   'sig_sample' not in locals():
        sample = bkg_sample
    elif 'bkg_sample' not in locals():
        sample = sig_sample
    else:
        if 'OoD'in sig_data:
            sig_sample = upsampling(sig_sample, len(list(bkg_sample.values())[0]), adjust_weights)
        sample = {key:np.concatenate([sig_sample[key], bkg_sample[key]])
                  for key in set(sig_sample) & set(bkg_sample)}
    if False:
        """ Scaling (E, px, py, px) with jet pt """
        sample['constituents'] = sample['constituents']/sample['pt'][:,np.newaxis]
    if 'constituents' in sample and n_dims == 3:
        """ Using (px, py, px) instead of (E, px, py, px) """
        shape = sample['constituents'].shape
        sample['constituents'] = np.reshape(sample['constituents']        , (-1,shape[1]//4,4))
        sample['constituents'] = np.reshape(sample['constituents'][...,1:], (shape[0],-1)     )
    return sample


def load_data(data_file, data_type, idx, n_constituents, adjust_weights, cuts, DSIDs=[], bkg=False):
    print('Loading', format(data_type,'^11s'), 'sample', end=' ', flush=True)
    data     = h5py.File(data_file,"r"); start_time = time.time()
    var_list = ['pt_calo','m_calo','rljet_m_comb','rljet_pt_comb','weights','JZW','rljet_n_constituents','DSID']
    sample   = {key:data[key][idx[0]:idx[1]] for key in set(var_list)&set(data) if key!='constituents'}
    sample['constituents'] = np.float32(data['constituents'][idx[0]:idx[1],:4*n_constituents])
    if 4*n_constituents > data['constituents'].shape[1]:
        shape = sample['constituents'].shape
        zeros_array = np.zeros((shape[0], 4*n_constituents-shape[1]), dtype=np.float32)
        sample['constituents'] = np.hstack([sample['constituents'], zeros_array])
    if len(set(sample) & {'pt_calo','m_calo'}) == 0:
        sample.update({key:val for key,val in jets_4v(sample['constituents']).items()})
    sample['pt'] = sample.pop('rljet_pt_comb' if 'rljet_pt_comb' in sample else 'pt_calo')
    sample['m']  = sample.pop('rljet_m_comb'  if 'rljet_m_comb'  in sample else  'm_calo')
    if 'JZW' not in sample:
        sample['JZW'] = np.full(len(sample['constituents']), 0 if 'qcd' in data_type else -1, dtype=np.float32)
    if 'weights' not in sample:
        sample['weights'] = np.full(len(sample['constituents']), 1, dtype=np.float32)
    """ Adjusting weights for cross-section """
    if adjust_weights:
        sample['weights'] = sample['weights']*weights_factors(sample['JZW'], data_file)
    """ Applying sample cuts"""
    cut_list = [np.full_like(sample['weights'], True)]
    for cut in cuts:
        try: cut_list.append(eval(cut))
        except KeyError: pass
    cuts = np.logical_and.reduce(cut_list)
    if DSIDs != []:
        dsid_cuts = np.logical_or.reduce([sample['DSID']==int(n) for n in DSIDs])
        cuts = np.logical_and(cuts, dsid_cuts)
    if not np.all(cuts):
        sample = {key:sample[key][cuts] for key in sample}
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return sample


def separate_sample(sample):
    JZW        = sample['JZW']
    bkg_sample = {key:val[JZW!=-1] for key,val in sample.items()}
    sig_sample = {key:val[JZW==-1] for key,val in sample.items()}
    return bkg_sample, sig_sample


def upsampling(sample, target_size, adjust_weights):
    source_size = len(list(sample.values())[0])
    np.random.seed(0)
    indices = np.random.choice(source_size, target_size, replace=source_size<target_size)
    if adjust_weights:
        sample['weights'] = sample['weights']*np.float32(source_size/target_size)
    return {key:np.take(sample[key], indices, axis=0) for key in sample}


def reweight_sample(sample, bin_sizes, weight_type='X-S'):
    """ None   : no weights (weight=1)                    """
    """ X-S    : default cross-section weights            """
    """ flat_m : 1D flat weighting for OoD and background """
    """ flat_pt: 1D flat weighting for OoD and background """
    """ flat_2d: 2D flat weighting for OoD and background """
    """ OoD_m  : 1D weighting of OoD to background        """
    """ OoD_pt : 1D weighting of OoD to background        """
    """ OoD_2d : 2D weighting of OoD to background        """
    sig, bkg = sample['JZW']==-1, sample['JZW']>=0
    if weight_type == None or weight_type.lower() == 'none':
        sample['weights'][sig] = np.ones_like(sample['weights'][sig])
        sample['weights'][bkg] = np.ones_like(sample['weights'][bkg])
    if 'flat' in weight_type:
        sample['weights'][sig] = get_weights(sample, sig, bkg, bin_sizes, weight_type)
        sample['weights'][bkg] = get_weights(sample, bkg, bkg, bin_sizes, weight_type)
        sample['weights'][sig] = get_weights(sample, sig, bkg, bin_sizes, weight_type='2d')
    if 'OoD' in weight_type:
        sample['weights'][sig] = get_weights(sample, sig, bkg, bin_sizes, weight_type)
    return sample


def get_weights(sample, sig, bkg, bin_sizes, weight_type, density=True):
    m_size, pt_size = bin_sizes['m'], bin_sizes['pt']
    m_sig, pt_sig, weights_sig = sample['m'][sig], sample['pt'][sig], sample['weights'][sig]
    m_bkg, pt_bkg, weights_bkg = sample['m'][bkg], sample['pt'][bkg], sample['weights'][bkg]
    m_min, pt_min = np.min(m_sig), np.min(pt_sig)
    m_max, pt_max = np.max(m_sig), np.max(pt_sig)
    if 'm'  in weight_type: pt_size = pt_max+1
    if 'pt' in weight_type:  m_size =  m_max+1
    m_bins  = get_idx( m_max, bin_size= m_size, min_val= m_min, integer=False, tuples=False)
    pt_bins = get_idx(pt_max, bin_size=pt_size, min_val=pt_min, integer=False, tuples=False)
    m_idx   = np.clip(np.digitize( m_sig,  m_bins, right=False), 1, len( m_bins)-1) - 1
    pt_idx  = np.clip(np.digitize(pt_sig, pt_bins, right=False), 1, len(pt_bins)-1) - 1
    hist_sig = np.histogram2d(m_sig, pt_sig, bins=[m_bins,pt_bins], density=density)[0]
    if density: hist_sig *= len(m_sig)
    hist_sig = np.maximum(hist_sig, np.min(hist_sig[hist_sig!=0]) if density else 1)
    if 'flat' in weight_type:
        weights = (1/hist_sig)[m_idx, pt_idx]
        return weights*np.sum(weights_sig)/np.sum(weights)
    hist_bkg = np.histogram2d(m_bkg, pt_bkg, bins=[m_bins,pt_bins], weights=weights_bkg, density=density)[0]
    if density: hist_bkg *= len(m_bkg)
    weights = (hist_bkg/hist_sig)[m_idx, pt_idx]
    return weights*np.sum(weights_bkg)/np.sum(weights)


def weights_factors(JZW, data_file):
    if np.all(JZW == -1) or np.all(JZW == 0):
        factors = len(list(h5py.File(data_file,"r").values())[0])/len(JZW)
    else:
        file_JZW = np.int_(h5py.File(data_file,"r")['JZW'])
        n_JZW    = [np.sum(file_JZW==n) for n in range(int(np.max(file_JZW))+1)]
        # topo samples JZWs
        #n_JZW = [ 35596, 13406964, 15909276, 17831457, 15981239, 15997303, 13913843, 13983297, 15946135, 15993849]
        # UFO samples JZWs
        #n_JZW = [181406, 27395322, 20376000, 13928294,  6964549,  6959061,  3143354,   796201,   796171,   796311]
        factors = np.ones_like(JZW, dtype=np.float32)
        for n in np.arange(len(n_JZW)):
            if np.sum(JZW==n) != 0: factors[JZW==n] = n_JZW[n]/np.sum(JZW==n)
    return factors


def jets_4v(sample):
    idx_tuples = get_idx(len(sample)); manager = mp.Manager(); return_dict = manager.dict()
    processes  = [mp.Process(target=get_4v, args=(sample, idx, return_dict)) for idx in idx_tuples]
    for task in processes: task.start()
    for task in processes: task.join()
    return {key:np.concatenate([return_dict[idx].pop(key) for idx in idx_tuples])
            for key in return_dict[idx_tuples[0]]}
def get_4v(sample, idx=(0,None), return_dict=None):
    sample = np.float32(sample[idx[0]:idx[1]])
    sample = np.sum(np.reshape(sample, (-1,int(sample.shape[1]/4),4)), axis=1)
    E, px, py, pz = [sample[:,n] for n in np.arange(sample.shape[-1])]
    pt = np.sqrt(px**2 + py**2)
    m  = np.sqrt(np.maximum(0, E**2 - px**2 - py**2 - pz**2))
    output_dict = {'pt_calo':pt, 'm_calo':m}
    if idx == (0,None): return output_dict
    else: return_dict[idx]  =  output_dict


def JSD(P, Q, idx, n_dims, return_dict, reshape=False):
    if reshape:
        P, Q = np.reshape(P, (-1,int(P.shape[1]/n_dims),n_dims)), np.reshape(Q, (-1,int(Q.shape[1]/n_dims),n_dims))
        return_dict[idx] = [np.mean([distance.jensenshannon(P[n,:,m], Q[n,:,m]) for m in np.arange(n_dims)])
                            for n in np.arange(idx[0],idx[1])]
    else: return_dict[idx] = [distance.jensenshannon(P[n,:], Q[n,:]) for n in np.arange(idx[0],idx[1])]


def KSD(P, Q, idx, n_dims, return_dict, reshape=False):
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
        P = np.maximum(np.float64(P), delta)
        Q = np.maximum(np.float64(Q), delta)
    if metric in ['Inputs']:
        loss = np.mean(P, axis=1)
    if metric in ['JSD', 'KSD', 'EMD']:
        idx_tuples = get_idx(len(P), mp.cpu_count()//3)
        target     = JSD if metric=='JSD' else KSD if metric=='KSD' else EMD
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


def fit_scaler(sample, n_dims, scaler_out, reshape=False, scaler_type='RobustScaler'):
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
    print('Saving scaler to', scaler_out)
    pickle.dump(scaler, open(scaler_out, 'wb'))
    return scaler
def apply_scaling(sample, n_dims, scaler, reshape, idx=(0,None), return_dict=None):
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
def apply_scaler(sample, n_dims, scaler, reshape=False):
    print('Applying scaler', end=' ', flush=True); start_time = time.time()
    idx_tuples = get_idx(len(sample), mp.cpu_count())
    manager    = mp.Manager(); return_dict = manager.dict()
    arguments  = [(sample[idx[0]:idx[1]], n_dims, scaler, reshape, idx, return_dict) for idx in idx_tuples]
    processes  = [mp.Process(target=apply_scaling, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    sample = np.concatenate([return_dict[idx] for idx in idx_tuples])
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return sample
def inverse_scaler(sample, n_dims, scaler, reshape=False):
    print('Applying inverse scaler', end=' ', flush=True); start_time = time.time()
    shape = sample.shape
    if reshape: sample = np.reshape(sample, (-1,n_dims))
    sample = scaler.inverse_transform(sample)
    sample = np.reshape(sample, shape)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return sample


def best_threshold(y_true, X_loss, X_mass, weights, cut_type):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, X_loss, pos_label=0, sample_weight=weights)
    len_0      = np.sum(fpr==0)
    thresholds = thresholds[len_0:][tpr[len_0:]>0.01]
    if cut_type=='gain' :
        cut_values = (tpr[len_0:]/fpr[len_0:])         [tpr[len_0:]>0.01]
    if cut_type=='sigma':
        cut_values = (tpr[len_0:]/np.sqrt(fpr[len_0:]))[tpr[len_0:]>0.01]
    cut_index = np.argmax(cut_values)
    if cut_type=='sigma':
        n_sig = np.sum(weights[y_true==0])
        n_bkg = np.sum(weights[y_true==1])
        cut_values *= n_sig/np.sqrt(n_bkg)
    return thresholds[cut_index], cut_values[cut_index]


def apply_best_cut(y_true, X_true, X_pred, sample, n_dims, model, metric, cut_type='gain'):
    if metric == 'Latent': X_loss = latent_loss(X_true, model)
    else: X_loss = loss_function(X_true, X_pred, n_dims, metric, multiloss=False)
    loss_cut, loss_val = best_threshold(y_true, X_loss, sample['m'], sample['weights'], cut_type)
    print('Best', metric, 'cut on '+format(cut_type,'5s')+':', metric, '>=', end= ' ')
    print(format(loss_cut, '.3f')+' / best '+format(cut_type,'5s'), '=', format(loss_val, '>5.2f'))
    sample = {key:sample[key][X_loss>loss_cut] for key in sample}
    return sample


def bump_hunter(sample, output_dir=None, cut_type=None, m_range=[0,300], bins=50,
                make_histo=True, print_info=True, logspace=False):
    #import pyBumpHunter as BH; sys.path.append('../')
    #from BumpHunter.BumpHunter.bumphunter_1dim import BumpHunter1D
    from BumpHunter.bumphunter_1dim import BumpHunter1D
    y_true = np.where(sample['JZW']==-1, 0, 1)
    data, data_weights = sample['m']    , sample['weights']
    bkg , bkg_weights  = data[y_true==1], data_weights[y_true==1]
    if logspace:
        bins = np.logspace(np.log10(max(10,m_range[0])), np.log10(m_range[1]), num=bins)
    data_hist, bin_edges = np.histogram(data, bins=bins, range=m_range, weights=data_weights)
    bkg_hist , bin_edges = np.histogram(bkg , bins=bins, range=m_range, weights= bkg_weights)
    start_time = time.time()
    """ BumpHunter1D class instance """
    #hunter = BH.BumpHunter1D(rang=m_range, width_min=2, width_max=6, width_step=1, scan_step=1,
    #                         npe=1000, nworker=1, seed=0, bins=bin_edges)
    hunter = BumpHunter1D(rang=m_range, width_min=2, width_max=6, width_step=1, scan_step=1,
                          npe=1000, nworker=1, seed=0, bins=bin_edges)
    hunter.bump_scan(data_hist, bkg_hist, is_hist=True, verbose=make_histo and print_info)
    filename = None if output_dir==None else output_dir+'/'+'BH_'+cut_type+'.png'
    if make_histo: print('Saving bump hunting plot to:', filename)
    max_sig = hunter.plot_bump(data_hist, bkg_hist, is_hist=True, filename=filename, make_histo=make_histo)
    if make_histo and print_info:
        hunter.print_bump_info()
        hunter.print_bump_true(data_hist, bkg_hist, is_hist=True)
        print(format(time.time() - start_time, '2.1f'), '\b'+' s')
    else:
        return max_sig


def latent_loss(X_true, model):
    idx_tuples = get_idx(len(X_true), bin_size=1e6)
    X_latent = []
    for idx in idx_tuples:
        model(X_true[idx[0]:idx[1]])
        X_latent += [model.losses[0].numpy()]
    return np.concatenate(X_latent)


def get_idx(max_val, n_bins=10, bin_size=None, min_val=0, integer=True, tuples=True):
    if bin_size == None:
        n_bins   = max(1, min(max_val-min_val, n_bins))
        bin_size = (max_val-min_val)//n_bins
    idx_list = np.append(np.arange(min_val, max_val, bin_size), max_val)
    if integer: idx_list = np.int_(idx_list)
    if tuples: return list(zip(idx_list[:-1], idx_list[1:]))
    else     : return idx_list
