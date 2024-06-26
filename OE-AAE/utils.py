import numpy           as np
import tensorflow      as tf
import multiprocessing as mp
import os, sys, h5py, time, pickle, warnings, itertools
from   threading     import Thread
from   sklearn       import preprocessing, metrics, utils
from   functools     import partial
from   scipy.spatial import distance
from   scipy         import stats, interpolate, optimize
from   BumpHunter.bumphunter_1dim import BumpHunter1D


def get_file(data_type, host_name='atlas'):
    if 'atlas'  in host_name: data_path = '/opt/tmp/godin/AD_data'
    if 'beluga' in host_name: data_path = '/project/def-arguinj/shared/AD_data'
    data_files = {
                  'QCD-Geneva' :'formatted_converted_20210629_QCDjj_pT_450_1200_nevents_10M_dPhifixed_float32.h5'         ,
                  #'QCD-Geneva':'formatted_converted_20210629_QCDjj_pT_450_inf_nevents_15M_weighted_float32.h5'            ,
                  'top-Geneva' :'formatted_converted_20211213_ttbar_allhad_pT_450_1200_nevents_10M_dPhifixed_float32.h5'  ,
                  '2HDM_200GeV':'formatted_converted_delphes_H_HpHm_tb_generation_mh2_2000_mhc_200_nevents_10M_float32.h5',
                  '2HDM_500GeV':'formatted_converted_delphes_H_HpHm_tb_generation_mh2_5000_mhc_500_nevents_10M_float32.h5',
                  'VZ_200GeV'  :'formatted_converted_delphes_z_zprime_tt_allhad_MVz_2000_MT_200_nevents_1M_float32.h5'    ,
                  'VZ_500GeV'  :'formatted_converted_delphes_z_zprime_tt_allhad_MVz_5000_MT_500_nevents_8M_float32.h5'    ,
                  'QCD-Delphes':'Delphes_dijet.h5'   ,
                  'top-Delphes':'Delphes_ttbar.h5'   ,
                  'QCD-topo'   :'Atlas_topo-dijet.h5',
                  'top-topo'   :'Atlas_topo-ttbar.h5',
                  'QCD-UFO'    :'Atlas_UFO-dijet.h5' ,
                  'top-UFO'    :'Atlas_UFO-ttbar.h5' ,
                  'BSM'        :'Atlas_BSM.h5'       ,
                  'OoD-W'      :'resamples_oe_w.h5'  ,
                  'OoD-H'      :'formatted_converted_Outliers_delphes_H_HpHm_generationredo_float32.h5',
                  #'OoD-H'      :'phase_1_H_HpHm_tb_generation_merged_reformatted.h5'                   ,
                  #'OoD-H'      :'phase_1_H_tb_jj_merged.h5'
                  }
    return data_path + '/' + data_files[data_type]


def get_data(Autoencoder, Discriminator, bkg_data, sig_data, n_bkg, n_sig, cuts, n_const, n_dims,
             constituents, HLVs, HLV_list, const_scaler, HLV_scaler, normal_loss, deco, output_dir):
    def loss_mapping(x):
        if   np.all(np.logical_and(x >= 0, x <= 1)): return  x
        elif np.all(np.logical_and(x >=-1, x <= 0)): return  x + 1
        elif np.all(x >= 0)                        : return  x/(1 + x)
        #elif np.all(x >= 0)                        : return  x/np.max(x)
        elif np.all(x <= 0)                        : return  1/(1 - x)
        else                                       : return (x/(np.abs(x)+1) + 1)/2
    n_sig = min(n_sig, len(list(h5py.File(get_file(sig_data),'r').values())[0]))
    sample = make_sample(bkg_data, sig_data, n_bkg, n_sig, cuts, n_const, n_dims, constituents, HLVs, HLV_list)
    y_true = np.where(sample['JZW']==-1, 0, 1)
    factor = 20 if sig_data=='2HDM_200GeV' or sig_data=='top-Geneva' else 20
    sample['weights'][y_true==0] /= adjust_weights(sample, y_true, factor=factor) #Adjusting weights for signal
    if 'constituents' in sample:
        sample['constituents'] = apply_scaler(sample['constituents'], n_dims, const_scaler)
    if 'HLVs' in sample:
        sample['HLVs'        ] = apply_scaler(sample['HLVs'        ], n_dims,   HLV_scaler)
    if 'constituents' in sample and 'HLVs' in sample:
        X_true = np.hstack([sample['constituents'], sample['HLVs']])
    elif 'constituents' in sample:
        X_true = sample['constituents']
    elif 'HLVs' in sample:
        X_true = sample['HLVs']
    print('\n'+(sig_data+': inference').upper())
    print('Autoencoder'  ) ; X_auto = Autoencoder  .predict(X_true, batch_size=int(1e4), verbose=1)
    print('Discriminator') ; X_disc = Discriminator.predict(X_true, batch_size=int(1e4), verbose=1)
    X_loss = {'Autoencoder':make_discriminant(X_true, X_auto, metric='MAE'), 'Discriminator':X_disc[:,2]}
    #X_loss['AAE'] = Discriminator.predict(X_auto, batch_size=int(1e4), verbose=1)[:,2]
    X_loss['Auto+Disc'] = (X_loss['Autoencoder']+X_loss['Discriminator'])/2
    if normal_loss == 'ON' or deco in ['m','pt','2d']:
        X_loss = {layer:loss_mapping(X_loss[layer]) for layer in X_loss}
    if deco in ['m','pt','2d']:
        manager   = mp.Manager(); return_dict = manager.dict()
        arguments = [(y_true, sample, X_loss[layer], layer, deco, return_dict) for layer in X_loss]
        threads = [Thread(target=bin_deco, args=arg) for arg in arguments]
        for job in threads: job.start()
        for job in threads: job.join()
        X_loss = return_dict
    print()
    return {'sample':sample, 'y_true':y_true, 'X_true':X_true, 'X_loss':X_loss}


def get_bins(var, var_bins=None, max_bins=100, min_bin_count=2, logspace=True, deco=True, offset=0):
    if not deco: return [np.min(var), np.max(var)]
    if var_bins is None:
        #if logspace: var_bins = np.logspace(np.log10(np.min(var)), np.log10(np.max(var)), num=max_bins)
        #else       : var_bins = np.linspace(         np.min(var) ,          np.max(var) , num=max_bins)
        min_var, max_var = np.min(np.float64(var)), np.max(np.float64(var))
        if logspace: var_bins = np.logspace(np.log10(min_var), np.log10(max_var), num=max_bins)
        else       : var_bins = np.linspace(         min_var ,          max_var , num=max_bins)
        var_bins[0], var_bins[-1] = min_var, max_var+offset
    while True:
        try:
            var_idx = np.clip(np.digitize(var, var_bins), 1, len(var_bins)-1) - 1
        except:
            print(var_bins)
            sys.exit()
        for idx in range(1,len(var_bins)-1)[::-1]:
            if np.sum(var_idx==idx) < max(2, min_bin_count):
                var_bins = np.delete(var_bins, idx)
                break
        if idx == 1: return var_bins
def cum_distribution(x):
    values, counts = np.unique(x, return_counts=True)
    if 0 not in values: values, counts = np.r_[0, values], np.r_[0, counts]
    if 1 not in values: values, counts = np.r_[values, 1], np.r_[counts, 0]
    return interpolate.interp1d(values, np.cumsum(counts)/len(x), fill_value=(0,1), bounds_error=False)
def bin_deco(y_true, sample, X_loss, layer, deco='2d', return_dict=None, multithreading=True):
    if deco not in ['m','pt','2d']: return
    def get_pt_bins(mass, pt, deco, m_range):
        cuts = mass>=m_range[0], mass<=m_range[1] if np.max(mass)==m_range[1] else mass<m_range[1]
        return get_bins(pt[np.logical_and(cuts[0], cuts[1])], deco=False if deco=='m' else True)
    def get_loss(X_loss, bkg_loss, m_idx_bkg, pt_idx_bkg, m_idx, pt_idx, multithreading, m):
        for n in range(np.max(pt_idx[m])+1):
            cdf = cum_distribution(bkg_loss[np.logical_and(m_idx_bkg==m, pt_idx_bkg[m]==n)])
            indices = np.where(np.logical_and(m_idx==m, pt_idx[m]==n))[0]
            X_loss[indices] = cdf(X_loss[indices])
        if not multithreading: return X_loss
    def get_digits(pt_var, bins):
        return np.clip(np.digitize(pt_var, bins), 1, max(len(bins)-1,1)) - 1
    bkg_mass, bkg_pt, bkg_loss = sample['m'][y_true==1], sample['pt'][y_true==1], X_loss[y_true==1]
    m_bins    = get_bins(bkg_mass, deco=False if deco=='pt' else True)
    m_idx_bkg = get_digits(bkg_mass   , m_bins)
    m_idx     = get_digits(sample['m'], m_bins)
    if multithreading:
        with mp.pool.ThreadPool() as pool:
            pt_bins    = pool.map(partial(get_pt_bins, bkg_mass, bkg_pt, deco), zip(m_bins[:-1], m_bins[1:]))
            pt_idx_bkg = pool.map(partial(get_digits, bkg_pt      ), pt_bins)
            pt_idx     = pool.map(partial(get_digits, sample['pt']), pt_bins)
            func_args  = (X_loss, bkg_loss, m_idx_bkg, pt_idx_bkg, m_idx, pt_idx, multithreading)
            pool.map(partial(get_loss, *func_args), np.arange(len(pt_idx)))
    else:
        pt_bins    = [get_pt_bins(bkg_mass, bkg_pt, deco, m_range) for m_range in zip(m_bins[:-1], m_bins[1:])]
        pt_idx_bkg = [get_digits(bkg_pt      , bins) for bins in pt_bins]
        pt_idx     = [get_digits(sample['pt'], bins) for bins in pt_bins]
        for m in range(len(pt_idx)):
            X_loss = get_loss(X_loss, bkg_loss, m_idx_bkg, pt_idx_bkg, m_idx, pt_idx, multithreading, m)
    return_dict[layer] = X_loss
def mass_deco(y_true, X_mass, X_loss, m_bins=None):
    mass, loss = X_mass[y_true==1], X_loss[y_true==1]
    if m_bins is None: m_bins = get_bins(mass, deco=True)
    m_idx = np.clip(np.digitize(  mass, m_bins), 1, len(m_bins)-1) - 1
    cdf_list = [cum_distribution(loss[m_idx==idx]) for idx in range(len(m_bins)-1)]
    m_idx = np.clip(np.digitize(X_mass, m_bins), 1, len(m_bins)-1) - 1
    for idx in range(len(m_bins)-1): X_loss[m_idx==idx] = cdf_list[idx](X_loss[m_idx==idx])
    return X_loss


class Batch_Generator(tf.keras.utils.Sequence):
    def __init__(self, bkg_data, n_const, n_dims, n_bkg, OoD_sample=None,
                 weight_type='X-S', cuts='', multithread=True, constituents='ON', HLVs='ON', HLV_list=None,
                 bin_sizes=None, HLV_scaler=None, const_scaler=None, output_dir=None, memGB=30):
        self.bkg_data     = bkg_data
        self.n_const      = n_const      ; self.n_dims       = n_dims
        self.n_bkg        = n_bkg        ; self.OoD_sample   = OoD_sample
        self.weight_type  = weight_type  ; self.cuts         = cuts
        self.multithread  = multithread  ; self.bin_sizes    = bin_sizes
        self.HLV_scaler   = HLV_scaler   ; self.const_scaler = const_scaler
        self.HLVs         = HLVs         ; self.HLV_list     = HLV_list
        self.constituents = constituents ; self.output_dir   = output_dir
        self.load_size = min(np.diff(n_bkg)[0], int(1e9*memGB/n_const/n_dims/4))
    def __len__(self):
        """ Number of batches per epoch """
        return int(np.ceil(np.diff(self.n_bkg)[0]/self.load_size))
    def __getitem__(self, gen_idx):
        if self.output_dir is not None: print('\nLoading QCD training sample'.upper())
        else                          : print('\nLoading QCD validation sample'.upper())
        bkg_idx = gen_idx*self.load_size  , (gen_idx+1)*self.load_size
        bkg_idx = bkg_idx[0]+self.n_bkg[0], min(bkg_idx[1]+self.n_bkg[0], self.n_bkg[1])
        bkg_sample = load_data(self.bkg_data, bkg_idx, self.cuts, self.n_const, self.n_dims,
                               self.constituents, self.HLVs, self.HLV_list)
        OoD_sample = bkg_sample if self.OoD_sample is None else self.OoD_sample
        if OoD_sample is not None:
            #OoD_sample = OoD_sampling(bkg_sample, OoD_sample) ; print()
            OoD_sample = OoD_pairing(bkg_sample, OoD_sample, self.multithread); print()
        if self.bin_sizes is not None:
            bkg_sample, OoD_sample = reweight_sample(bkg_sample, OoD_sample, self.bin_sizes, self.weight_type)
        if self.output_dir is not None and gen_idx == 0:
            from plots import sample_distributions
            sample = {key:np.concatenate([bkg_sample[key], OoD_sample[key]]) for key in ['m','pt','weights','JZW']}
            sample_distributions(sample, 'OoD', self.output_dir, 'train', self.bin_sizes, self.weight_type)
        if 'constituents' in bkg_sample:
            bkg_sample['constituents'] = apply_scaler(bkg_sample['constituents'], self.n_dims, self.const_scaler, 'QCD')
        if 'HLVs' in bkg_sample:
            bkg_sample['HLVs'        ] = apply_scaler(bkg_sample['HLVs'        ], self.n_dims, self.HLV_scaler  , 'QCD')
        return {'bkg':bkg_sample, 'OoD':OoD_sample}


def load_data(data_type, idx, cuts, n_const, n_dims, constituents, HLVs, HLV_list,
              var_list=None, DSIDs=None, adjust_weights=False, verbose=True, pt_scaling=False):
    start_time = time.time()
    if np.isscalar(idx): idx = (0, idx)
    data_file = get_file(data_type)
    data      = h5py.File(data_file,"r")
    if verbose: print('Loading', data_file.split('/')[-1], end='', flush=True)
    if var_list == None:
        #var_list = ['m_calo','pt_calo','rljet_m_comb','rljet_pt_comb','weights','constituents','JZW','DSID']
        #var_list = ['weights','constituents','JZW','DSID']
        var_list = data.keys()
    sample = {key:data[key][idx[0]:idx[1]] for key in set(data) & set(var_list) if 'constituents' not in key}
    if len(set(sample) & {'m_calo','pt_calo','rljet_m_comb','rljet_pt_comb'}) == 0: var_list += ['constituents']
    if constituents == 'ON':
        # Enforcing jets pt sorting
        sample['constituents'] = jets_sorting(data['constituents'][idx[0]:idx[1],:])[:,:4*n_const]
        # Assuming jets pt sorting
        #sample['constituents'] = np.float32(data['constituents'][idx[0]:idx[1],:4*n_const])
        if 4*n_const > data['constituents'].shape[1]:
            shape = sample['constituents'].shape
            zeros_array = np.zeros((shape[0], 4*n_const-shape[1]), dtype=np.float32)
            sample['constituents'] = np.hstack([sample['constituents'], zeros_array])
        if len(set(sample) & {'m_calo','pt_calo','rljet_m_comb','rljet_pt_comb'}) == 0:
            sample.update({key:val for key,val in jets_4v(sample['constituents']).items()})
    sample['pt'] = sample.pop('rljet_pt_comb' if 'rljet_pt_comb' in sample else 'pt_calo')
    sample['m' ] = sample.pop('rljet_m_comb'  if 'rljet_m_comb'  in sample else  'm_calo')
    sample_size = len(list(sample.values())[0])
    if 'JZW'     not in sample:
        sample['JZW'    ] = np.full(sample_size, 0 if 'QCD' in data_type.upper() else -1, dtype=np.float32)
    if 'weights' not in sample:
        sample['weights'] = np.full(sample_size, 1                                      , dtype=np.float32)
    """ Applying sample cuts """
    sample = sample_cuts(sample, cuts, DSIDs)
    """ Adjusting weights for cross-section """
    if adjust_weights:
        sample['weights'] = sample['weights']*weights_factors(sample['JZW'], data_file)
    if pt_scaling:
        """ Scaling (E, px, py, px) with jet pt """
        sample['constituents'] = sample['constituents']/np.float32(sample['pt'][:,np.newaxis])
    if 'constituents' in sample and n_dims == 3:
        """ Using (px, py, px) instead of (E, px, py, px) """
        shape = sample['constituents'].shape
        sample['constituents'] = np.reshape(sample['constituents']        , (-1,shape[1]//4,4))
        sample['constituents'] = np.reshape(sample['constituents'][...,1:], (shape[0],-1)     )
    if verbose: print(' (', '\b'+format(time.time()-start_time, '2.1f'), '\b'+' s)')
    if HLVs == 'ON':
        if 'tau21' in HLV_list:
            denominator = np.maximum(sample['rljet_Tau1_wta'], 1e-16)
            sample['tau21'] = sample['rljet_Tau2_wta']/denominator
        if 'tau32' in HLV_list:
            denominator = np.maximum(sample['rljet_Tau2_wta'], 1e-16)
            sample['tau32'] = sample['rljet_Tau3_wta']/denominator
        sample['HLVs' ] = np.hstack([np.float32(sample[key])[:,np.newaxis] for key in HLV_list])
    #for key,val in sample.items(): print(key, val.shape, val.dtype)
    return sample


def make_sample(bkg_data, sig_data, bkg_idx=1, sig_idx=1, cuts='', n_const=20, n_dims=4, constituents='ON', HLVs='ON',
                HLV_list=None, var_list=None, DSIDs=None, adjust_weights=False, shuffling=False, verbose=True):
    print((sig_data+': loading data').upper())
    sig_sample = load_data(sig_data, sig_idx, cuts, n_const, n_dims, constituents, HLVs,
                           HLV_list, var_list, DSIDs, adjust_weights, verbose)
    bkg_sample = load_data(bkg_data, bkg_idx, cuts, n_const, n_dims, constituents, HLVs,
                           HLV_list, var_list, DSIDs, adjust_weights, verbose)
    if 'OoD' in sig_data: sig_sample = OoD_sampling(sig_sample, len(list(bkg_sample.values())[0]))
    sample = {key:np.concatenate([bkg_sample[key], sig_sample[key]]) for key in set(bkg_sample)&set(sig_sample)}
    if shuffling: sample = {key:utils.shuffle(sample[key], random_state=0) for key in sample}
    return sample


def split_sample(sample):
    JZW        = sample['JZW']
    bkg_sample = {key:val[JZW!=-1] for key,val in sample.items()}
    sig_sample = {key:val[JZW==-1] for key,val in sample.items()}
    return bkg_sample, sig_sample


def make_datasets(sample, sample_OE, batch_size=1):
    X    = (sample   ['constituents'], sample   ['weights'])
    X_OE = (sample_OE['constituents'], sample_OE['weights'])
    dataset = tf.data.Dataset.from_tensor_slices( X + X_OE )
    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def merge_losses(losses, history, model_in, output_dir):
    print('\nSaving training losses to:', history)
    if os.path.isfile(history) and model_in != output_dir+'/':
        old_losses = pickle.load(open(history, 'rb'))
        losses = {key:np.append(old_losses[key], losses[key]) for key in losses}
    pickle.dump(losses, open(history,'wb'))
    return losses


def sample_cuts(sample, cuts, DSIDs=None):
    sample_size = len(list(sample.values())[0])
    cut_list    = [np.full(sample_size, True, dtype=bool)]
    for cut in cuts:
        try   : cut_list.append(eval(cut))
        except: print('WARNING: invalid cut:' , cut)
    cuts = np.logical_and.reduce(cut_list)
    if DSIDs is not None:
        if np.isscalar(DSIDs): DSIDs = [DSIDs]
        dsid_cuts = np.logical_or.reduce([sample['DSID']==int(n) for n in DSIDs])
        cuts = np.logical_and(cuts, dsid_cuts)
    if not np.all(cuts):
        sample = {key:sample[key][cuts] for key in sample}
    return sample


def OoD_sampling(bkg_sample, OoD_sample, adjust_weights=False, seed=None):
    np.random.seed(seed)
    source_size = len(list(OoD_sample.values())[0])
    target_size  = len(list(bkg_sample.values())[0])
    indices = np.random.choice(source_size, target_size, replace=source_size<target_size)
    if adjust_weights: OoD_sample['weights'] = OoD_sample['weights']*np.float32(source_size/target_size)
    return {key:np.take(OoD_sample[key], indices, axis=0) for key in OoD_sample}


def OoD_pairing(bkg_sample, OoD_sample, multithread=True, verbose=True, seed=0):
    np.random.seed(seed)
    m_idx = np.argsort(OoD_sample['m'])
    OoD_sample = {key:np.take(OoD_sample[key], m_idx, axis=0) for key in OoD_sample}
    m_OoD, pt_OoD = OoD_sample['m'], OoD_sample['pt']
    m_bkg, pt_bkg = bkg_sample['m'], bkg_sample['pt']
    def get_cuts(m_OoD, pt_OoD, m_val, pt_val, m_width, pt_width, pairing='2d'):
        idx  = np.searchsorted(m_OoD, m_val-m_width/2), np.searchsorted(m_OoD, m_val+m_width/2)
        cuts = [pt_OoD[idx[0]:idx[1]] >= pt_val-pt_width/2, pt_OoD[idx[0]:idx[1]] <= pt_val+pt_width/2]
        return idx[0], np.full(np.diff(idx),True) if pairing=='m' else np.logical_and.reduce(cuts)
    def get_indice(m_OoD, pt_OoD, m_val, pt_val, m_width=10, pt_width=10, factor=2):
        while True:
            idx, cuts = get_cuts(m_OoD, pt_OoD, m_val, pt_val, m_width, pt_width)
            if np.sum(cuts) == 0: pt_width *= factor
            else                : return np.random.choice(np.where(cuts)[0]) + idx
            idx, cuts = get_cuts(m_OoD, pt_OoD, m_val, pt_val, m_width, pt_width)
            if np.sum(cuts) == 0:  m_width *= factor
            else                : return np.random.choice(np.where(cuts)[0]) + idx
    def get_indices(m_bkg, pt_bkg, m_OoD, pt_OoD, idx, return_dict=None):
        indices = [get_indice(m_OoD, pt_OoD, m_bkg[n], pt_bkg[n]) for n in range(idx[0], idx[1])]
        if return_dict is None: return indices
        else:       return_dict[idx] = indices
    if verbose: print('Pairing OoD with QCD', end=' ', flush=True); start_time = time.time()
    if multithread:
        manager = mp.Manager(); return_dict = manager.dict()
        idx_tuples = get_idx(len(m_bkg), min(mp.cpu_count(),16))
        arguments = [(m_bkg, pt_bkg, m_OoD, pt_OoD, idx, return_dict) for idx in idx_tuples]
        processes = [mp.Process(target=get_indices, args=arg) for arg in arguments]
        for task in processes: task.start()
        for task in processes: task.join()
        indices = np.concatenate([return_dict[key] for key in idx_tuples])
    else:
        indices = get_indices(m_bkg, pt_bkg, m_OoD, pt_OoD, (0,len(m_bkg)))
    if verbose: print('(', '\b'+format(time.time()-start_time, '2.1f'), '\b'+' s)')
    return {key:np.take(OoD_sample[key], indices, axis=0) for key in OoD_sample}


def reweight_sample(bkg_sample, sig_sample, bin_sizes, weight_type='X-S'):
    """ None   : no weights (weight=1)                    """
    """ X-S    : default cross-section weights            """
    """ flat_m : 1D flat weighting for OoD and background """
    """ flat_pt: 1D flat weighting for OoD and background """
    """ flat_2d: 2D flat weighting for OoD and background """
    """ OoD_m  : 1D weighting of OoD to background        """
    """ OoD_pt : 1D weighting of OoD to background        """
    """ OoD_2d : 2D weighting of OoD to background        """
    if weight_type == None or weight_type.lower() == 'none':
        sig_sample['weights'] = np.ones_like(sig_sample['weights'])
        bkg_sample['weights'] = np.ones_like(bkg_sample['weights'])
    if 'flat' in weight_type:
        sig_sample['weights'] = get_weights(bkg_sample, sig_sample, bin_sizes, weight_type)
        bkg_sample['weights'] = get_weights(bkg_sample, bkg_sample, bin_sizes, weight_type)
        sig_sample['weights'] = get_weights(bkg_sample, sig_sample, bin_sizes, weight_type='2d')
    if 'OoD' in weight_type:
        sig_sample['weights'] = get_weights(bkg_sample, sig_sample, bin_sizes, weight_type)
    if weight_type == 'X-S':
        sig_sample['weights'] *= np.sum(bkg_sample['weights'])/np.sum(sig_sample['weights'])
    return bkg_sample, sig_sample


def get_weights(bkg_sample, sig_sample, bin_sizes, weight_type, max_val=1e4, density=True):
    m_size, pt_size = bin_sizes['m'], bin_sizes['pt']
    m_bkg, pt_bkg, weights_bkg = [bkg_sample[key] for key in ['m','pt','weights']]
    m_sig, pt_sig, weights_sig = [sig_sample[key] for key in ['m','pt','weights']]
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
    return np.minimum( max_val, weights*np.sum(weights_bkg)/np.sum(weights) )


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


def adjust_weights(sample, y_true, bin_size=5, m_range=None, factor=10**0.5):
    m_sig      , m_bkg       = sample['m'      ][y_true==0], sample['m'      ][y_true==1]
    weights_sig, weights_bkg = sample['weights'][y_true==0], sample['weights'][y_true==1]
    m_bins = get_idx(np.max(m_sig), bin_size=bin_size, integer=False, tuples=False)
    histo_sig = np.histogram(m_sig, m_bins, m_range, weights=weights_sig)[0]
    histo_bkg = np.histogram(m_bkg, m_bins, m_range, weights=weights_bkg)[0]
    m_idx = np.argmax(histo_sig)
    return factor*histo_sig[m_idx]/histo_bkg[m_idx]


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
        return_dict[idx] = [np.mean([distance.jensenshannon(P[n,:,m], Q[n,:,m], base=2) for m in np.arange(n_dims)])
                            for n in np.arange(idx[0],idx[1])]
    else: return_dict[idx] = [distance.jensenshannon(P[n,:], Q[n,:], base=2) for n in np.arange(idx[0],idx[1])]


#def KSD(P, Q, idx, n_dims, return_dict, reshape=False):
#    if reshape:
#        P, Q = np.reshape(P, (-1,int(P.shape[1]/n_dims),n_dims)), np.reshape(Q, (-1,int(Q.shape[1]/n_dims),n_dims))
#        return_dict[idx] = [np.mean([stats.ks_2samp(P[n,:,m], Q[n,:,m])[0] for m in np.arange(n_dims)])
#                            for n in np.arange(idx[0],idx[1])]
#    else: return_dict[idx] = [stats.ks_2samp(P[n,:], Q[n,:])[0] for n in np.arange(idx[0],idx[1])]
def KSD(P, Q, idx, return_dict):
    return_dict[idx] = [stats.ks_2samp(P[n,:], Q[n,:])[0] for n in np.arange(idx[0],idx[1])]


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


def make_discriminant(P, Q, metric, layer=None, X_losses=None, delta=1e-32):
    def KLD(P, Q, base=2):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.nan_to_num(P*np.log(P/Q)/np.log(base))
    if layer == 'DISCRIMINATOR':
        return Q[:,0]
    if metric in ['JSD', 'KLD', 'X-S', 'MARE']:
        P = np.maximum(np.float64(P), delta)
        Q = np.maximum(np.float64(Q), delta)
    if metric in ['Inputs', 'Inputs_scaled']:
        loss = np.mean(P, axis=1)
    if metric == 'MSE' : loss = np.mean(      (P - Q)**2 , axis=1)
    if metric == 'MAE' : loss = np.mean(np.abs(P - Q)    , axis=1)
    if metric == 'MARE': loss = np.mean(np.abs(P - Q)/P  , axis=1)
    if metric in ['JSD', 'KSD','KLD', 'X-S']:
        P, Q = P/np.sum(P,axis=1)[:,np.newaxis], Q/np.sum(Q,axis=1)[:,np.newaxis]
    if metric in ['KSD', 'EMD']:
        idx_tuples = get_idx(len(P), mp.cpu_count()//3)
        target     = JSD if metric=='JSD' else KSD if metric=='KSD' else EMD
        manager    = mp.Manager(); return_dict = manager.dict()
        #processes  = [mp.Process(target=target, args=(P, Q, idx, n_dims, return_dict)) for idx in idx_tuples]
        processes  = [mp.Process(target=target, args=(P, Q, idx, return_dict)) for idx in idx_tuples]
        for task in processes: task.start()
        for task in processes: task.join()
        loss = np.concatenate([return_dict[idx] for idx in idx_tuples])
    if metric == 'KLD' :
        loss = np.sum(KLD(P,Q), axis=1)
    if metric == 'JSD':
        M = (P + Q) / 2
        loss = np.sqrt( np.sum((KLD(P,M)+KLD(Q,M))/2, axis=1) )
    if metric == 'X-S' :
        loss = np.sum(KLD(P,P*Q), axis=1)
    if X_losses is None: return loss
    else               : X_losses[metric] = loss


def latent_loss(X_true, model):
    idx_tuples = get_idx(len(X_true), bin_size=1e5)
    X_latent = []
    for idx in idx_tuples:
        model(X_true[idx[0]:idx[1]])
        X_latent += [model.losses[0].numpy()]
    X_latent = np.concatenate(X_latent)
    X_latent = np.where(np.isfinite(X_latent), X_latent, 0)
    return X_latent


def fit_scaler(sample, n_dims, scaler_out, scaler_type='RobustScaler', reshape=False):
    print('Fitting', scaler_type, 'to QCD sample', end='', flush=True)
    start_time = time.time()
    if reshape: sample = np.reshape(sample, (-1,n_dims))
    if scaler_type == 'QuantileTransformer':
        scaler = preprocessing.QuantileTransformer(output_distribution='normal', n_quantiles=10000, random_state=0)
    if scaler_type == 'PowerTransformer':
        scaler = preprocessing.PowerTransformer()
    if scaler_type == 'RobustScaler':
        scaler = preprocessing.RobustScaler()
    if scaler_type == 'MaxAbsScaler':
        scaler = preprocessing.MaxAbsScaler()
    scaler.fit(sample)
    print(' (', '\b'+format(time.time()-start_time, '2.1f'), '\b'+' s)')
    print('Saving to ' + scaler_out)
    pickle.dump(scaler, open(scaler_out, 'wb'))
    return scaler
def apply_scaling(sample, n_dims, scaler, tag, reshape=False, verbose=True, idx=(0,None), return_dict=None):
    if idx == (0,None) and verbose: print('Applying scaler/transformer to '+tag, end='', flush=True)
    start_time = time.time()
    shape = sample.shape
    if reshape: sample = np.reshape(sample, (-1,n_dims))
    sample = scaler.transform(sample)
    sample = np.reshape(sample, shape)
    if idx == (0,None):
        if verbose: print(' (', '\b'+format(time.time()-start_time, '2.1f'), '\b'+' s)')
        return sample
    else: return_dict[idx] = sample
def apply_scaler(sample, n_dims, scaler, tag='sample', reshape=False, verbose=True):
    if scaler is None: return sample
    if verbose: print('Applying scaler/transformer to '+tag, end='', flush=True)
    start_time = time.time()
    idx_tuples = get_idx(len(sample), int(mp.cpu_count()/2))
    manager = mp.Manager(); return_dict = manager.dict()
    arguments = [(sample[idx[0]:idx[1]], n_dims, scaler, tag, reshape, False, idx, return_dict)
                 for idx in idx_tuples]
    processes = [mp.Process(target=apply_scaling, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    sample = np.concatenate([return_dict[idx] for idx in idx_tuples])
    if verbose: print(' (', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return sample
def inverse_scaler(sample, n_dims, scaler, reshape=False):
    print('Applying inverse scaler', end=' ', flush=True); start_time = time.time()
    shape = sample.shape
    if reshape: sample = np.reshape(sample, (-1,n_dims))
    sample = scaler.inverse_transform(sample)
    sample = np.reshape(sample, shape)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return sample


def bump_hunter(sample, filename=None, sig_label=None, max_sigma=np.nan,
                m_range=[0,800], bin_size=5, print_info=False, logspace=False):
    verbose = filename is not None and print_info
    y_true = np.where(sample['JZW']==-1, 0, 1)
    data, data_weights = sample['m']    , sample['weights']
    bkg , bkg_weights  = data[y_true==1], data_weights[y_true==1]
    m_min = max(m_range[0], np.min(bkg))
    m_max = min(m_range[1], np.max(bkg))
    if logspace: bins = np.logspace(np.log10(max(1,m_min)), np.log10(m_max), num=100)
    else:        bins = get_idx(m_max, bin_size=bin_size, min_val=m_min, integer=False, tuples=False)
    bins = get_bins(bkg, bins, min_bin_count=20)
    data_hist, bin_edges = np.histogram(data, bins=bins, range=m_range, weights=data_weights)
    bkg_hist , bin_edges = np.histogram(bkg , bins=bins, range=m_range, weights= bkg_weights)
    """ BumpHunter1D class instance """
    #import pyBumpHunter as BH; sys.path.append('../')
    #from BumpHunter.BumpHunter.bumphunter_1dim import BumpHunter1D
    #hunter = BH.BumpHunter1D(rang=m_range, width_min=1, width_max=10, width_step=1, scan_step=1,
    #                         npe=100, nworker=1, seed=None, bins=bin_edges)
    hunter = BumpHunter1D(rang=m_range, width_min=1, width_max=10, width_step=1, scan_step=1,
                          npe=100, nworker=1, seed=None, bins=bin_edges)
    hunter.bump_scan(data_hist, bkg_hist, is_hist=True, verbose=verbose)
    bin_sigma, bump_range = hunter.plot_bump(data_hist, bkg_hist, is_hist=True)
    loc_sigma = hunter.bump_info(data_hist, verbose=verbose)
    try:
        gaussian_par = fit_gaussian(bins, bin_sigma, bump_range)
    except:
        try: gaussian_par = fit_gaussian(bins, bin_sigma)
        except Exception as error: print(error)
    if max_sigma is np.nan and gaussian_par is not np.nan:
        max_sigma = max(np.max(bin_sigma), gaussian_par[0]*gaussian_par[3])
    elif max_sigma is np.nan:
        max_sigma = np.max(bin_sigma)
    if filename is not None:
        from plots import plot_bump
        plot_bump(data, data_weights, y_true, bins, bin_sigma, loc_sigma, max_sigma,
                  bump_range, m_range, gaussian_par, sig_label, filename)
    return loc_sigma, max_sigma
def Gaussian(x, A, B, C):
    return A*np.exp(-(x-B)**2/(2*C**2))
def fit_gaussian(bins, bin_sigma, bump_range=None):
    x_val, y_val = (bins[:-1]+bins[1:])/2, bin_sigma
    if bump_range is None:
        selections = x_val != 0
    else:
        try   : selections = np.logical_and(x_val>=bump_range[0], x_val<=bump_range[1])
        except: selections = np.full_like(x_val, True, dtype=bool)
    x_val, y_val = x_val[selections], y_val[selections]
    A_approx, B_approx, C_approx = np.max(y_val), x_val[np.argmax(y_val)], np.sqrt(np.var(x_val))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x_val, y_val = (x_val - B_approx)/C_approx, y_val/A_approx
        try   : height, mean, std = optimize.curve_fit(Gaussian, x_val, y_val)[0]
        except: return np.nan
    return A_approx, B_approx, C_approx, height, mean, std


def filtering(y_true, X_true, X_pred, sample):
    bad_idx = list(set(np.where(~np.isfinite(X_pred))[0]))
    y_true  = np.delete(y_true, bad_idx, axis=0)
    X_true  = np.delete(X_true, bad_idx, axis=0)
    X_pred  = np.delete(X_pred, bad_idx, axis=0)
    for key in sample: sample[key] = np.delete(sample[key], bad_idx, axis=0)
    return y_true, X_true, X_pred, sample


def get_idx(max_val, n_bins=10, bin_size=None, min_val=0, integer=True, tuples=True):
    if bin_size == None:
        n_bins   = max(1, min(max_val-min_val, n_bins))
        bin_size = (max_val-min_val)//n_bins
    idx_list = np.append(np.arange(min_val, max_val, bin_size), max_val)
    if integer: idx_list = np.int_(idx_list)
    if tuples: return list(zip(idx_list[:-1], idx_list[1:]))
    else     : return idx_list


def jets_pt(jets):
    def get_mean(jets, idx, return_dict):
        jets = np.cumsum(np.reshape(np.float32(jets), (-1,int(jets.shape[1]/4),4)), axis=1)
        return_dict[idx] = np.sqrt(jets[:,:,1]**2 + jets[:,:,2]**2)
    idx_tuples = get_idx(len(jets), int(mp.cpu_count()/2))
    manager = mp.Manager(); return_dict = manager.dict()
    arguments = [(jets[idx[0]:idx[1]], idx, return_dict) for idx in idx_tuples]
    processes = [mp.Process(target=get_mean, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    return np.concatenate([return_dict[key] for key in idx_tuples], axis=0)


def jets_sorting(jets, idx=(0,None), return_dict=None):
    jets   = np.reshape(np.float32(jets), (-1,int(jets.shape[1]/4),4))
    pt     = np.sqrt(jets[:,:,1]**2 + jets[:,:,2]**2)
    pt_idx = np.argsort(pt, axis=-1)[:,::-1,np.newaxis]
    pt_idx = np.concatenate(4*[pt_idx], axis=2)
    jets   = np.take_along_axis(jets, pt_idx, axis=1)
    if idx == (0,None): return np.reshape(jets, (-1,np.prod(jets.shape[1:])))
    else: return_dict[idx]  =  np.reshape(jets, (-1,np.prod(jets.shape[1:])))
def jets_sorting_mp(jets):
    idx_tuples = get_idx(len(jets), int(mp.cpu_count()/2))
    manager = mp.Manager(); return_dict = manager.dict()
    arguments = [(jets[idx[0]:idx[1]], idx, return_dict) for idx in idx_tuples]
    processes = [mp.Process(target=jets_sorting, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    return np.concatenate([return_dict[key] for key in idx_tuples], axis=0)


def n_constituents(jets):
    def get_number(jets, idx, return_dict):
        jets = np.sum(np.reshape(np.abs(np.float32(jets)), (-1,int(jets.shape[1]/4),4)), axis=2)
        jets = np.hstack([jets, np.zeros((jets.shape[0],1))])
        jets = np.sort(jets, axis=-1)[:,::-1]
        return_dict[idx] = np.argmin(jets, axis=1)
    idx_tuples = get_idx(len(jets), int(mp.cpu_count()/2))
    manager = mp.Manager(); return_dict = manager.dict()
    arguments = [(jets[idx[0]:idx[1]], idx, return_dict) for idx in idx_tuples]
    processes = [mp.Process(target=get_number, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    return np.concatenate([return_dict[key] for key in idx_tuples])


def grid_search(**kwargs):
    if len(kwargs.items()) <= 1: array_tuple = list(kwargs.values())[0]
    else                       : array_tuple = list(itertools.product(*kwargs.values()))
    return dict( zip(np.arange(len(array_tuple)), array_tuple) )
