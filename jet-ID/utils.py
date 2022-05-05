import tensorflow        as tf
import numpy             as np
import multiprocessing   as mp
import matplotlib.pyplot as plt
import os, sys, h5py, pickle, time
from   sklearn  import metrics, utils, preprocessing
from   tabulate import tabulate
from   skimage  import transform
from   plots import valid_accuracy, plot_history, plot_distributions_DG, plot_ROC_curves, var_histogram
from   plots import plot_weights


def sample_histograms(valid_sample, valid_labels, train_sample, train_labels, weights, bins, output_dir):
    arguments = [(valid_sample, valid_labels, None, bins, output_dir, 'valid')]
    if np.any(train_labels) != None:
        arguments += [(train_sample, train_labels, weights, bins, output_dir, 'train')]
    processes = [mp.Process(target=var_histogram, args=arg+(var,)) for arg in arguments for var in ['pt','eta']]
    for job in processes: job.start()
    for job in processes: job.join()


def split_samples(valid_sample, valid_labels, train_sample, train_labels):
    #generate a different validation sample from training sample with downsampling
    valid_sample, valid_labels, extra_sample, extra_labels = downsampling(valid_sample, valid_labels)
    train_sample  = {key:np.concatenate([train_sample[key], extra_sample[key]]) for key in train_sample}
    train_labels  = np.concatenate([train_labels, extra_labels])
    sample_weight = match_distributions(train_sample, train_labels, valid_sample, valid_labels)
    return valid_sample, valid_labels, train_sample, train_labels, sample_weight


def get_class_weight(labels, bkg_ratio):
    n_e = len(labels); n_classes = max(labels) + 1
    if bkg_ratio == 0 and n_classes == 2: return None
    if bkg_ratio == 0 and n_classes != 2: bkg_ratio = 1
    ratios       = {**{0:1}, **{n:bkg_ratio for n in np.arange(1, n_classes)}}
    class_weight = {n:n_e/np.sum(labels==n)*ratios[n]/sum(ratios.values()) for n in np.arange(n_classes)}
    return class_weight


def get_sample_weights(sample, labels, weight_type=None, bkg_ratio=None, hist='2d', ref_class=0, density=False):
    if weight_type not in ['bkg_ratio', 'flattening', 'match2class', 'match2max']: return None, None
    pt = sample['pt']; eta = abs(sample['eta']); n_classes = max(labels)+1
    n_bins   = 100; base = (np.max(pt)/np.min(pt))**(1/n_bins)
    pt_bins  = [np.min(pt)*base**n for n in np.arange(n_bins+1)]
    pt_bins[-1]  = max( pt_bins[-1], max( pt)) + 1e-3
    n_bins   = 50; step = np.max(eta)/n_bins
    eta_bins = np.arange(np.min(eta), np.max(eta)+step, step)
    eta_bins[-1] = max(eta_bins[-1], max(eta)) + 1e-3
    if hist == 'pt' : eta_bins = [eta_bins[0], eta_bins[-1]]
    if hist == 'eta':  pt_bins = [ pt_bins[0],  pt_bins[-1]]
    pt_ind   = np.digitize( pt,  pt_bins, right=False) -1
    eta_ind  = np.digitize(eta, eta_bins, right=False) -1
    hist_ref = np.histogram2d(pt[labels==ref_class], eta[labels==ref_class],
                              bins=[pt_bins,eta_bins], density=density)[0]
    if density: hist_ref *= np.sum(labels==ref_class)
    hist_ref = np.maximum(hist_ref, np.min(hist_ref[hist_ref!=0]))
    total_ref_array = []; total_bkg_array = []; hist_bkg_array = []
    if np.isscalar(bkg_ratio): bkg_ratio = n_classes*[bkg_ratio] #bkg_ratio = [0.25, 1, 1, 1, 1, 1]
    for n in [n for n in np.arange(n_classes) if n != ref_class]:
        hist_bkg = np.histogram2d(pt[labels==n], eta[labels==n], bins=[pt_bins,eta_bins], density=density)[0]
        if density: hist_bkg *= np.sum(labels==n)
        hist_bkg = np.maximum(hist_bkg, np.min(hist_bkg[hist_bkg!=0]))
        ratio    = np.sum(hist_bkg)/np.sum(hist_ref) if bkg_ratio == None else bkg_ratio[n]
        if   weight_type == 'bkg_ratio':
            total_ref = hist_ref * max(1, np.sum(hist_bkg)/np.sum(hist_ref)/ratio)
            total_bkg = hist_bkg * max(1, np.sum(hist_ref)/np.sum(hist_bkg)*ratio)
        elif weight_type == 'flattening':
            total_ref = np.ones(hist_ref.shape) * max(np.max(hist_ref), np.max(hist_bkg)/ratio)
            total_bkg = np.ones(hist_bkg.shape) * max(np.max(hist_bkg), np.max(hist_ref)*ratio)
        elif weight_type == 'match2class':
            total_ref = hist_ref * max(1, np.max(hist_bkg/hist_ref)/ratio)
            total_bkg = hist_ref * max(1, np.max(hist_bkg/hist_ref)/ratio) * ratio
        elif weight_type == 'match2max':
            total_ref = np.maximum(hist_ref, hist_bkg/ratio)
            total_bkg = np.maximum(hist_bkg, hist_ref*ratio)
        total_ref_array.append(total_ref[np.newaxis,...])
        total_bkg_array.append(total_bkg[np.newaxis,...])
        hist_bkg_array.append ( hist_bkg[np.newaxis,...])
    hist_ref_array  = hist_ref[np.newaxis,...]
    hist_bkg_array  = np.concatenate( hist_bkg_array, axis=0)
    total_ref_array = np.concatenate(total_ref_array, axis=0)
    total_bkg_array = np.concatenate(total_bkg_array, axis=0)
    total_ref_ratio = total_ref_array / np.max(total_ref_array, axis=0)
    total_ref_array = np.max(total_ref_array, axis=0)
    total_bkg_array = total_bkg_array / total_ref_ratio
    weights_array = np.concatenate([total_ref_array/hist_ref_array, total_bkg_array/hist_bkg_array])
    sample_weight = np.zeros(len(labels), dtype=np.float32)
    class_list    = [ref_class] + [n for n in np.arange(n_classes) if n != ref_class]
    for n in np.arange(n_classes):
        sample_weight = np.where(labels==class_list[n], weights_array[n,...][pt_ind, eta_ind], sample_weight)
    return sample_weight*len(labels)/np.sum(sample_weight), {'pt':pt_bins, 'eta':eta_bins}


def gen_weights(n_train, weight_idx, sample_weight):
    weights = np.zeros(np.diff(n_train)[0])
    np.put(weights, weight_idx, sample_weight)
    return weights


def upsampling(sample, labels, bins, indices, hist_sig, hist_bkg, total_sig, total_bkg):
    new_sig = np.int_(np.around(total_sig)) - hist_sig
    new_bkg = np.int_(np.around(total_bkg)) - hist_bkg
    ind_sig = [np.where((indices==n) & (labels==0))[0] for n in np.arange(len(bins)-1)]
    ind_bkg = [np.where((indices==n) & (labels!=0))[0] for n in np.arange(len(bins)-1)]
    np.random.seed(0)
    ind_sig = [np.append(ind_sig[n], np.random.choice(ind_sig[n], new_sig[n],
               replace = len(ind_sig[n])<new_sig[n])) for n in np.arange(len(bins)-1)]
    ind_bkg = [np.append(ind_bkg[n], np.random.choice(ind_bkg[n], new_bkg[n],
               replace = len(ind_bkg[n])<new_bkg[n])) for n in np.arange(len(bins)-1)]
    indices = np.concatenate(ind_sig + ind_bkg); np.random.shuffle(indices)
    return {key:np.take(sample[key], indices, axis=0) for key in sample}, np.take(labels, indices)


def downsampling(sample, labels, bkg_ratio=None):
    pt = sample['p_et_calo']; bins = [0, 10, 20, 30, 40, 60, 80, 100, 130, 180, 250, 500]
    indices  = np.digitize(pt, bins, right=True) -1
    hist_sig = np.histogram(pt[labels==0], bins)[0]
    hist_bkg = np.histogram(pt[labels!=0], bins)[0]
    if bkg_ratio == None: bkg_ratio = np.sum(hist_bkg)/np.sum(hist_sig)
    total_sig = np.int_(np.around(np.minimum(hist_sig, hist_bkg/bkg_ratio)))
    total_bkg = np.int_(np.around(np.minimum(hist_bkg, hist_sig*bkg_ratio)))
    ind_sig   = [np.where((indices==n) & (labels==0))[0][:total_sig[n]] for n in np.arange(len(bins)-1)]
    ind_bkg   = [np.where((indices==n) & (labels!=0))[0][:total_bkg[n]] for n in np.arange(len(bins)-1)]
    valid_ind = np.concatenate(ind_sig+ind_bkg); np.random.seed(0); np.random.shuffle(valid_ind)
    train_ind = list(set(np.arange(len(pt))) - set(valid_ind))
    valid_sample = {key:np.take(sample[key], valid_ind, axis=0) for key in sample}
    valid_labels = np.take(labels, valid_ind)
    extra_sample = {key:np.take(sample[key], train_ind, axis=0) for key in sample}
    extra_labels = np.take(labels, train_ind)
    return valid_sample, valid_labels, extra_sample, extra_labels


def match_distributions(sample, labels, target_sample, target_labels):
    pt = sample['p_et_calo']; target_pt = target_sample['p_et_calo']
    bins = [0, 10, 20, 30, 40, 60, 80, 100, 130, 180, 250, 500]
    indices         = np.digitize(pt, bins, right=False) -1
    hist_sig        = np.histogram(       pt[labels==0]       , bins)[0]
    hist_bkg        = np.histogram(       pt[labels!=0]       , bins)[0]
    hist_sig_target = np.histogram(target_pt[target_labels==0], bins)[0]
    hist_bkg_target = np.histogram(target_pt[target_labels!=0], bins)[0]
    total_sig   = hist_sig_target * np.max(np.append(hist_sig/hist_sig_target, hist_bkg/hist_bkg_target))
    total_bkg   = hist_bkg_target * np.max(np.append(hist_sig/hist_sig_target, hist_bkg/hist_bkg_target))
    weights_sig = total_sig/hist_sig * len(labels)/np.sum(total_sig+total_bkg)
    weights_bkg = total_bkg/hist_bkg * len(labels)/np.sum(total_sig+total_bkg)
    return np.where(labels==0, weights_sig[indices], weights_bkg[indices])


def get_dataset(host_name='lps', node_dir='', eta_region=''):
    if 'lps'    in host_name                   : node_dir = '/opt/tmp/godin/e-ID_data/presamples'
    if 'beluga' in host_name and node_dir == '': node_dir = '/project/def-arguinj/shared/e-ID_data/2020-10-30'
    if eta_region in ['0.0-1.3', '0.0-1.3_old', '1.3-1.6', '1.6-2.5', '0.0-2.5']:
        folder = node_dir+'/'+eta_region
        data_files = sorted([folder+'/'+h5_file for h5_file in os.listdir(folder) if 'e-ID_' in h5_file])
    else:
        barrel_dir, midgap_dir, endcap_dir = [node_dir+'/'+folder for folder in ['0.0-1.3', '1.3-1.6', '1.6-2.5']]
        barrel_files = sorted([barrel_dir+'/'+h5_file for h5_file in os.listdir(barrel_dir) if 'e-ID_' in h5_file])
        midgap_files = sorted([midgap_dir+'/'+h5_file for h5_file in os.listdir(midgap_dir) if 'e-ID_' in h5_file])
        endcap_files = sorted([endcap_dir+'/'+h5_file for h5_file in os.listdir(barrel_dir) if 'e-ID_' in h5_file])
        data_files = [h5_file for group in zip(barrel_files, midgap_files, endcap_files) for h5_file in group]
    #for key, val in h5py.File(data_files[0], 'r').items(): print(key, val.shape)
    return data_files


def make_sample(data_file, idx, input_data, n_tracks, n_classes, verbose='OFF', prefix='p_'):
    upsize_images = False; preprocess_images = False
    scalars, images, others = input_data.values()
    if verbose == 'ON':
        print('Loading sample [', format(str(idx[0]),'>8s')+', '+format(str(idx[1]),'>8s'), end='] ')
        print('from', data_file.split('/')[-2]+'/'+data_file.split('/')[-1], end=' --> ', flush=True)
        start_time = time.time()
    with h5py.File(data_file, 'r') as data:
        sample = {key:data[key][idx[0]:idx[1]] for key in set(scalars+others)&set(data)}
        if 'constituents' in scalars:
            sample['constituents'] = sample['constituents'][:,:400]#/sample['rljet_pt_calo'][:,np.newaxis]
        if 'JZW' not in sample:
            sample['JZW'] = np.full(len(sample[list(sample.keys())[0]]), -1, dtype=np.float32)
        if 'weights' not in sample:
            sample['weights'] = np.full(len(sample[list(sample.keys())[0]]), 1, dtype=np.float32)
    #if tf.__version__ < '2.1.0':
    #    for key in set(sample)-set(others): sample[key] = np.float32(sample[key])
    labels = make_labels(sample, n_classes)
    if verbose == 'ON': print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return sample, labels


def make_labels(sample, n_classes, match_to_vertex=False):
    try: return sample['labels']
    except: return np.int_( np.where(sample['JZW']==-1, 0, 1) )


def weights_factors(JZW, data_file):
    if np.any(JZW == -1):
        factors = len(list(h5py.File(data_file,"r").values())[0])/len(JZW)
    else:
        file_JZW = np.int_(h5py.File(data_file,"r")['JZW'])
        n_JZW    = [np.sum(file_JZW==n) for n in range(int(np.max(file_JZW))+1)]
        # old samples JZWs
        #n_JZW = [ 35596, 13406964, 15909276, 17831457, 15981239, 15997303, 13913843, 13983297, 15946135, 15993849]
        # UFO samples JZWs
        #n_JZW = [181406, 27395322, 20376000, 13928294,  6964549,  6959061,  3143354,   796201,   796171,   796311]
        factors = np.ones_like(JZW, dtype=np.float32)
        for n in np.arange(len(n_JZW)):
            if np.sum(JZW==n) != 0: factors[JZW==n] = n_JZW[n]/np.sum(JZW==n)
    return factors


def batch_idx(data_files, batch_size, interval, weights=None, shuffle='OFF'):
    def return_idx(n_e, cum_batches, batch_size, index):
        file_index  = np.searchsorted(cum_batches, index, side='right')
        batch_index = index - np.append(0, cum_batches)[file_index]
        idx         = batch_index*batch_size; idx = [idx, min(idx+batch_size, n_e[file_index])]
        return file_index, idx
    #n_e = [len(h5py.File(data_file,'r')['eventNumber']) for data_file in data_files]
    n_e = [len(h5py.File(data_file,'r')['constituents']) for data_file in data_files]
    cum_batches = np.cumsum(np.int_(np.ceil(np.array(n_e)/batch_size)))
    indexes     = [return_idx(n_e, cum_batches, batch_size, index) for index in np.arange(cum_batches[-1])]
    cum_n_e     = np.cumsum(np.diff(list(zip(*indexes))[1]))
    n_e_index   = np.searchsorted(cum_n_e, [interval[0],interval[1]-1], side='right')
    batch_list  = [indexes[n] for n in np.arange(cum_batches[-1]) if n >= n_e_index[0] and n <= n_e_index[1]]
    cum_n_e     = [cum_n_e[n] for n in np.arange(cum_batches[-1]) if n >= n_e_index[0] and n <= n_e_index[1]]
    batch_list[ 0][1][0] = batch_list[ 0][1][1] + interval[0] - cum_n_e[ 0]
    batch_list[-1][1][1] = batch_list[-1][1][1] + interval[1] - cum_n_e[-1]
    if shuffle == 'ON': batch_list = utils.shuffle(batch_list, random_state=0)
    batch_dict = {batch_list.index(n):{'file':n[0], 'indices':n[1], 'weights':None} for n in batch_list}
    if np.all(weights) != None:
        weights = np.split(weights, np.cumsum(n_e))
        for n in batch_list: batch_dict[batch_list.index(n)]['weights'] = weights[n[0]][n[1][0]:n[1][1]]
    #for key in batch_dict: print(key, batch_dict[key])
    return batch_dict


def merge_samples(data_files, idx, input_data, n_tracks, n_classes, cuts, scaler=None, t_scaler=None):
    batch_dict = batch_idx(data_files, np.diff(idx)[0], idx)
    samples, labels = zip(*[make_sample(data_files[batch_dict[key]['file']], batch_dict[key]['indices'],
                            input_data, n_tracks, n_classes, verbose='ON') for key in batch_dict])
    labels = np.concatenate(labels); sample = {}
    for key in list(samples[0].keys()):
        sample[key] = np.concatenate([n[key] for n in samples])
        for n in samples: n.pop(key)
    indices = np.where( np.logical_and(labels!=-1, eval(cuts) if cuts!='' else True) )[0]
    sample, labels, _ = sample_cuts(sample, labels, cuts=cuts, verbose='ON')
    if scaler != None:
        sample = apply_scaler(sample, input_data['scalars'], scaler, verbose='ON')
    if t_scaler != None: sample = apply_t_scaler(sample, t_scaler, verbose='ON')
    #else: print()
    return sample, labels, indices


class Batch_Generator(tf.keras.utils.Sequence):
    def __init__(self, data_files, indexes, input_data, n_tracks, n_classes,
                 batch_size, cuts, scaler, t_scaler, weights=None, shuffle='OFF'):
        self.data_files = data_files; self.indexes    = indexes
        self.input_data = input_data; self.n_tracks   = n_tracks
        self.n_classes  = n_classes ; self.batch_size = batch_size
        self.cuts       = cuts      ; self.scaler     = scaler ;self.t_scaler = t_scaler
        self.weights    = weights   ; self.shuffle    = shuffle
        self.batch_dict = batch_idx(self.data_files, self.batch_size, self.indexes, self.weights, self.shuffle)
    def __len__(self):
        #number of batches per epoch
        return len(self.batch_dict)
    def __getitem__(self, gen_index):
        file_index = self.batch_dict[gen_index]['file']
        file_idx   = self.batch_dict[gen_index]['indices']
        weights    = self.batch_dict[gen_index]['weights']
        data_file  = self.data_files[file_index]
        sample, labels = make_sample(data_file, file_idx, self.input_data, self.n_tracks, self.n_classes)
        sample, labels, weights = sample_cuts(sample, labels, weights, self.cuts)
        if len(labels) != 0:
            if self.scaler != None: sample = apply_scaler(sample, self.input_data['scalars'], self.scaler)
            if self.t_scaler != None: sample = apply_t_scaler(sample, self.t_scaler)
        return sample, labels, weights


def sample_cuts(sample, labels, weights=None, cuts='', verbose='OFF'):
    if np.sum(labels==-1) != 0:
        length = len(labels)
        sample = {key:sample[key][labels!=-1] for key in sample}
        if np.all(weights) != None: weights = weights[labels!=-1]
        labels = labels[labels!=-1]
        if verbose == 'ON':
            print('Applying IFF labels cuts -->', format(len(labels),'8d'), 'e conserved', end=' ')
            print('(' + format(100*len(labels)/length, '.2f') + ' %)')
    if cuts != '':
        length = len(labels)
        labels = labels[eval(cuts)]
        if np.all(weights) != None: weights = weights[eval(cuts)];
        sample = {key:sample[key][eval(cuts)] for key in sample}
        if verbose == 'ON':
            print('Applying features cuts -->', format(len(labels),'8d') ,'e conserved', end=' ')
            print('(' + format(100*len(labels)/length,'.2f')+' %)\napplied cuts:', cuts)
    return sample, labels, weights


def process_images(sample, image_list, verbose='OFF'):
    if image_list == []: return sample
    if verbose == 'ON':
        start_time = time.time()
        print('Processing images for best axis', end=' --> ', flush=True)
    for cal_image in [key for key in image_list if 'tracks' not in key]:
        images = sample[cal_image]
        images = images.T
        #images  = abs(images)                               # negatives to positives
        #images -= np.minimum(0, np.min(images, axis=(0,1))) # shift to positive domain
        images = np.maximum(0, images)                       # clips negative values
        mean_1 = np.mean(images[:images.shape[0]//2   ], axis=(0,1))
        mean_2 = np.mean(images[ images.shape[0]//2:-1], axis=(0,1))
        images = np.where(mean_1 > mean_2, images[::-1,::-1,:], images).T
        sample[cal_image] = images
    if verbose == 'ON': print('('+format(time.time()-start_time,'.1f'), '\b'+' s)')
    return sample


def process_images_mp(sample, image_list, verbose='OFF', n_tasks=16):
    if image_list == []: return sample
    def rotation(images, indices, return_dict):
        images = images[indices[0]:indices[1]].T
        #images  = abs(images)                               # negatives to positives
        #images -= np.minimum(0, np.min(images, axis=(0,1))) # shift to positive domain
        images = np.maximum(0, images)                       # clips negative values
        mean_1 = np.mean(images[:images.shape[0]//2   ], axis=(0,1))
        mean_2 = np.mean(images[ images.shape[0]//2:-1], axis=(0,1))
        return_dict[indices] = np.where(mean_1 > mean_2, images[::-1,::-1,:], images).T
    n_samples = len(sample['eventNumber'])
    idx_list  = [task*(n_samples//n_tasks) for task in np.arange(n_tasks)] + [n_samples]
    idx_list  = list( zip(idx_list[:-1], idx_list[1:]) )
    if verbose == 'ON':
        start_time = time.time()
        print('Processing images for best axis', end=' --> ', flush=True)
    for cal_image in [key for key in image_list if 'tracks' not in key]:
        images    = sample[cal_image]; manager = mp.Manager(); return_dict = manager.dict()
        processes = [mp.Process(target=rotation, args=(images, idx, return_dict)) for idx in idx_list]
        for job in processes: job.start()
        for job in processes: job.join()
        sample[cal_image] = np.concatenate([return_dict[idx] for idx in idx_list])
    if verbose == 'ON': print('('+format(time.time()-start_time,'.1f'), '\b'+' s)')
    return sample


def fit_scaler(sample, scalars, scaler_out):
    print('Fitting quantile transform to training scalars', end=' --> ', flush=True)
    start_time = time.time()
    scalars = [scalar for scalar in scalars if scalar != 'constituents']
    scalars_array = np.hstack([np.expand_dims(sample[key], axis=1) for key in scalars])
    #scaler = preprocessing.QuantileTransformer(output_distribution='normal', n_quantiles=10000, random_state=0)
    #scaler = preprocessing.MaxAbsScaler()
    scaler = preprocessing.RobustScaler()
    scaler.fit(scalars_array) #scaler.fit_transform(scalars_array)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    print('Saving transform to', scaler_out, '\n')
    pickle.dump(scaler, open(scaler_out, 'wb'))
    return scaler


def apply_scaler(sample, scalars, scaler, verbose='OFF'):
    if verbose == 'ON':
        start_time = time.time()
        print('Applying quantile transform to scalar features', end=' --> ', flush=True)
    scalars = [scalar for scalar in scalars if scalar != 'constituents']
    scalars_array = scaler.transform(np.hstack([np.expand_dims(sample[key], axis=1) for key in scalars]))
    for n in np.arange(len(scalars)): sample[scalars[n]] = scalars_array[:,n]
    if verbose == 'ON': print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)\n')
    return sample


def fit_t_scaler(sample, scaler_out, reshape=True):
    jets = sample['constituents']
    print('Fitting quantile transform', end='  --> ', flush=True); start_time = time.time()
    if reshape: jets = np.reshape(jets, (-1,4))
    #scaler = preprocessing.QuantileTransformer(n_quantiles=10000, random_state=0).fit(jets)
    #scaler = preprocessing.MaxAbsScaler().fit(jets)
    scaler = preprocessing.RobustScaler().fit(jets)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    print('Saving transform to', scaler_out)
    pickle.dump(scaler, open(scaler_out, 'wb'))
    return scaler


def apply_t_scaler(sample, scaler, reshape=True, verbose='OFF'):
    jets = sample['constituents']
    if verbose == 'ON':
        start_time = time.time()
        print('Applying quantile transform', end=' --> ', flush=True)
    shape = jets.shape
    if reshape: jets = np.reshape(jets, (-1,4))
    jets = scaler.transform(jets)
    jets = np.reshape(jets, shape)
    sample['constituents'] = jets
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return sample


def sample_composition(sample):
    MC_type, IFF_type = sample['p_TruthType']    , sample['p_iffTruth']
    MC_list, IFF_list = np.arange(max(MC_type)+1), np.arange(max(IFF_type)+1)
    ratios = np.array([ [np.sum(MC_type[IFF_type==IFF]==MC) for MC in MC_list] for IFF in IFF_list ])
    IFF_sum, MC_sum = 100*np.sum(ratios, axis=0)/len(MC_type), 100*np.sum(ratios, axis=1)/len(MC_type)
    ratios = np.round(1e4*ratios/len(MC_type))/100
    MC_empty, IFF_empty = np.where(np.sum(ratios, axis=0)==0)[0], np.where(np.sum(ratios, axis=1)==0)[0]
    MC_list,  IFF_list  = list(set(MC_list)-set(MC_empty))      , list(set(IFF_list)-set(IFF_empty))
    print('IFF AND MC TRUTH CLASSIFIERS TRAINING SAMPLE COMPOSITION (', '\b'+str(len(MC_type)), 'e)')
    dash = (26+7*len(MC_list))*'-'
    print(dash, format('\n| IFF \ MC |','10s'), end='')
    for col in MC_list:
        print(format(col, '7.0f'), end='   |  Total  | \n' + dash + '\n' if col==MC_list[-1] else '')
    for row in IFF_list:
        print('|', format(row, '5.0f'), '   |', end='' )
        for col in MC_list:
            print(format(ratios[row,col], '7.0f' if ratios[row,col]==0 else '7.2f'), end='', flush=True)
        print('   |' + format(MC_sum[row], '7.2f')+ '  |')
        if row != IFF_list[-1]: print('|' + 10*' ' + '|' + (3+7*len(MC_list))*' ' + '|' + 9*' ' + '|')
    print(dash + '\n|   Total  |', end='')
    for col in MC_list: print(format(IFF_sum[col], '7.2f'), end='')
    print('   |  100 %  |\n' + dash + '\n')


def class_ratios(labels):
    def get_ratios(labels, n, return_dict): return_dict[n] = 100*np.sum(labels==n)/len(labels)
    manager   =  mp.Manager(); return_dict = manager.dict(); n_classes = max(labels) + 1
    processes = [mp.Process(target=get_ratios, args=(labels, n, return_dict)) for n in np.arange(n_classes)]
    for job in processes: job.start()
    for job in processes: job.join()
    return [return_dict[n] for n in np.arange(n_classes)]


def compo_matrix(valid_labels, train_labels=[], valid_probs=[]):
    valid_pred   = np.argmax(valid_probs, axis=1) if valid_probs != [] else valid_labels
    matrix       = metrics.confusion_matrix(valid_labels, valid_pred)
    matrix       = 100*matrix.T/matrix.sum(axis=1); n_classes = len(matrix)
    classes      = ['CLASS '+str(n) for n in np.arange(n_classes)]
    valid_ratios = class_ratios(valid_labels)
    train_ratios = class_ratios(train_labels) if train_labels != [] else n_classes*['n/a']
    if valid_probs == []:
        print('+---------------------------------------+\n| CLASS DISTRIBUTIONS'+19*' '+'|')
        headers = ['CLASS #', 'TRAIN (%)', 'VALID (%)']
        table   = zip(classes, train_ratios, valid_ratios)
        print(tabulate(table, headers=headers, tablefmt='psql', floatfmt=".2f"))
    else:
        if n_classes > 2:
            headers = ['CLASS #', 'TRAIN', 'VALID'] + classes
            table   = [classes] + [train_ratios] + [valid_ratios] + matrix.T.tolist()
            table   = list(map(list, zip(*table)))
            print_dict[2]  = '+'+31*'-'+'+'+35*'-'+12*(n_classes-3)*'-'+'+\n| CLASS DISTRIBUTIONS (%)'
            print_dict[2] += '       | VALID SAMPLE PREDICTIONS (%)      '+12*(n_classes-3)*' '+ '|\n'
        else:
            headers = ['CLASS #', 'TRAIN (%)', 'VALID (%)', 'ACC. (%)']
            table   = zip(classes, train_ratios, valid_ratios, matrix.diagonal())
            print_dict[2]  = '+----------------------------------------------------+\n'
            print_dict[2] += '| CLASS DISTRIBUTIONS AND VALID SAMPLE ACCURACIES    |\n'
        valid_accuracy = np.array(valid_ratios) @ np.array(matrix.diagonal())/100
        print_dict[2] += tabulate(table, headers=headers, tablefmt='psql', floatfmt=".2f")+'\n'
        print_dict[2] += 'VALIDATION SAMPLE ACCURACY: '+format(valid_accuracy,'.2f')+' %\n'


def validation(output_dir, results_in, plotting, n_valid, data_files, inputs, valid_cuts, sep_bkg, diff_plots):
    print('\nLOADING VALIDATION RESULTS FROM', output_dir+'/'+results_in)
    valid_data = pickle.load(open(output_dir+'/'+results_in, 'rb'))
    if len(valid_data) > 1: sample, labels, probs   = valid_data
    else:                                  (probs,) = valid_data
    n_e = min(len(probs), int(n_valid[1]-n_valid[0]))
    if False or len(valid_data) == 1: #add variables to the results
        print('Loading valid sample', n_e, end=' --> ', flush=True)
        sample, labels, _ = merge_samples(data_files, n_valid, inputs, n_tracks=5, n_classes=probs.shape[1],
                                          valid_cuts=valid_cuts, scaler=None)
        n_e = len(labels)
    sample, labels, probs = {key:sample[key][:n_e] for key in sample}, labels[:n_e], probs[:n_e]
    #multi_cuts(sample, labels, probs, output_dir); sys.exit()
    if False: #save the added variables to the results file
        print('Saving validation data to:', output_dir+'/'+'valid_data.pkl', '\n')
        pickle.dump((sample, labels, probs), open(output_dir+'/'+'valid_data.pkl','wb')); sys.exit()
    print('GENERATING PERFORMANCE RESULTS FOR', n_e, 'ELECTRONS', end=' --> ', flush=True)
    #valid_cuts = '(labels==0) & (probs[:,0]<=0.05)'
    #valid_cuts = '(sample["pt"] >= 20) & (sample["pt"] <= 80)'
    cuts = n_e*[True] if valid_cuts == '' else eval(valid_cuts)
    sample, labels, probs = {key:sample[key][cuts] for key in sample}, labels[cuts], probs[cuts]
    if False: #generate calorimeter images
        layers = ['em_barrel_Lr0'  , 'em_barrel_Lr1_fine'  , 'em_barrel_Lr2', 'em_barrel_Lr3',
                  'tile_barrel_Lr1', 'tile_barrel_Lr2', 'tile_barrel_Lr3']
        from plots_DG import cal_images
        cal_images(sample, labels, layers, output_dir, mode='mean', soft=False)
    if len(labels) == n_e: print('')
    else: print('('+str(len(labels))+' selected = '+format(100*len(labels)/n_e,'0.2f')+'%)')
    valid_results(sample, labels, probs, [], None, output_dir, plotting, sep_bkg, diff_plots); print()
    #sample_histograms(sample, labels, None, None, weights=None, bins=None, output_dir=output_dir)


def multi_cuts(sample, labels, probs, output_dir, input_file=None, step=0.2, index=6, multi=True):
    import itertools; from functools import partial
    global efficiencies
    def efficiency(labels, cuts, class_type):
        class_cut = labels!=0 if class_type=='bkg' else labels==class_type
        return np.sum(np.logical_and(class_cut, cuts))/np.sum(class_cut)
    def classes_eff(labels, probs, fracs, multi):
        if multi: cuts = probs[:,0] >= np.max(probs[:,1:]*(fracs/(1-fracs)), axis=1)
        else    : cuts = probs[:,0] >= (probs[:,1:]@fracs[1:])*(fracs[0]/(1-fracs[0])) #>= probs@fracs
        return [efficiency(labels, cuts, class_type) for class_type in list(range(probs.shape[1]))+['bkg']]
    def efficiencies(labels, probs, cut_tuples, multi, idx):
        return [classes_eff(labels, probs, cut_tuples[n], multi) for n in np.arange(idx[0],idx[1])]
    def apply_filter(ROC_values, index):
        pos_rates=[]; min_eff=1
        for n in np.arange(len(ROC_values)):
            if ROC_values[n][index] < min_eff:
                min_eff = ROC_values[n][index]
                pos_rates.append(ROC_values[n])
        return np.array(pos_rates)
    def ROC_curves(sample, labels, probs, bkg, ROC_values, output_dir):
        sample, labels, _ = discriminant(sample, labels, probs, sig_list=[0], bkg=bkg, printing=False)
        output_dir += '/'+'class_0_vs_'+str(bkg)
        plot_ROC_curves(sample, labels, None, output_dir, ROC_type=2, ECIDS=bkg==1, ROC_values=ROC_values)
    #input_file = 'pos_rates_step-0.2_solo.pkl'
    if input_file == None:
        start_time = time.time()
        repeat     = probs.shape[1]-1 if multi else probs.shape[1]
        cut_list   = np.arange(0, 1, step)
        cut_tuples = np.array(list(itertools.product(cut_list,repeat=repeat)))
        idx_tuples = get_idx(len(cut_tuples), n_sets=mp.cpu_count()); print(len(cut_tuples),'cuts')
        ROC_values = mp.Pool().map(partial(efficiencies, labels, probs, cut_tuples, multi), idx_tuples)
        ROC_values = np.concatenate(ROC_values)
        ROC_values = ROC_values[ROC_values[:,0].argsort()[::-1]]
        pickle.dump(ROC_values, open(output_dir+'/'+'pos_rates.pkl','wb'))
        print('Run time:', format(time.time()-start_time, '2.2f'), '\b'+' s\n')
    else:
        ROC_values = pickle.load(open(output_dir+'/'+input_file,'rb'))
    #for row in apply_filter(ROC_values, index):
    #    print(format(100*row[0],'5.1f')+' % -->', [int(1/row[n]) for n in range(1,ROC_values.shape[1])])
    ROC_values = (ROC_values[::int(np.ceil(ROC_values.shape[0]/10e6))], apply_filter(ROC_values, index))
    processes  = [mp.Process(target=ROC_curves, args=(sample, labels, probs, bkg, ROC_values, output_dir))
                  for bkg in list(range(1,ROC_values[0].shape[1]-1))+['bkg']]
    for job in processes: job.start()
    for job in processes: job.join()


def cross_valid(valid_sample, valid_labels, scalars, output_dir, n_folds, data_files, n_valid,
                input_data, n_tracks, valid_cuts, model, generator='OFF', verbose=1):
    print('########################################################################'  )
    print('##### STARTING '+str(n_folds)+'-FOLD CROSS-VALIDATION ################################'  )
    print('########################################################################\n')
    n_classes    = max(valid_labels)+1
    valid_probs  = np.full(valid_labels.shape + (n_classes,), -1.)
    event_number = valid_sample['eventNumber']
    for fold_number in np.arange(1, n_folds+1):
        print('FOLD '+str(fold_number)+'/'+str(n_folds), 'EVALUATION')
        weight_file = output_dir+'/model_' +str(fold_number)+'.h5'
        scaler_file = output_dir+'/scaler_'+str(fold_number)+'.pkl'
        print('Loading pre-trained weights from', weight_file)
        model.load_weights(weight_file); start_time = time.time()
        indices =               np.where(event_number%n_folds==fold_number-1)[0]
        labels  =           valid_labels[event_number%n_folds==fold_number-1]
        sample  = {key:valid_sample[key][event_number%n_folds==fold_number-1] for key in valid_sample}
        if scalars != [] and os.path.isfile(scaler_file):
            print('Loading scalars scaler from', scaler_file)
            scaler = pickle.load(open(scaler_file, 'rb'))
            sample = apply_scaler(sample, scalars, scaler, verbose='ON')
        print('\033[FCLASSIFIER:', weight_file.split('/')[-1], 'class predictions for', len(labels), 'e')
        if generator == 'ON':
            valid_cuts = '(sample["eventNumber"]%'+str(n_folds)+'=='+str(fold_number-1)+')'
            valid_gen  = Batch_Generator(data_files, n_valid, input_data, n_tracks, n_classes,
                                         batch_size=20000, cuts=valid_cuts, scaler=scaler)
            probs = model.predict(valid_gen, workers=4, verbose=verbose)
        else: probs = model.predict(sample, batch_size=20000, verbose=verbose)
        print('FOLD '+str(fold_number)+'/'+str(n_folds)+' ACCURACY: ', end='')
        print(format(100*valid_accuracy(labels, probs), '.2f'), end='')
        print(' % (', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)\n')
        #for n in np.arange(len(indices)): valid_probs[indices[n],:] = probs[n,:]
        for n in np.arange(n_classes): np.put(valid_probs[:,n], indices, probs[:,n])
    print('MERGING ALL FOLDS AND PREDICTING CLASSES ...')
    return valid_sprobs


def discriminant(sample, labels, probs, sig_list, bkg, printing=True):
    from functools import reduce
    def class_weights(sig_list, optimal_bkg=None):
        weights = np.ones(probs.shape[1])
        if optimal_bkg != None:
            for n in np.arange(probs.shape[1]):
                if n not in sig_list and n != optimal_bkg: weights[n] = 0
        if optimal_bkg == 'bkg': weights = class_ratios(labels)
        return weights
    if probs.shape[1] > 2: #multi-class discriminant
        bkg_list = set(np.arange(probs.shape[1]))-set(sig_list)
        bkg      = bkg_list if bkg=='bkg' else [bkg]
        if printing: print_dict[1] += 'SIGNAL = '+str(set(sig_list))+' vs BACKGROUND = '+str(set(bkg))+'\n'
        #for n in [None]+list(np.arange(1, probs.shape[1]))+['bkg']: print(class_weights(sig_list, n))
        weights = class_weights(sig_list, 'bkg')
        labels = np.array([0 if label in sig_list else 1 if label in bkg else -1 for label in labels])
        sig_probs = reduce(np.add, [weights[n]*probs[:,n] for n in sig_list])[labels!=-1]
        bkg_probs = reduce(np.add, [weights[n]*probs[:,n] for n in bkg_list])[labels!=-1]
        sample    = {key:sample[key][labels!=-1] for key in sample}
        labels    = labels[labels!=-1]
        sig_probs = np.where(sig_probs!=bkg_probs, sig_probs, 0.5)
        bkg_probs = np.where(sig_probs!=bkg_probs, bkg_probs, 0.5)
        probs     = sig_probs/(sig_probs+bkg_probs)
    else: #background separation for binary classification
        if bkg == 'bkg': return sample, labels, probs[:,0]
        if printing: print_dict[1] += 'SIGNAL = {0} VS BACKGROUND = {'+str(bkg)+'}\n'
        multi_labels = make_labels(sample, n_classes=6)
        cuts = np.logical_or(multi_labels==0, multi_labels==bkg)
        sample, labels, probs = {key:sample[key][cuts] for key in sample}, labels[cuts], probs[:,0][cuts]
    return sample, labels, probs


def print_performance(labels, probs, sig_eff=[90, 80, 70]):
    fpr, tpr, _ = metrics.roc_curve(labels, probs[:,0], pos_label=0)
    for val in sig_eff:
        print_dict[3] += 'BACKGROUND REJECTION AT '+str(val)+'%: '
        print_dict[3] += format(np.nan_to_num(1/fpr[np.argwhere(tpr>=val/100)[0]][0]),'>6.0f')+'\n'


def print_results(sample, labels, probs, plotting, output_dir, sig_list, bkg,
                  return_dict, separation=False, ECIDS=True):
    #if probs.shape[1] > 2: sample, labels, probs = discriminant  (sample, labels, probs, sig_list, bkg)
    #else                 : sample, labels, probs = bkg_separation(sample, labels, probs[:,0], bkg)
    sample, labels, probs = discriminant(sample, labels, probs, sig_list, bkg)
    #pickle.dump((sample,labels,probs), open(output_dir+'/'+'results_0_vs_'+str(bkg)+'.pkl','wb'))
    if plotting == 'ON':
        folder = output_dir+'/'+'class_0_vs_'+str(bkg)
        if not os.path.isdir(folder): os.mkdir(folder)
        arguments  = [(sample, labels, probs, folder, ROC_type, ECIDS and bkg==1) for ROC_type in [1]]
        processes  = [mp.Process(target=plot_ROC_curves, args=arg) for arg in arguments]
        arguments  = (sample, labels, probs, folder, separation and bkg=='bkg', bkg)
        processes += [mp.Process(target=plot_distributions_DG, args=arguments)]
        #arguments  = (sample, labels, probs, folder)
        #processes += [mp.Process(target=plot_weights, args=arguments)]
        for job in processes: job.start()
        for job in processes: job.join()
    return_dict[bkg] = print_dict


def valid_results(sample, labels, probs, train_labels, training, output_dir, plotting, sep_bkg, diff_plots):
    global print_dict; print_dict = {n:'' for n in [1,2,3]}
    print(); compo_matrix(labels, train_labels, probs); print(print_dict[2])
    sig_list = [0]
    bkg_list = ['bkg'] + list(set(np.arange(6))-set(sig_list)) if sep_bkg =='ON' else ['bkg']
    manager = mp.Manager(); return_dict = manager.dict()
    arguments = [(sample, labels, probs, plotting, output_dir, sig_list, bkg, return_dict) for bkg in bkg_list]
    processes = [mp.Process(target=print_results, args=arg) for arg in arguments]
    if training != None: processes += [mp.Process(target=plot_history, args=(training, output_dir,))]
    for job in processes: job.start()
    for job in processes: job.join()
    if plotting=='OFF': # bkg_rej extraction
        for bkg in bkg_list: print("".join(list(return_dict[bkg].values())))
        return [int(return_dict[bkg][3].split()[-1]) for bkg in bkg_list]


def feature_removal(scalars, images, groups, index):
    if   index <= 0: return scalars, images, 'none'
    elif index  > len(scalars+images+groups): sys.exit()
    elif index <= len(scalars+images):
        #removed_feature = (scalars+images)[index-1]
        removed_feature = dict(zip(np.arange(1, len(scalars+images)+1), scalars+images))[index]
        scalars_list    = list(set(scalars) - set([removed_feature]))
        images_list     = list(set(images ) - set([removed_feature]))
    elif index  > len(scalars+images):
        removed_feature = groups[index-1-len(scalars+images)]
        scalars_list    = list(set(scalars) - set(removed_feature))
        images_list     = list(set(images ) - set(removed_feature))
        removed_feature = 'group_'+str(index-len(scalars+images))
    scalars = [scalar for scalar in scalars if scalar in scalars_list]
    images  = [ image for  image in  images if  image in  images_list]
    return scalars, images, removed_feature


def feature_ranking(output_dir, results_out, scalars, images, groups):
    data_dict = {}
    with open(results_out,'rb') as file_data:
        try:
            while True: data_dict.update(pickle.load(file_data))
        except EOFError: pass
    try: pickle.dump(data_dict, open(results_out,'wb'))
    except IOError: print('FILE ACCESS CONFLICT IN FEATURE RANKING --> SKIPPING FILE ACCESS\n')
    # SECTION TO MODIFY
    #from importance import ranking_plot
    #ranking_plot(data_dict, output_dir, 'put title here', images, scalars, groups)
    print('BACKGROUND REJECTION DICTIONARY:')
    for key in data_dict: print(format(key,'30s'), data_dict[key])
