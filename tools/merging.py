import numpy as np
import multiprocessing as mp
import h5py, sys, time, os
from   sklearn import utils
from   root_utils import get_idx


def file_processing(data_path, n_constituents='unknown', n_files=40, n_tasks=10):
    data_files = sorted([h5_file for h5_file in os.listdir(data_path) if '.h5' in h5_file])
    shapes = [h5py.File(data_path+'/'+h5_file,'r')['constituents'].shape for h5_file in data_files]
    n_jets, max_components = zip(*shapes)
    if n_constituents == 'unknown':
        n_constituents = max(max_components)//4
    idx_list   = [get_idx(jet, n_sets=n_files) for jet in n_jets]
    file_idx   = list(utils.shuffle(np.arange(n_files), random_state=0))
    merge_path = data_path+'/'+'merging'; os.mkdir(merge_path)
    for idx in np.split(file_idx, np.arange(n_tasks, n_files, n_tasks)):
        start_time = time.time()
        arguments = [(data_path, data_files, idx_list, file_idx, out_idx, n_constituents) for out_idx in idx]
        processes = [mp.Process(target=mix_samples, args=arg) for arg in arguments]
        for job in processes: job.start()
        for job in processes: job.join()
        print('Run time:', format(time.time() - start_time, '2.1f'), '\b'+' s')
    merge_files(data_path+'/'+'merging'); print()
    for h5_file in os.listdir(merge_path):
        os.rename(merge_path+'/'+h5_file, data_path+'/../'+h5_file)
    os.rmdir(merge_path); print(); sys.exit()


def mix_samples(data_path, data_files, idx_list, file_idx, out_idx, n_constituents):
    type_dict = {'constituents':np.float16, 'rljet_n_constituents':np.uint8}
    keys = [key for key in h5py.File(data_path+'/'+data_files[0],'r')]
    for key in keys:
        sample_list = []
        for in_idx in utils.shuffle(np.arange(len(data_files)), random_state=out_idx):
            idx  = idx_list[in_idx][out_idx]
            data = h5py.File(data_path+'/'+data_files[in_idx],'r')[key]
            if key == 'constituents':
                sample = np.zeros((idx[1]-idx[0],4*n_constituents))
                sample[:,0:data.shape[1]] = data[idx[0]:idx[1]]
                sample_list += [np.float16(sample)]
                del sample
            else:
                sample_list += [data[idx[0]:idx[1]]]
        sample = np.concatenate(sample_list)
        file_name = 'merging'+'/'+data_path.split('/')[-1]+'_'+'{:=02}'.format(file_idx.index(out_idx))+'.h5'
        attribute = 'w' if keys.index(key)==0 else 'a'
        dtype     = type_dict[key] if key in type_dict else data.dtype
        shape     = (len(sample),) + sample.shape[1:]
        maxshape  = (None,) + sample.shape[1:]
        chunks    = (10000,)+sample.shape[1:]
        data      = h5py.File(data_path+'/'+file_name, attribute, rdcc_nbytes=20*1024**3, rdcc_nslots=10000019)
        data.create_dataset(key, shape, maxshape=maxshape, dtype=dtype, compression='lzf', chunks=chunks)
        data[key][:] = utils.shuffle(sample, random_state=0)


def merge_files(data_path):
    h5_files = sorted([h5_file for h5_file in os.listdir(data_path) if '.h5' in h5_file])
    idx = np.cumsum([len(h5py.File(data_path+'/'+h5_file, 'r')['constituents']) for h5_file in h5_files])
    output_file = data_path.split('/')[-2]+'.h5'
    os.rename(data_path+'/'+h5_files[0], data_path+'/'+output_file)
    data = h5py.File(data_path+'/'+output_file, 'a')
    print('MERGING DATA FILES IN:', end=' '); print(output_file, end=' .', flush=True)
    start_time = time.time()
    for key in data: data[key].resize((idx[-1],) + data[key].shape[1:])
    for h5_file in h5_files[1:]:
        index = h5_files.index(h5_file)
        for key in data: data[key][idx[index-1]:idx[index]] = h5py.File(data_path+'/'+h5_file, 'r')[key]
        os.remove(data_path+'/'+h5_file)
        print('.', end='', flush=True)
