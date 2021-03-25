import numpy as np
import multiprocessing as mp
import h5py, pickle, bz2, sys, time, os
from   sklearn import utils
from   matplotlib        import pylab
from   matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, FixedLocator
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_idx(size, n_sets=50):
    idx_list = [n*(size//n_sets) for n in np.arange(n_sets)] + [size]
    return list(zip(idx_list[:-1], idx_list[1:]))
def get_pt(h5_file, idx, sub_idx):
    jets = h5py.File(h5_file,'r')['constituents'][idx[0]+sub_idx[0]:idx[0]+sub_idx[1]]
    jets = np.reshape(jets, (-1,int(jets.shape[1]/4),4))
    mult = jets.shape[1]         if jets.dtype == 'float16' else 1
    jets = np.mean(jets, axis=1) if jets.dtype == 'float16' else np.sum(jets, axis=1)
    E, px, py, pz = [jets[:,n] for n in np.arange(jets.shape[-1])]
    pt   = mult * np.sqrt(px**2 + py**2)
    mass = mult * np.sqrt(np.maximum(0, E**2 - px**2 - py**2 - pz**2))
    return mult*E, pt, mass
def pt_dict(idx_tuple, return_dict):
    h5_file, idx = idx_tuple
    E, pt, mass = list(zip(*[get_pt(h5_file, idx, sub_idx) for sub_idx in get_idx(idx[1]-idx[0])]))
    return_dict[idx_tuple] = {'E':np.concatenate(E), 'pt':np.concatenate(pt), 'mass':np.concatenate(mass)}
def jets_pt(data_files, data_path=None):
    if data_path != None:
        data_files = sorted([data_path+'/'+h5_file for h5_file in os.listdir(data_path) if '.h5' in h5_file])
    idx_tuples = [(n, idx) for n in data_files for idx in get_idx(len(h5py.File(n,'r')['constituents']))]
    manager    =  mp.Manager(); return_dict = manager.dict(); start_time = time.time()
    processes  = [mp.Process(target=pt_dict, args=(idx_tuple, return_dict)) for idx_tuple in idx_tuples]
    for task in processes: task.start()
    for task in processes: task.join()
    E       = np.concatenate([return_dict[idx_tuple]['E'      ] for idx_tuple in idx_tuples])
    pt      = np.concatenate([return_dict[idx_tuple]['pt'     ] for idx_tuple in idx_tuples])
    mass    = np.concatenate([return_dict[idx_tuple]['mass'   ] for idx_tuple in idx_tuples])
    weights = np.concatenate([h5py.File(h5_file,'r')['weights'] for h5_file   in data_files])
    print('run time:', format(time.time() - start_time, '2.1f'), '\b'+' s')
    return E, pt, mass, weights
def make_histograms(pt, weights, JZW, data_files):
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    bin_width = 50
    bins = np.arange(0, 8000, bin_width)

    #for n in np.arange(np.max(JZW)+1):
    #    pylab.hist(pt[JZW==n], bins, histtype='step', weights=36*weights[JZW==n], log=True, label='$p_t$', lw=2)
    #for n in np.arange(len(data_files)):
    #    pt, _, weights = jets_pt('', [data_files[n]])
    #    pylab.hist(pt, bins, histtype='step', weights=36*weights/bin_width, log=True, label='$p_t$', lw=2)

    #pt = pt[JZW<=5]
    #weights = weights[JZW<=5]
    pylab.hist(pt, bins, histtype='step', weights=weights/bin_width, log=True, label='$p_t$', lw=2, color='black')
    for idx in get_idx(len(pt), n_sets=5):
        print(idx)
        pylab.hist(pt[idx[0]:idx[1]], bins, histtype='step', weights=weights[idx[0]:idx[1]]/bin_width,
                   log=True, label='$p_t$', lw=2)

    pylab.xlim(0, 5000)
    pylab.ylim(1e-6, 1e6)
    axes.xaxis.set_minor_locator(AutoMinorLocator(10))
    plt.yticks([10**int(n) for n in np.arange(-6,7)])
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('$p_t$ (GeV)', fontsize=24)
    plt.ylabel('Distribution density (GeV$^{-1}$)', fontsize=24)
    #plt.ylabel('Distribution', fontsize=24)
    plt.legend(loc='upper right', fontsize=18)
    file_name ='pt_histo.png'
    plt.savefig(file_name)


#'''
#data_path  = '/nvme1/atlas/godin/AD_data/16-bit'
#data_files = sorted([data_path+'/'+h5_file for h5_file in os.listdir(data_path) if '.h5' in h5_file])
data_files = ['/nvme1/atlas/godin/AD_data/AtlasMCdijet36.h5']
#data_files = ['/opt/tmp/godin/AD_data/AtlasMCdijet36.h5']
#data_files = ['/lcg/storage18/atlas/pilette/atlasdata/train_test/AtlasMCdijetJZ3JZ8.h5']
#print([h5py.File(data_files[0], 'r')[key] for key in h5py.File(data_files[0], 'r')])
#sys.exit()

E, pt, mass, weights = jets_pt(data_files)
#pickle.dump({'E':E, 'pt':pt, 'mass':mass}, open('E_pt_mass.pkl','wb'))
#pickle.dump({'E':E, 'pt':pt, 'mass':mass}, bz2.BZ2File('E_pt_mass.pkl','wb'))
'''
with h5py.File('E_pt_mass.h5', 'w') as data:
    shape = E.shape
    chunks = (10000,)+shape[1:]
    data.create_dataset('E', shape, dtype=np.float16, compression='lzf', chunks=chunks)
    data['E'][:] = E
    data.create_dataset('pt', shape, dtype=np.float16, compression='lzf', chunks=chunks)
    data['pt'][:] = pt
    data.create_dataset('mass', shape, dtype=np.float16, compression='lzf', chunks=chunks)
    data['mass'][:] = mass
'''
#print( E.shape, np.min(E), np.max(E) )
#weights = h5py.File(data_files[0],'r')['weights']
#print(np.min(weights), np.max(weights), np.sum(weights))
sys.exit()

JZW = np.concatenate([h5py.File(h5_file,'r')['JZW'] for h5_file in data_files])
print('pt shape, mass_shape, weights shape =', pt.shape, '\b,', mass.shape, '\b,', weights.shape)

#make_histograms(pt, weights, JZW, data_files)
sys.exit()
#'''

#'''
### MIXING FUNCTIONS ###
def mix_samples(data_path, data_files, idx_list, file_idx, out_idx):
    for key in ['constituents', 'weights']:
        sample_list = []; JZW_list = []
        for in_idx in utils.shuffle(np.arange(len(data_files)), random_state=out_idx):
            idx  = idx_list[in_idx][out_idx]
            data = h5py.File(data_path+'/'+data_files[in_idx],'r')[key]
            if key == 'constituents':
                sample = np.zeros((idx[1]-idx[0],400))
                sample[:,0:data.shape[1]] = data[idx[0]:idx[1]]
                sample_list += [np.float16(sample)]; del sample
            else:
                sample_list += [data[idx[0]:idx[1]]]
                JZW_list    += [np.full((idx[1]-idx[0],), in_idx, dtype=np.uint8)]
        sample = np.concatenate(sample_list)
        file_name = 'merged/AtlasMCdijet36_'+'{:=02}'.format(file_idx.index(out_idx))+'.h5'
        attribute = 'w' if key=='constituents' else 'a'
        data   = h5py.File(data_path+'/'+file_name, attribute, rdcc_nbytes=20*1024**3, rdcc_nslots=10000019)
        shape  = (len(sample),) + sample.shape[1:]
        maxshape = (None,)+sample.shape[1:]
        dtype  = np.float16 if key=='constituents' else np.float64
        chunks = (10000,)+sample.shape[1:]
        data.create_dataset(key, shape, maxshape=maxshape, dtype=dtype, compression='lzf', chunks=chunks)
        data[key][:] = utils.shuffle(sample, random_state=0)
        if key == 'weights':
            data.create_dataset('JZW', shape, maxshape=maxshape, dtype=np.uint8, compression='lzf', chunks=chunks)
            data['JZW'][:] = utils.shuffle(np.concatenate(JZW_list), random_state=0)
    merge_files(data_path)
def merge_files(data_path):
    h5_files = sorted([h5_file for h5_file in os.listdir(data_path) if '.h5' in h5_file])
    idx = np.cumsum([len(h5py.File(data_path+'/'+h5_file, 'r')['constituents']) for h5_file in h5_files])
    output_file = h5_files[0].split('_')[0]+'.h5'
    os.rename(data_path+'/'+h5_files[0], data_path+'/'+output_file)
    data = h5py.File(data_path+'/'+output_file, 'a')
    print('MERGING DATA FILES IN:', end=' '); print('output/'+output_file, end=' .', flush=True)
    start_time = time.time()
    for key in data: data[key].resize((idx[-1],) + data[key].shape[1:])
    for h5_file in h5_files[1:]:
        index = h5_files.index(h5_file)
        for key in data: data[key][idx[index-1]:idx[index]] = h5py.File(data_path+'/'+h5_file, 'r')[key]
        os.remove(data_path+'/'+h5_file)
        print('.', end='', flush=True)
data_path  = '/nvme1/atlas/godin/AD_data/64-bit'
data_files = sorted([h5_file for h5_file in os.listdir(data_path) if '.h5' in h5_file])
n_jets     = [len(h5py.File(data_path+'/'+h5_file,'r')['constituents']) for h5_file in data_files]
n_files    = 40; n_tasks = 10
idx_list   = [get_idx(jet, n_files) for jet in n_jets]
file_idx   = list(utils.shuffle(np.arange(n_files), random_state=0))
for idx in np.split(file_idx, np.arange(n_tasks, n_files, n_tasks)):
    start_time = time.time()
    arguments = [(data_path, data_files, idx_list, file_idx, out_idx) for out_idx in idx]
    processes = [mp.Process(target=mix_samples, args=arg) for arg in arguments]
    for job in processes: job.start()
    for job in processes: job.join()
    print('run time:', format(time.time() - start_time, '2.1f'), '\b'+' s')
sys.exit()
#'''


'''
### MIXING PROGRAM ###
data_path  = '/nvme1/atlas/godin/anomaly_detection/Atlas_data/16-bit'
data_files = sorted([h5_file for h5_file in os.listdir(data_path) if '.h5' in h5_file])
for key in ['constituents', 'weights']:
    sample_list = []; start_time = time.time()
    for h5_file in data_files:
        if key == 'constituents':
            shape  = h5py.File(data_path+'/'+h5_file,'r')[key].shape
            sample = np.zeros((shape[0],100))
            sample[:,0:min(100,shape[1]-200)] = h5py.File(data_path+'/'+h5_file,'r')[key][:,200:300]
            sample_list += [sample]
        else:
            sample_list += [h5py.File(data_path+'/'+h5_file,'r')[key]]
    #if key == 'constituents': sample = np.concatenate([n[:,100:200] for n in sample_list])
    #else                    : sample = np.concatenate(sample_list)
    sample = np.concatenate(sample_list)
    file_name = 'merged/AtlasMCdijet36.h5'; attribute = 'w' if key=='constituents' else 'a'
    data   = h5py.File(data_path+'/'+file_name, attribute, rdcc_nbytes=20*1024**3, rdcc_nslots=10000019)
    shape  = (len(sample),) + sample.shape[1:]
    chunks = (10000,)+sample.shape[1:]
    data.create_dataset(key, shape, dtype=np.float16, compression='lzf', chunks=chunks)
    data[key][:] = utils.shuffle(np.float16(sample), random_state=0)
sys.exit()
'''

'''
### CONVERTING PROGRAM ###
input_path  = '/nvme1/atlas/godin/AD_data/64-bit'
output_path = '/nvme1/atlas/godin/AD_data/16-bit_test'
data_files  = sorted([h5_file for h5_file in os.listdir(input_path) if '.h5' in h5_file])
def convert_file(idx_tuple):
    file_in, file_idx, idx1 = idx_tuple
    file_out = file_in.split('.h5')[-2]+'_'+str(file_idx)+'.h5'
    data = h5py.File(output_path+'/'+file_out, 'w', rdcc_nbytes=1024**3)
    for key in ['constituents', 'weights']:
        shape = (idx1[1]-idx1[0],)+h5py.File(input_path+'/'+file_in,'r')[key].shape[1:]
        chunk_shape = (1000,)+shape[1:] if 'jet361023' in file_in else (10000,)+shape[1:]
        maxshape    = (None,)+shape[1:]
        dtype  = np.float16 if key=='constituents' else np.float64
        data.create_dataset(key, shape, dtype=dtype, compression='lzf', maxshape=maxshape, chunks=chunk_shape)
        for idx2 in get_idx(idx1[1]-idx1[0], n_sets=10):
            idx = (idx1[0]+idx2[0], idx1[0]+idx2[1])
            sample = h5py.File(input_path+'/'+file_in,'r')[key][idx[0]:idx[1]]
            if key == 'constituents': sample = np.float16(sample)
            data[key][idx2[0]:idx2[1]] = sample
def merge_files(h5_file):
    h5_files = sorted([filename for filename in os.listdir(output_path) if h5_file.split('.h5')[0] in filename])
    idx = np.cumsum([len(h5py.File(output_path+'/'+h5_file, 'r')['constituents']) for h5_file in h5_files])
    output_file = h5_files[0].split('_')[0]+'.h5'
    os.rename(output_path+'/'+h5_files[0], output_path+'/'+output_file)
    dataset = h5py.File(output_path+'/'+output_file, 'a')
    for key in dataset: dataset[key].resize((idx[-1],) + dataset[key].shape[1:])
    for h5_file in h5_files[1:]:
        index = h5_files.index(h5_file)
        for key in dataset: dataset[key][idx[index-1]:idx[index]] = h5py.File(output_path+'/'+h5_file, 'r')[key]
        os.remove(output_path+'/'+h5_file)
idx_list   = [get_idx(len(h5py.File(input_path+'/'+h5_file,'r')['constituents']), n_sets=5) for h5_file in data_files]
idx_tuples = [(data_files[n], idx_list[n].index(idx), idx) for n in np.arange(len(data_files)) for idx in idx_list[n]]
processes  = [mp.Process(target=convert_file, args=(idx_tuple,)) for idx_tuple in idx_tuples]
start_time = time.time()
for task in processes: task.start()
for task in processes: task.join()
print('run time:', format(time.time() - start_time, '2.1f'), '\b'+' s')
processes  = [mp.Process(target=merge_files, args=(h5_file,)) for h5_file in data_files]
for task in processes: task.start()
for task in processes: task.join()
print('run time:', format(time.time() - start_time, '2.1f'), '\b'+' s')
sys.exit()
'''

'''
path_in    = '/nvme1/atlas/godin/anomaly_detection/Atlas_data/64-bit'
path_out   = '/nvme1/atlas/godin/anomaly_detection/Atlas_data/16-bit'
data_files = sorted([h5_file for h5_file in os.listdir(path_in) if '.h5' in h5_file])
def convert_file(h5_file):
    start_time = time.time()
    file_name = path_out+'/'+h5_file
    with h5py.File(file_name, 'w', rdcc_nbytes=1024**3) as data:
        for key in ['constituents', 'weights']:
            size        = len(h5py.File(path_in+'/'+h5_file,'r')['constituents'])
            idx_list    = get_idx(size, n_sets=10)
            shape       = (size,) + h5py.File(path_in+'/'+h5_file,'r')[key].shape[1:]
            chunk_shape = (1000,)+shape[1:] if 'jet361023' in h5_file else (10000,)+shape[1:]
            data.create_dataset(key, shape, dtype=np.float16, compression='lzf', chunks=chunk_shape)
            for idx in idx_list:
                data[key][idx[0]:idx[1]] = np.float16(h5py.File(path_in+'/'+h5_file,'r')[key][idx[0]:idx[1]])
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)\n')
processes = [mp.Process(target=convert_file, args=(h5_file,)) for h5_file in data_files[:1]]
for job in processes: job.start()
for job in processes: job.join()
'''
