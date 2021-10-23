import numpy           as np
import multiprocessing as mp
import awkward         as ak
import os, sys, time, h5py, itertools, uproot
from functools import partial
try: from ROOT import TLorentzVector
except ModuleNotFoundError: pass


def get_files(main_path, paths):
    root_files = {path.split('.')[2]: sorted([main_path+'/'+path+'/'+root_file+':nominal'
                  for root_file in os.listdir(main_path+'/'+path)]) for path in paths}
    return np.concatenate(list(root_files.values()))


def get_data(root_list, var_list, jet_var, n_constituents, sample_type, ID_weights, library):
    start_time = time.time()
    if sample_type in ['topo-dijet', 'UFO-dijet']       : var_list += ['JZW']
    if sample_type in ['topo-ttbar', 'UFO-ttbar', 'BSM']: var_list += ['DSID']
    with mp.Pool() as pool:
        root_tuples = list(itertools.product(root_list, var_list))
        func_args   = (jet_var, n_constituents, sample_type, ID_weights, library)
        arrays_list = pool.map(partial(root_conversion, *func_args), root_tuples)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    #return {key:np.concatenate([n[1] for n in arrays_list if key in n[0]]) for key in var_list}
    root_data = {key:np.concatenate([n[1] for n in arrays_list if key in n[0]]) for key in var_list}
    del arrays_list
    return root_data


def root_conversion(jet_var, n_constituents, sample_type, ID_weights, library, root_tuple):
    root_file, key = root_tuple
    DSID = root_file.split('.')[2]
    if key == 'JZW':
        arrays = uproot.open(root_file)['weight_mc'].array(library=library)
        return root_tuple, np.full_like(arrays, list(ID_weights.keys()).index(DSID), dtype=np.int8)
    if key == 'DSID':
        arrays = uproot.open(root_file)['weight_mc'].array(library=library)
        return root_tuple, np.full_like(arrays, DSID, dtype=np.int32)
    arrays = uproot.open(root_file)[key].array(library=library)
    if key in jet_var:
        if library == 'ak': arrays = [ak.to_numpy(n[0]) for n in arrays]
        else              : arrays = [            n[0]  for n in arrays]
        arrays = [np.pad(n, (0,max(n_constituents-len(n),0)),'constant')[:n_constituents] for n in arrays]
        arrays = np.float16(np.vstack(arrays)/1000. if key=='rljet_assoc_cluster_pt' else np.vstack(arrays))
        print(format(key,'23s'), 'converted from', '...' + DSID + '...' + root_file.split('._')[-1])
    else:
        if library == 'ak': arrays = ak.to_numpy(arrays)
        arrays = np.reshape(arrays, (len(arrays),))
        if key in ['rljet_m_calo', 'rljet_m_comb', 'rljet_pt_calo', 'rljet_pt_comb']: arrays = arrays/1000
        if key == 'weight_mc': arrays *= ID_weights[DSID]
    return root_tuple, arrays


def final_jets(jets, n_tasks=32):
    start_time = time.time()
    jets = np.concatenate([jets[key][...,np.newaxis] for key in jets          ], axis=2)
    jets = np.concatenate([jets, np.zeros_like(jets[...,:1], dtype=np.float32)], axis=2)
    manager    = mp.Manager(); return_dict = manager.dict()
    idx_tuples = get_idx(len(jets), n_sets=min(mp.cpu_count(), n_tasks))
    arguments  = [(jets[idx[0]:idx[1]], idx, return_dict) for idx in idx_tuples]
    processes  = [mp.Process(target=transform_jets, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return {key:np.concatenate([return_dict[idx].pop(key) for idx in idx_tuples])
            for key in return_dict[idx_tuples[0]]}


def get_idx(size, start_val=0, n_sets=8):
    n_sets   = min(size, n_sets)
    idx_list = [start_val + n*(size//n_sets) for n in np.arange(n_sets)] + [start_val+size]
    return list(zip(idx_list[:-1], idx_list[1:]))


def transform_jets(jets, idx, return_dict):
    for n in np.arange(len(jets)):
        jets[n,...] = jet_pt_ordering(jets[n,...])
        jets[n,...] = jet_Lorentz_4v (jets[n,...])
        jets[n,...] = jet_processing (jets[n,...])
    return_dict[idx] = {'constituents':np.float16(np.reshape(jets,(jets.shape[0],-1))),
                        **{key:val for key,val in get_4v(jets).items()}}


def get_4v(sample):
    sample = np.sum(sample, axis=1)
    E, px, py, pz = [sample[:,n] for n in np.arange(sample.shape[-1])]
    pt = np.sqrt(px**2 + py**2)
    M  = np.sqrt(np.maximum(0, E**2 - px**2 - py**2 - pz**2))
    return {'E':E, 'pt_calo':pt, 'm_calo':M}


def jet_pt_ordering(jet):
    pt_idx = np.argsort(jet[:,0])[::-1]
    for n in [0,1,2]: jet[:,n] = jet[:,n][pt_idx]
    return jet


def jet_Lorentz_4v(jet):
    for n in np.arange(len(jet)):
        if np.sum(jet[n,:]) != 0:
            v = TLorentzVector(0, 0, 0, 0)
            v.SetPtEtaPhiM(jet[n,0], jet[n,1], jet[n,2], jet[n,3])
            jet[n,:] = v.E(), v.Px(), v.Py(), v.Pz()
    return jet


def jet_processing(jet):
    # Find the jet (eta, phi)
    center=jet.sum(axis=0)
    v_jet=TLorentzVector(center[1], center[2], center[3], center[0])
    # Centering parameters
    phi=v_jet.Phi()
    bv = v_jet.BoostVector()
    bv.SetPerp(0)
    for n in np.arange(len(jet)):
        if np.sum(jet[n,:]) != 0:
            v = TLorentzVector(jet[n,1], jet[n,2], jet[n,3], jet[n,0])
            v.RotateZ(-phi)
            v.Boost(-bv)
            jet[n,:] = v[3], v[0], v[1], v[2] #(E,Px,Py,Pz)
    # Rotating parameters
    weighted_phi=0
    weighted_eta=0
    for n in np.arange(len(jet)):
        if np.sum(jet[n,:]) != 0:
            v = TLorentzVector(jet[n,1], jet[n,2], jet[n,3], jet[n,0])
            r = np.sqrt(v.Phi()**2 + v.Eta()**2)
            if r != 0: #in case there is only one component
                weighted_phi += v.Phi() * v.E()/r
                weighted_eta += v.Eta() * v.E()/r
    #alpha = np.arctan2(weighted_phi, weighted_eta) #approximately align at eta
    alpha = np.arctan2(weighted_eta, weighted_phi) #approximately align at phi
    for n in np.arange(len(jet)):
        if np.sum(jet[n,:]) != 0:
            v = TLorentzVector(jet[n,1], jet[n,2], jet[n,3], jet[n,0])
            #v.rotate_x(alpha) #approximately align at eta
            v.RotateX(-alpha) #approximately align at phi
            jet[n,:] = v[3], v[0], v[1], v[2] #(E,Px,Py,Pz)
    return jet


def count_constituents(root_files, verbose=False):
    if verbose:
        print('PROCESSED FILES:')
        for root_file in root_files: print(root_file)
    with mp.Pool() as pool:
        return np.max(pool.map(max_constituents, root_files))
def max_constituents(root_file):
    #events = uproot.open(root_file)['rljet_assoc_cluster_pt'].array(library='ak')
    #return np.max([len(n[0]) for n in events])
    events = uproot.open(root_file)['rljet_n_constituents'].array(library='np')
    return np.max([n for n in events])
