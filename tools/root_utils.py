import numpy           as np
import multiprocessing as mp
import awkward         as ak
import os, sys, time, h5py, itertools, uproot
from functools import partial
try: from ROOT import TLorentzVector
except ModuleNotFoundError: pass


def get_files(main_path, paths):
    return {path.split('.')[2]: sorted([main_path+'/'+path+'/'+root_file+':nominal' for
            root_file in os.listdir(main_path+'/'+path)]) for path in paths}


def get_data(root_list, var_list, jet_var, n_jets=100):
    start_time = time.time()
    with mp.Pool() as pool:
        root_tuples = list(itertools.product(root_list, var_list))
        arrays_list = pool.map(partial(root_conversion, jet_var, n_jets), root_tuples)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return {key:np.concatenate([n[1] for n in arrays_list if key in n[0]]) for key in var_list}


def root_conversion(jet_var, n_jets, root_tuple):
    root_file, key = root_tuple
    tag = root_file.split('.')[2]
    arrays = uproot.open(root_file)[key].array(library='ak')
    if key in jet_var:
        arrays = [np.pad(ak.to_numpy(n[0]), (0,max(n_jets-len(n[0]),0)),'constant')[:n_jets] for n in arrays]
        arrays = np.float16(np.vstack(arrays)/1000. if key=='rljet_assoc_cluster_pt' else np.vstack(arrays))
        print(format(key,'23s'), 'converted from', '...'+tag+'...'+root_file.split('._')[-1])
    else:
        arrays = np.reshape(ak.to_numpy(arrays), (len(arrays),))
        if key == 'weight_mc':
            arrays *= weights_factor(tag)
    return root_tuple, arrays


def weights_factor(jet_tag):
    #DIJET simulation (en ordre de dataset: JZ3, JZ4, JZ5, etc.)
    DSIDsDijet        = ['361023', '361024', '361025', '361026', '361027',
                         '361028', '361029', '361030', '361031', '361032']
    crossSecDijet     = [26454000000.00, 254630000.000, 4553500.0000, 257530.000000, 16215.0000000,
                                 625.03,        19.639,       1.1962,      0.042259,     0.0010367] # in fb
    filtEffDijet      = [3.2016E-04, 5.3138E-04, 9.2409E-04, 9.4242E-04, 3.9280E-04,
                         1.0176E-02, 1.2076E-02, 5.9087E-03, 2.6761E-03, 4.2592E-04]
    NeventsDijet      = [15879500. , 15925500. , 15993500. , 17834000. , 15983000. ,
                         15999000. , 13915500. , 13985000. , 15948000. , 15995600. ]
    #TTBar simulation
    DSIDsTtbar        = ['410284', '410285', '410286', '410287', '410288']
    crossSecTtbar     = [831760, 831760, 831760, 831760, 831760] # in fb
    filtEffTtbar      = [3.8853E-03, 1.5818E-03, 6.8677E-04, 4.2095E-04, 2.3943E-04]
    sumofweightsTtbar = [436767582., 136981616., 66933351. , 51910542. , 31469215.]
    if jet_tag in DSIDsDijet:
        tag_dict = dict(zip(DSIDsDijet, np.array(crossSecDijet)*np.array(filtEffDijet)/np.array(NeventsDijet)))
    if jet_tag in DSIDsTtbar:
        tag_dict = dict(zip(DSIDsTtbar, np.array(crossSecTtbar)*np.array(filtEffTtbar)/np.array(sumofweightsTtbar)))
    return tag_dict[jet_tag]


def final_jets(jets):
    start_time = time.time()
    jets = np.concatenate([jets[key][...,np.newaxis] for key in jets          ], axis=2)
    jets = np.concatenate([jets, np.zeros_like(jets[...,:1], dtype=np.float32)], axis=2)
    manager    = mp.Manager(); return_dict = manager.dict()
    idx_tuples = get_idx(len(jets), n_sets=mp.cpu_count())
    arguments  = [(jets[idx[0]:idx[1]], idx, return_dict) for idx in idx_tuples]
    processes  = [mp.Process(target=transform_jets, args=arg) for arg in arguments]
    for task in processes: task.start()
    for task in processes: task.join()
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return {key:np.concatenate([return_dict[idx][key] for idx in idx_tuples])
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
    return {'E':E, 'pt':pt, 'M':M}


def jet_pt_ordering(jet):
    pt_idx = np.argsort(jet[:,0])[::-1]
    for n in [0,1,2]: jet[:,n] = jet[:,n][pt_idx]
    return jet


def jet_Lorentz_4v(jet):
    for n in np.arange(len(jet)):
        if jet[n,0] != 0:
            v = TLorentzVector(0, 0, 0, 0)
            v.SetPtEtaPhiM(jet[n,0], jet[n,1], jet[n,2], jet[n,3])
            jet[n,:] = v.E(), v.Px(), v.Py(), v.Pz()
    return jet


def jet_processing(jet):
    #find the jet (eta, phi)
    center=jet.sum(axis=0)
    v_jet=TLorentzVector(center[1], center[2], center[3], center[0])
    #centering parameters
    phi=v_jet.Phi()
    bv = v_jet.BoostVector()
    bv.SetPerp(0)
    for n in np.arange(len(jet)):
        if jet[n,0] != 0:
            v = TLorentzVector(jet[n,1], jet[n,2], jet[n,3], jet[n,0])
            v.RotateZ(-phi)
            v.Boost(-bv)
            jet[n,:] = v[3], v[0], v[1], v[2] #(E,Px,Py,Pz)
    #rotating parameters
    weighted_phi=0
    weighted_eta=0
    for n in np.arange(len(jet)):
        if jet[n,0] != 0:
            v = TLorentzVector(jet[n,1], jet[n,2], jet[n,3], jet[n,0])
            r = np.sqrt(v.Phi()**2 + v.Eta()**2)
            if r != 0: #in case there is only one component. In fact these data points should generally be invalid.
                weighted_phi += v.Phi() * v.E()/r
                weighted_eta += v.Eta() * v.E()/r
    #alpha = np.arctan2(weighted_phi, weighted_eta) #approximately align at eta
    alpha = np.arctan2(weighted_eta, weighted_phi) #approximately align at phi
    for n in np.arange(len(jet)):
        if jet[n,0] != 0:
            v = TLorentzVector(jet[n,1], jet[n,2], jet[n,3], jet[n,0])
            #v.rotate_x(alpha) #approximately align at eta
            v.RotateX(-alpha) #approximately align at phi
            jet[n,:] = v[3], v[0], v[1], v[2] #(E,Px,Py,Pz)
    return jet


'''
def constituent_count(root_files):
    with mp.Pool() as pool:
        root_list = np.sum(list(root_files.values()))
        return np.max(pool.map(max_constituents, root_list))
def max_constituents(root_files):
    events = uproot.open(root_files)['nominal']['rljet_assoc_cluster_pt'].array(library='ak')
    return np.max([len(n[0]) for n in events])
'''
