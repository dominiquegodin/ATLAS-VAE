import numpy           as np
import multiprocessing as mp
import awkward         as ak
import os, sys, time, h5py, itertools, uproot
from functools import partial
try: from ROOT import TLorentzVector
except ModuleNotFoundError: pass


def get_files(main_path, paths):
    return {path.split('.')[2]: sorted([main_path+'/'+path+'/'+root_file+':nominal'
            for root_file in os.listdir(main_path+'/'+path)]) for path in paths}


def get_data(root_list, var_list, jet_var, n_constituents, sample_type, library):
    start_time = time.time()
    with mp.Pool() as pool:
        root_tuples = list(itertools.product(root_list, var_list))
        func_args   = (jet_var, n_constituents, sample_type, library)
        arrays_list = pool.map(partial(root_conversion, *func_args), root_tuples)
    print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
    return {key:np.concatenate([n[1] for n in arrays_list if key in n[0]]) for key in var_list}


def root_conversion(jet_var, n_constituents, sample_type, library, root_tuple):
    root_file, key = root_tuple
    tag = root_file.split('.')[2]
    arrays = uproot.open(root_file)[key].array(library=library)
    if key in jet_var:
        if library == 'ak':
            arrays = [ak.to_numpy(n[0]) for n in arrays]
        else:
            arrays = [            n[0]  for n in arrays]
        arrays = [np.pad(n, (0,max(n_constituents-len(n),0)),'constant')[:n_constituents] for n in arrays]
        arrays = np.float16(np.vstack(arrays)/1000. if key=='rljet_assoc_cluster_pt' else np.vstack(arrays))
        print(format(key,'23s'), 'converted from', '...'+tag+'...'+root_file.split('._')[-1])
    else:
        if library == 'ak':
            arrays = ak.to_numpy(arrays)
        arrays = np.reshape(arrays, (len(arrays),))
        if key == 'weight_mc':
            arrays *= weights_factor(tag, sample_type)
    return root_tuple, arrays


def weights_factor(jet_tag, sample_type):
    if sample_type == 'UFO':
        """ DIJET simulation (en ordre de dataset: JZ3, JZ4, JZ5, etc.) """
        DSIDsDijet    = ['364703', '364704', '364705', '364706', '364707',
                         '364708', '364709', '364710', '364711', '364712']
        crossSecDijet = [26450000000.00, 254610000.000, 4552900.0000, 257540.000000, 16215.0000000,
                                 625.06,        19.639,       1.1962,      0.042263,     0.0010367] # in fb
        filtEffDijet  = [1.1658E-02, 1.3366E-02, 1.4526E-02, 9.4734E-03, 1.1097E-02,
                         1.0156E-02, 1.2056E-02, 5.8933E-03, 2.6730E-03, 4.2889E-04]
        NeventsDijet  = [258.536   , 8.67297    , 0.345287   , 0.0389311  , 0.00535663,
                         0.00154999, 0.000271431, 3.20958e-05, 2.10202e-06, 9.86921e-06]
        """ TTBar simulation """
        DSIDsTtbar         = ['410284', '410285', '410286', '410287', '410288']
        crossSecTtbar      = [7.2978E+05, 7.2976E+05, 7.2978E+05, 7.2975E+05, 7.2975E+05] #in fb
        filtEffTtbar       = [3.8208E-03, 1.5782E-03, 6.9112E-04, 4.1914E-04, 2.3803E-04]
        sumofweightsTtbar  = [4.23372e+08, 1.78314e+08, 8.72442e+07, 8.33126e+07, 3.69924e+07]
        """ New BSM simulation """
        DSIDsBSM           = ['302321', '302326', '302331', '310464', '310465',
                              '310466', '310467', '310468', '310469', '310470',
                              '310471', '310472', '310473', '310474', '310475',
                              '310476', '310477', '450279', '450280', '450281',
                              '450282', '450283', '450284', '450291', '450292',
                              '450293', '450294', '450295', '450296', '449929',
                              '449930', '503739']
        crossSecTtbar      = [2.7610E+02, 4.6380E+01, 1.1160E+01, 2.5712E-03, 2.8366E-04,
                              5.0358E-05, 1.1463E-05, 2.5735E-03, 2.8576E-04, 5.0138E-05,
                              1.1473E-05, 2.5757E-03, 2.8336E-04, 5.0392E-05, 1.1403E-05,
                              2.5715E-03, 2.8401E-04, 1.0342E+00, 6.1132E+00, 2.0469E+01,
                              1.0501E+00, 4.1859E+00, 1.1302E+00, 3.7231E-02, 2.1800E-01,
                              7.3190E-01, 3.3723E-02, 1.2120E-01, 2.8290E-02, 1.0211E+00,
                              1.0214E+00, 3.4485E+00] #in fb
        filtEffTtbar       = [1.0000E+00, 1.0000E+00, 1.0000E+00, 4.6361E-01, 7.7126E-01,
                              8.7641E-01, 9.2337E-01, 6.5735E-01, 8.5953E-01, 9.2481E-01,
                              9.4986E-01, 2.8195E-01, 6.5096E-01, 8.0945E-01, 8.7866E-01,
                              5.2363E-01, 8.0082E-01, 1.0000E+00, 1.0000E+00, 1.0000E+00,
                              1.0000E+00, 1.0000E+00, 1.0000E+00, 1.0000E+00, 1.0000E+00,
                              1.0000E+00, 1.0000E+00, 1.0000E+00, 1.0000E+00, 1.0000E+00,
                              1.0000E+00, 1.0000E+00,]
        sumofweightsTtbar  = [59663. , 69940. , 59977. , 40000. , 40000. ,
                              40000. , 40000. , 40000. , 40000. , 39998. ,
                              40000. , 40000. , 40000. , 40000. , 40000. ,
                              39999. , 40000. , 19325. , 19636. , 19924. ,
                              19823. , 19962. , 19990. , 17729. , 18670. ,
                              20216.7, 19431.4, 20355.3, 20336.5, 100998. ,
                              101026. , 378.34]
    else:
        """ DIJET simulation (en ordre de dataset: JZ3, JZ4, JZ5, etc.) """
        DSIDsDijet    = ['361023', '361024', '361025', '361026', '361027',
                         '361028', '361029', '361030', '361031', '361032']
        crossSecDijet = [26454000000.00, 254630000.000, 4553500.0000, 257530.000000, 16215.0000000,
                         625.04,        19.639,       1.1962,      0.042259,     0.0010367] #in fb
        filtEffDijet  = [3.2012E-04, 5.3137E-04, 9.2395E-04, 9.4270E-04, 3.9280E-04, 
                         1.0166E-02, 1.2077E-02, 5.9083E-03, 2.6734E-03, 4.2592E-04,]
        NeventsDijet  = [15362751. , 15925231. , 15993500. , 17834000. , 15983000. ,
                         15999000. , 13915500. , 13985000. , 15948000. , 15995600. ]
        """ TTBar simulation """
        DSIDsTtbar        = ['410284', '410285', '410286', '410287', '410288']
        crossSecTtbar     = [7.2978E+05, 7.2976E+05, 7.2978E+05, 7.2975E+05, 7.2975E+05] #in fb
        filtEffTtbar      = [3.8208E-03, 1.5782E-03, 6.9112E-04, 4.1914E-04, 2.3803E-04]
        sumofweightsTtbar = [3.17751e+08, 1.00548e+08, 4.96933e+07, 3.87139e+07, 2.32803e+07]
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


def count_constituents(root_files):
    with mp.Pool() as pool:
        root_list = np.sum(list(root_files.values()))
        print('PROCESSED FILES:')
        for root_file in root_list:
            print(root_file)
        return np.max(pool.map(max_constituents, root_list))
def max_constituents(root_file):
    #events = uproot.open(root_file)['rljet_assoc_cluster_pt'].array(library='ak')
    #return np.max([len(n[0]) for n in events])
    events = uproot.open(root_file)['rljet_n_constituents'].array(library='np')
    return np.max([n for n in events])
