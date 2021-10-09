# IMPORT PACKAGES AND FUNCTIONS
import numpy as np, os, sys, h5py, uproot
from sklearn    import utils
from argparse   import ArgumentParser
from root_utils import get_files, get_data, final_jets, count_constituents
from merging    import file_processing


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--tag', '--names-list', nargs='+', default = [0]          )
parser.add_argument( '--sample_type'                   , default = 'topo-dijet' )
parser.add_argument( '--merging'                       , default = 'OFF'        )
parser.add_argument( '--library'                       , default = 'np'         ) #('np':fast, 'ak':slow)
args = parser.parse_args()


# INPUT/OUTPUT PATHS
if args.sample_type in ['topo-dijet', 'topo-ttbar']:
    input_path = '/nvme1/atlas/godin/AD_data/topo-rootfiles'
if args.sample_type in ['UFO-dijet', 'UFO-ttbar', 'BSM']:
    input_path = '/lcg/storage19/atlas/nguyenn/Jacinthe/rootfiles'
output_path = '/nvme1/atlas/godin/AD_data'


# ROOT VARIABLES
scalars = ['rljet_m_calo'   , 'rljet_m_comb'     , 'rljet_pt_calo'   , 'rljet_pt_comb'   , 'rljet_ECF3'       ,
           'rljet_C2'       , 'rljet_D2'         , 'rljet_Tau1_wta'  , 'rljet_Tau2_wta'  , 'rljet_Tau3_wta'   ,
           'rljet_Tau32_wta', 'rljet_FoxWolfram2', 'rljet_PlanarFlow', 'rljet_Angularity', 'rljet_Aplanarity' ,
           'rljet_ZCut12'   , 'rljet_Split12'    , 'rljet_Split23'   , 'rljet_KtDR'      , 'rljet_Qw'         ,
           'rljet_eta'      , 'rljet_phi'                                                                     ]
jet_var = ['rljet_assoc_cluster_pt', 'rljet_assoc_cluster_eta', 'rljet_assoc_cluster_phi'                     ]
others  = ['weight_mc', 'weight_pileup', 'rljet_topTag_DNN19_qqb_score', 'rljet_n_constituents'               ]


# LUMINOSITY
if args.sample_type in ['topo-dijet', 'topo-ttbar']:
    luminosity = 36.07456
if args.sample_type in ['UFO-dijet', 'UFO-ttbar', 'BSM']:
    luminosity = 58.45010


# DSIDS
if args.sample_type == 'topo-dijet':
    """ DIJET simulation (for JZ3, JZ4, JZ5, etc.) """
    DSIDs    = ['361023', '361024', '361025', '361026', '361027',
                '361028', '361029', '361030', '361031', '361032']
    crossSec = [26454000000.00, 254630000.000, 4553500.0000, 257530.000000, 16215.0000000,
                625.04,        19.639,       1.1962,      0.042259,     0.0010367] #in fb
    filtEff  = [3.2012E-04, 5.3137E-04, 9.2395E-04, 9.4270E-04, 3.9280E-04,
                1.0166E-02, 1.2077E-02, 5.9083E-03, 2.6734E-03, 4.2592E-04,]
    Nevents  = [15362751  , 15925231  , 15993500  , 17834000  , 15983000   ,
                15999000  , 13915500  , 13985000  , 15948000  , 15995600   ]
    n_constituents = 100
if args.sample_type == 'UFO-dijet':
    """ DIJET simulation (for JZ3, JZ4, JZ5, etc.) """
    DSIDs    = ['364703', '364704', '364705', '364706', '364707',
                '364708', '364709', '364710', '364711', '364712']
    crossSec = [26450000000.00, 254610000.000, 4552900.0000, 257540.000000, 16215.0000000,
                625.06,        19.639,       1.1962,      0.042263,     0.0010367] # in fb
    filtEff  = [1.1658E-02, 1.3366E-02, 1.4526E-02, 9.4734E-03, 1.1097E-02,
                1.0156E-02, 1.2056E-02, 5.8933E-03, 2.6730E-03, 4.2889E-04]
    Nevents  = [258.536   , 8.67297    , 0.345287   , 0.0389311  , 0.00535663,
                0.00154999, 0.000271431, 3.20958e-05, 2.10202e-06, 9.86921e-06]
    n_constituents = 377
if args.sample_type == 'topo-ttbar':
    DSIDs       = ['410284', '410285', '410286', '410287', '410288']
    crossSec    = [7.2978E+05, 7.2976E+05, 7.2978E+05, 7.2975E+05, 7.2975E+05] #in fb
    filtEff     = [3.8208E-03, 1.5782E-03, 6.9112E-04, 4.1914E-04, 2.3803E-04]
    weights_sum = [3.17751e+08, 1.00548e+08, 4.96933e+07, 3.87139e+07, 2.32803e+07]
if args.sample_type == 'UFO-ttbar':
    DSIDs       = ['410284', '410285', '410286', '410287', '410288']
    crossSec    = [7.2978E+05, 7.2976E+05, 7.2978E+05, 7.2975E+05, 7.2975E+05] #in fb
    filtEff     = [3.8208E-03, 1.5782E-03, 6.9112E-04, 4.1914E-04, 2.3803E-04]
    weights_sum = [4.23372e+08, 1.78314e+08, 8.72442e+07, 8.33126e+07, 3.69924e+07]
if args.sample_type == 'BSM':
    DSIDs       = ['302321', '302326', '302331', '310464', '310465', '310466', '310467', '310468',
                   '310469', '310470', '310471', '310472', '310473', '310474', '310475', '310476',
                   '310477', '450279', '450280', '450281', '450282', '450283', '450284', '450291',
                   '450292', '450293', '450294', '450295', '450296', '449929', '449930', '503739']
    crossSec    = [2.7610E+02, 4.6380E+01, 1.1160E+01, 2.5712E-03, 2.8366E-04, 5.0358E-05, 1.1463E-05, 2.5735E-03,
                   2.8576E-04, 5.0138E-05, 1.1473E-05, 2.5757E-03, 2.8336E-04, 5.0392E-05, 1.1403E-05, 2.5715E-03,
                   2.8401E-04, 1.0342E+00, 6.1132E+00, 2.0469E+01, 1.0501E+00, 4.1859E+00, 1.1302E+00, 3.7231E-02,
                   2.1800E-01, 7.3190E-01, 3.3723E-02, 1.2120E-01, 2.8290E-02, 1.0211E+00, 1.0214E+00, 3.4485E+00] #in fb
    filtEff     = [1.0000E+00, 1.0000E+00, 1.0000E+00, 4.6361E-01, 7.7126E-01, 8.7641E-01, 9.2337E-01, 6.5735E-01,
                   8.5953E-01, 9.2481E-01, 9.4986E-01, 2.8195E-01, 6.5096E-01, 8.0945E-01, 8.7866E-01, 5.2363E-01,
                   8.0082E-01, 1.0000E+00, 1.0000E+00, 1.0000E+00, 1.0000E+00, 1.0000E+00, 1.0000E+00, 1.0000E+00,
                   1.0000E+00, 1.0000E+00, 1.0000E+00, 1.0000E+00, 1.0000E+00, 1.0000E+00, 1.0000E+00, 1.0000E+00]
    weights_sum = [59663. , 69940. , 59977. , 40000. , 40000. , 40000. , 40000. , 40000.,
                   40000. , 39998. , 40000. , 40000. , 40000. , 40000. , 40000. , 39999.,
                   40000. , 19325. , 19636. , 19924. , 19823. , 19962. , 19990. , 17729.,
                   18670. , 20216.7, 19431.4, 20355.3, 20336.5, 100998., 101026., 378.34]
if args.sample_type in ['topo-dijet', 'UFO-dijet']:
    ID_weights = dict(zip(DSIDs, np.array(crossSec)*np.array(filtEff)/np.array(Nevents)))
if args.sample_type in ['topo-ttbar', 'UFO-ttbar', 'BSM']:
    ID_weights = dict(zip(DSIDs, np.array(crossSec)*np.array(filtEff)/np.array(weights_sum)))


# OUTPUT DATA FILES
if args.sample_type in ['topo-dijet', 'UFO-dijet']:
    ID_list      = [ DSIDs[int(args.tag[0])] ]
    output_file  = args.sample_type + '_' + ID_list[0]+'.h5'
    output_path += '/' + args.sample_type
if args.sample_type in ['topo-ttbar', 'UFO-ttbar', 'BSM']:
    ID_list      = DSIDs
    output_file  = args.sample_type+'.h5'


# INPUT DATA PATHS
data_paths = sorted([path for path in os.listdir(input_path) if path.split('.')[2] in ID_weights])
root_files = get_files(input_path, data_paths)
root_list  = np.concatenate([root_files[n] for n in ID_list])
if 'n_constituents' not in locals(): n_constituents = count_constituents(root_files)


# MERGING AND MIXING DATA FILES
if args.merging == 'ON': file_processing(output_path, n_constituents)


# READING AND PROCESSING ROOT DATA
var_list  = scalars + jet_var + others
root_data = get_data(root_list, var_list, jet_var, n_constituents, args.sample_type, ID_weights, args.library)
if np.all([n in var_list for n in jet_var]):
    root_data.update(final_jets({key:root_data.pop(key) for key in jet_var}))
root_data['weights'] = luminosity * root_data.pop('weight_mc') * root_data.pop('weight_pileup')
#for key in root_data: print(format(key,'28s'), root_data[key].shape, root_data[key].dtype)


# WRITING OUTPUT FILES
if not os.path.isdir(output_path): os.mkdir(output_path)
data = h5py.File(output_path+'/'+output_file, 'w')
for key, val in root_data.items():
    if args.library != 'ak': val = val.astype(val[0].dtype)
    data.create_dataset(key, data=utils.shuffle(val,random_state=0), compression='lzf')
