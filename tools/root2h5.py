# IMPORT PACKAGES AND FUNCTIONS
import numpy as np, os, sys, h5py, uproot
from sklearn    import utils
from argparse   import ArgumentParser
from root_utils import get_files, get_data, final_jets, count_constituents
from merging    import file_processing


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--tag'       , '--names-list', nargs='+', default = []    )
parser.add_argument( '--luminosity',  type = float            , default = 58.45 )
parser.add_argument( '--merging'                              , default = 'OFF' ) #(36.21:topo, 58.45:UFO))
parser.add_argument( '--library'                              , default = 'np'  ) #('np':fast, 'ak':slow)
args = parser.parse_args()


# INPUT/OUTPUT PATHS
#input_path  = '/nvme1/atlas/godin/AD_data/rootfiles'
input_path  = '/lcg/storage19/atlas/nguyenn/Jacinthe/rootfiles'
#if not os.path.isdir(input_path):
#    input_path = '/lcg/storage18/atlas/pilette/atlasdata/rootfiles'
output_path = '/nvme1/atlas/godin/AD_data'
dijet_label = 'Atlas_MC_dijet'
ttbar_label = 'Atlas_MC_ttbar'


# MERGING AND MIXING DATA FILES
if args.merging == 'ON':
    file_processing(output_path+'/'+dijet_label)


# ROOT VARIABLES
scalars = ['rljet_m_comb'    , 'rljet_pt_calo'   , 'rljet_ECF3'      , 'rljet_C2'       , 'rljet_D2'         ,
           'rljet_Tau1_wta'  , 'rljet_Tau2_wta'  , 'rljet_Tau3_wta'  , 'rljet_Tau32_wta', 'rljet_FoxWolfram2',
           'rljet_PlanarFlow', 'rljet_Angularity', 'rljet_Aplanarity', 'rljet_ZCut12'   , 'rljet_Split12'    ,
           'rljet_Split23'   , 'rljet_KtDR'      , 'rljet_Qw'        , 'rljet_eta'      , 'rljet_phi'        ]
jet_var = ['rljet_assoc_cluster_pt', 'rljet_assoc_cluster_eta', 'rljet_assoc_cluster_phi'                    ]
others  = ['weight_mc', 'weight_pileup', 'rljet_topTag_DNN19_qqb_score', 'rljet_n_constituents'              ]


# QCD AND TOP TAGS
#qcd_tags = ['361023', '361024', '361025', '361026', '361027', #for topo-clusters
#            '361028', '361029', '361030', '361031', '361032']
qcd_tags = ['364703', '364704', '364705', '364706', '364707', #for UFO jets
            '364708', '364709', '364710', '364711', '364712']
top_tags = ['410284', '410285', '410286', '410287', '410288']


# OUTPUT DATA FILES
if 'ttbar' in args.tag or int(args.tag[0]) >= len(qcd_tags):
    n_constituents = 196
    JZW = -1
    tag_list = top_tags
    output_file = ttbar_label+'.h5'
else:
    n_constituents = 100
    JZW = int(args.tag[0])
    tag_list = [qcd_tags[JZW]]
    output_file = dijet_label+'_'+qcd_tags[JZW]+'.h5'


# INPUT DATA PATHS
data_paths = sorted([path for path in os.listdir(input_path) if path.split('.')[2] in qcd_tags+top_tags])
root_files = get_files(input_path, data_paths)
#print(count_constituents(root_files))
root_list  = np.concatenate([root_files[tag] for tag in tag_list])
#for tag in tag_list:
#    for root_file in root_files[tag]: print(root_file)
#    print()
#for key in uproot.open(root_list[0]).keys(): print(key)


# READING AND PROCESSING ROOT DATA
var_list  = scalars + jet_var + others
root_data = get_data(root_list, var_list, jet_var, n_constituents, args.library)
if np.all([n in var_list for n in jet_var]):
    root_data.update(final_jets({key:root_data.pop(key) for key in jet_var}))
root_data['weights'] = args.luminosity*root_data.pop('weight_mc')*root_data.pop('weight_pileup')
#for key in root_data: print(format(key,'28s'), root_data[key].shape, root_data[key].dtype)


# WRITING OUTPUT FILES
if 'dijet' in output_file:
    output_path += '/'+dijet_label
if not os.path.isdir(output_path):
    os.mkdir(output_path)
data = h5py.File(output_path+'/'+output_file, 'w')
for key, val in root_data.items():
    if args.library != 'ak':
        val = val.astype(val[0].dtype)
    data.create_dataset(key, data=utils.shuffle(val,random_state=0), compression='lzf')
data.create_dataset('JZW', data=np.full_like(root_data['weights'], JZW, dtype=np.int8), compression='lzf')
