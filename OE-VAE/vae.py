# IMPORT PACKAGES AND FUNCTIONS
import numpy           as np
import multiprocessing as mp
import os, sys, pickle, h5py
from   argparse  import ArgumentParser
from   pathlib   import Path
from   tabulate  import tabulate
from   models    import VariationalAutoEncoder, train_model
from   utils     import get_file, load_data, make_sample, Batch_Generator
from   utils     import fit_scaler, apply_scaler, filtering, grid_search
from   plots     import sample_distributions, plot_results, plot_history


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--n_train'      , default = 1e6          , type = float         )
parser.add_argument( '--n_valid'      , default = 1e6          , type = float         )
parser.add_argument( '--n_OoD'        , default = 10e6         , type = float         )
parser.add_argument( '--n_sig'        , default = 1e6          , type = float         )
parser.add_argument( '--n_const'      , default = 20           , type = int           )
parser.add_argument( '--n_dims'       , default = 3            , type = int           )
parser.add_argument( '--batch_size'   , default = 1e4          , type = float         )
parser.add_argument( '--n_epochs'     , default = 100          , type = int           )
parser.add_argument( '--FC_layers'    , default = [80,40,20,10], type = int, nargs='+')
parser.add_argument( '--lr'           , default = 1e-3         , type = float         )
parser.add_argument( '--beta'         , default = 0            , type = float         )
parser.add_argument( '--lamb'         , default = 0            , type = float         )
parser.add_argument( '--margin'       , default = 1            , type = float         )
parser.add_argument( '--n_iter'       , default = 1            , type = int           )
parser.add_argument( '--OE_type'      , default = 'KLD'                               )
parser.add_argument( '--weight_type'  , default = 'X-S'                               )
parser.add_argument( '--model_in'     , default = ''                                  )
parser.add_argument( '--model_out'    , default = 'model.h5'                          )
parser.add_argument( '--const_scaler_type', default = 'QuantileTransformer'           )
parser.add_argument( '--const_scaler_in'  , default = ''                              )
parser.add_argument( '--const_scaler_out' , default = ''                              )
parser.add_argument( '--HLV_scaler_type'  , default = 'RobustScaler'                  )
parser.add_argument( '--HLV_scaler_in'    , default = ''                              )
parser.add_argument( '--HLV_scaler_out'   , default = ''                              )
parser.add_argument( '--hist_file'    , default = 'history.pkl'                       )
parser.add_argument( '--output_dir'   , default = 'outputs'                           )
parser.add_argument( '--plotting'     , default = 'ON'                                )
parser.add_argument( '--apply_cuts'   , default = 'OFF'                               )
parser.add_argument( '--normal_losses', default = 'ON'                                )
parser.add_argument( '--decorrelation', default = 'OFF'                               )
parser.add_argument( '--slurm_id'     , default = 0            , type = int           )
parser.add_argument( '--constituents' , default = 'OFF'                               )
parser.add_argument( '--HLVs'         , default = 'ON'                                )
args = parser.parse_args()
for key in ['n_train', 'n_valid', 'n_OoD', 'n_sig', 'batch_size']: vars(args)[key] = int(vars(args)[key])
if args.const_scaler_out == '': args.const_scaler_out = 'const_' + args.const_scaler_type + '.pkl'
if args.HLV_scaler_out   == '': args.HLV_scaler_out   = 'HLV_'   + args.HLV_scaler_type   + '.pkl'
# HYPER-PARAMETERS GRID SEARCH
#args.n_const = grid_search(n_const=[20, 40, 60, 80, 100])[args.slurm_id]
#args.output_dir += '/n_const'+str(int(args.n_const))
#args.beta, args.lamb = grid_search(beta=[0, 0.1, 1, 10], lamb=[0, 1, 10, 100])[args.slurm_id]
#args.output_dir += '/beta'+format(args.beta, '.1f')+'_lamb'+format(args.lamb,'.1f')
args.model_in         = args.output_dir+'/'+args.model_in
args.model_out        = args.output_dir+'/'+args.model_out
args.const_scaler_in  = args.output_dir+'/'+args.const_scaler_in
args.const_scaler_out = args.output_dir+'/'+args.const_scaler_out
args.HLV_scaler_in    = args.output_dir+'/'+args.HLV_scaler_in
args.HLV_scaler_out   = args.output_dir+'/'+args.HLV_scaler_out
args.hist_file        = args.output_dir+'/'+args.hist_file
args.output_dir       = args.output_dir+'/'+'plots'
Path(args.output_dir).mkdir(parents=True, exist_ok=True)


# SAMPLES SELECTIONS
bkg_data, OoD_data, sig_data = 'QCD-Geneva', 'OoD-H', '2HDM-Geneva'
#bkg_data, OoD_data, sig_data = 'QCD-Geneva', 'OoD-H', 'top-Geneva'
HLVs = ['rljet_Tau1_wta', 'rljet_Tau2_wta', 'rljet_Tau3_wta', 'rljet_eta', 'rljet_phi',
        'rljet_ECF3'    , 'ECF2'          , 'd12'           , 'd23'      , 'pt'       , 'm']
if args.constituents == 'ON'  and args.HLVs == 'ON' : input_dim = args.n_dims*args.n_const + len(HLVs)
if args.constituents == 'ON'  and args.HLVs == 'OFF': input_dim = args.n_dims*args.n_const
if args.constituents == 'OFF' and args.HLVs == 'ON' : input_dim =                            len(HLVs)
sample_size  = len(list(h5py.File(get_file(bkg_data),'r').values())[0])
args.n_train = [0                       , min(args.n_train, sample_size-args.n_valid)]
args.n_valid = [sample_size-args.n_valid, sample_size                                ]
gen_cuts   =            ['(sample["m" ] >=   30)']
train_cuts = gen_cuts + ['(sample["pt"] <= 5000)']
valid_cuts = gen_cuts + ['(sample["pt"] <= 5000)']
for n in range(len(train_cuts)): vars(args)['train cuts ('+str(n+1)+')'] = train_cuts[n]
for n in range(len(valid_cuts)): vars(args)['valid cuts ('+str(n+1)+')'] = valid_cuts[n]
print('\nPROGRAM ARGUMENTS:\n'+tabulate(vars(args).items(), tablefmt='psql'))
#for key,val in h5py.File(get_file(sig_data),"r").items(): print(key, val.shape, val.dtype)


# LOADIND PRE-TRAINED WEIGHTS AND/OR CONSTITUENTS SCALER
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Suppressing Tensorflow infos and warnings
model = VariationalAutoEncoder(args.FC_layers, input_dim, seed=None if args.n_iter>1 else 0)
multithread = True ; const_scaler = None ; HLV_scaler = None
if args.model_in != args.output_dir[0:args.output_dir.rfind('/')]+'/':
    if not os.path.isfile(args.model_in): sys.exit()
    print('\nLoading pre-trained weights from: ' + args.model_in)
    sys.stdout = open(os.devnull, 'w')   #Stopping screen output
    sample = Batch_Generator(bkg_data, OoD_data, args.n_const, args.n_dims, n_bkg=[0,1],
                             constituents=args.constituents, HLVs=args.HLVs)
    train_model(model, sample, sample, args.OE_type)
    sys.stdout = sys.__stdout__          #Resuming screen output
    model.load_weights(args.model_in)
    multithread = False
if args.const_scaler_type.lower() != 'none' and os.path.isfile(args.const_scaler_in):
    print('\nLoading scaler/transformer from: ' + args.const_scaler_in)
    const_scaler = pickle.load(open(args.const_scaler_in, 'rb'))
if args.HLV_scaler_type.lower()   != 'none' and os.path.isfile(args.HLV_scaler_in):
    print('\nLoading scaler/transformer from: ' + args.HLV_scaler_in)
    HLV_scaler   = pickle.load(open(args.HLV_scaler_in,   'rb'))


# MODEL TRAINING
if args.n_epochs > 0:
    if (args.const_scaler_type.lower()!='none' and const_scaler==None)\
    or (args.HLV_scaler_type.lower()  !='none' and   HLV_scaler==None):
        print('\nLoading QCD training sample'.upper())
        n_jets = min(args.n_train[1], int(1e9*30/args.n_const/args.n_dims/4))
        train_sample = load_data(bkg_data, n_jets, train_cuts, args.n_const, args.n_dims, args.constituents, args.HLVs)
        if args.constituents == 'ON' and const_scaler == None:
            const_scaler = fit_scaler(train_sample['constituents'], args.n_dims,
                                      args.const_scaler_out, args.const_scaler_type)
        if args.HLVs == 'ON' and HLV_scaler == None:
            HLV_scaler   = fit_scaler(train_sample['HLVs']        , args.n_dims,
                                      args.HLV_scaler_out  , args.HLV_scaler_type)
    print('\nLoading outlier sample'.upper())
    OoD_sample = load_data(OoD_data, args.n_OoD, train_cuts, args.n_const, args.n_dims,
                           args.constituents, args.HLVs)
    if 'constituents' in OoD_sample:
        OoD_sample['constituents'] = apply_scaler(OoD_sample['constituents'], args.n_dims, const_scaler, 'OoD')
    if 'HLVs' in OoD_sample:
        OoD_sample['HLVs'        ] = apply_scaler(OoD_sample['HLVs'        ], args.n_dims,   HLV_scaler, 'OoD')
    bin_sizes = {'m':20,'pt':40} if args.weight_type.split('_')[0] in ['flat','OoD'] else {'m':10,'pt':20}
    train_sample = Batch_Generator(bkg_data, OoD_data, args.n_const, args.n_dims, args.n_train, OoD_sample,
                                   args.weight_type, train_cuts, multithread, args.constituents, args.HLVs,
                                   bin_sizes, HLV_scaler, const_scaler, args.output_dir)
    valid_sample = Batch_Generator(bkg_data, OoD_data, args.n_const, args.n_dims, args.n_valid, OoD_sample,
                                   args.weight_type, train_cuts, multithread, args.constituents, args.HLVs,
                                   bin_sizes, HLV_scaler, const_scaler)
    train_model(model, train_sample, valid_sample, args.OE_type, args.n_epochs, args.batch_size, args.beta,
                args.lamb, args.margin, args.lr, args.hist_file, args.model_in, args.model_out)
    model.load_weights(args.model_out)
if args. plotting == 'OFF' and args.apply_cut == 'OFF': sys.exit()


# MODEL PREDICTIONS ON VALIDATION SAMPLE
print('\n+'+36*'-'+'+\n+--- VALIDATION SAMPLE EVALUATION ---+\n+'+36*'-'+'+\n')
#DSIDs: 302321,310464,449929,450282,450283,450284
valid_sample = make_sample(bkg_data, sig_data, args.n_valid, args.n_sig, valid_cuts, args.n_const,
                           args.n_dims, args.constituents, args.HLVs, DSIDs=None, adjust_weights=False)
y_true = np.where(valid_sample['JZW']==-1, 0, 1)
if 'Geneva' in sig_data: valid_sample['weights'][y_true==0] /= 1e3 #Adjusting weights for Delphes samples
#sample_distributions(valid_sample, sig_data, args.output_dir, 'valid'); sys.exit()
if 'constituents' in valid_sample:
    valid_sample['constituents'] = apply_scaler(valid_sample['constituents'], args.n_dims, const_scaler)
if 'HLVs' in valid_sample:
    valid_sample['HLVs'        ] = apply_scaler(valid_sample['HLVs'        ], args.n_dims,   HLV_scaler)
if 'constituents' in valid_sample and 'HLVs' in valid_sample:
    X_true = np.hstack([valid_sample['constituents'], valid_sample['HLVs']])
elif 'constituents' in valid_sample:
    X_true = valid_sample['constituents']
elif 'HLVs' in valid_sample:
    X_true = valid_sample['HLVs']
if args.n_iter > 1: print('\nEvaluating with', args.n_iter, 'iterations:')
X_pred = np.empty(X_true.shape+(args.n_iter,), dtype=np.float32)
for n in np.arange(args.n_iter): X_pred[...,n] = model.predict(X_true, batch_size=int(1e4), verbose=1)
X_pred = np.mean(X_pred, axis=2); print()
y_true, X_true, X_pred, valid_sample = filtering(y_true, X_true, X_pred, valid_sample)


# PLOTTING PERFORMANCE RESULTS
if args.plotting == 'ON':
    metric_list = ['Latent','MAE','KLD','JSD'] #+ ['Inputs'] #+ ['X-S','JSD','EMD','KSD','KLD']
    loss_metric = 'MAE'
    if os.path.isfile(args.hist_file): plot_history(args.hist_file, args.output_dir)
    plot_results(y_true, X_true, X_pred, valid_sample, args.n_dims, model, metric_list, loss_metric,
                 sig_data, args.output_dir, args.apply_cuts, args.normal_losses, args.decorrelation)
