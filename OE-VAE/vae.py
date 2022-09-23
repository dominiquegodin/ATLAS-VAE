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
from   plots     import apply_cuts, plot_results, plot_history


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--n_train'     , default = 1e6          , type = float         )
parser.add_argument( '--n_valid'     , default = 1e6          , type = float         )
parser.add_argument( '--n_OoD'       , default = 5e6          , type = float         )
parser.add_argument( '--n_sig'       , default = 1e6          , type = float         )
parser.add_argument( '--n_const'     , default = 20           , type = int           )
parser.add_argument( '--n_dims'      , default = 3            , type = int           )
parser.add_argument( '--batch_size'  , default = 1e4          , type = float         )
parser.add_argument( '--n_epochs'    , default = 100          , type = int           )
parser.add_argument( '--FC_layers'   , default = [80,40,20,10], type = int, nargs='+')
parser.add_argument( '--lr'          , default = 1e-3         , type = float         )
parser.add_argument( '--beta'        , default = 0            , type = float         )
parser.add_argument( '--lamb'        , default = 0            , type = float         )
parser.add_argument( '--margin'      , default = 0            , type = float         )
parser.add_argument( '--n_iter'      , default = 1            , type = int           )
parser.add_argument( '--OE_type'     , default = 'KLD'                               )
parser.add_argument( '--weight_type' , default = 'X-S'                               )
parser.add_argument( '--model_in'    , default = ''                                  )
parser.add_argument( '--model_out'   , default = 'model.h5'                          )
parser.add_argument( '--scaler_in'   , default = ''                                  )
parser.add_argument( '--scaler_out'  , default = 'scaler.pkl'                        )
parser.add_argument( '--hist_file'   , default = 'history.pkl'                       )
parser.add_argument( '--output_dir'  , default = 'outputs'                           )
parser.add_argument( '--scaler_type' , default = 'RobustScaler'                      )
parser.add_argument( '--plotting'    , default = 'OFF'                               )
parser.add_argument( '--apply_cut'   , default = 'OFF'                               )
parser.add_argument( '--cut_types'   , default = ['bkg_eff','gain'],        nargs='+')
parser.add_argument( '--slurm_id'    , default = 0            , type = int           )
args = parser.parse_args()
for key in ['n_train', 'n_valid', 'n_OoD', 'n_sig', 'batch_size']: vars(args)[key] = int(vars(args)[key])
# HYPER-PARAMETERS GRID SEARCH
#args.n_const = grid_search(n_const=[20, 40, 60, 80, 100])[args.slurm_id]
#args.output_dir += '/n_const'+str(int(args.n_const))
#args.beta, args.lamb = grid_search(beta=[0, 0.1, 1, 10], lamb=[0, 1, 10, 100])[args.slurm_id]
#args.output_dir += '/beta'+format(args.beta, '.1f')+'_lamb'+format(args.lamb,'.1f')
args.model_in  = args.output_dir+'/'+args.model_in  ; args.scaler_in  = args.output_dir+'/'+args.scaler_in
args.model_out = args.output_dir+'/'+args.model_out ; args.scaler_out = args.output_dir+'/'+args.scaler_out
args.hist_file = args.output_dir+'/'+args.hist_file ; args.output_dir = args.output_dir+'/'+'plots'
Path(args.output_dir).mkdir(parents=True, exist_ok=True)


# SAMPLES SELECTIONS
bkg_data, OoD_data, sig_data = 'qcd-Geneva', 'H-OoD', 'top-Geneva'
sample_size  = len(list(h5py.File(get_file(bkg_data),'r').values())[0])
args.n_train = [0                       , min(args.n_train, sample_size-args.n_valid)]
args.n_valid = [sample_size-args.n_valid, sample_size                                ]
print('\nPROGRAM ARGUMENTS:\n'+tabulate(vars(args).items(), tablefmt='psql'))
gen_cuts   = ['(sample["pt"] >=    0)'] #['(sample["pt"] >= 550) & (sample["pt"] <= 650)']
train_cuts = ['(sample["pt"] <= 3000)'] + gen_cuts
valid_cuts = ['(sample["pt"] <= 3000)'] + gen_cuts


# LOADIND PRE-TRAINED WEIGHTS AND/OR CONSTITUENTS SCALER
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #Suppressing Tensorflow warnings
seed  = None if (args.n_epochs > 0 or args.n_iter > 1) else 0
model = VariationalAutoEncoder(args.FC_layers, args.n_dims*args.n_const, seed=seed)
multithread = True ; scaler = None
if args.model_in != args.output_dir[0:args.output_dir.rfind('/')]+'/':
    if not os.path.isfile(args.model_in): sys.exit()
    print('\nLoading pre-trained weights from:', args.model_in,)
    sys.stdout = open(os.devnull, 'w')   #Stopping screen output
    sample = Batch_Generator(bkg_data, OoD_data, args.n_const, args.n_dims)
    train_model(model, sample, sample)
    sys.stdout = sys.__stdout__          #Resuming screen output
    model.load_weights(args.model_in)
    multithread = False
if args.scaler_type.lower() != 'none' and os.path.isfile(args.scaler_in):
    print('\nLoading scaler transform from:', args.scaler_in)
    scaler = pickle.load(open(args.scaler_in, 'rb'))


# MODEL TRAINING
if args.n_epochs > 0:
    if args.scaler_type.lower() != 'none' and scaler == None:
        train_sample = load_data(bkg_data, args.n_train, train_cuts, args.n_const, args.n_dims)
        scaler = fit_scaler(train_sample['constituents'], args.n_dims, args.scaler_out, args.scaler_type)
    bin_sizes = {'m':20,'pt':40} if args.weight_type.split('_')[0] in ['flat','OoD'] else {'m':10,'pt':20}
    train_sample = Batch_Generator(bkg_data, OoD_data, args.n_const, args.n_dims, args.n_train, args.n_OoD,
                                   args.weight_type, train_cuts, multithread, bin_sizes, scaler, args.output_dir)
    valid_sample = Batch_Generator(bkg_data, OoD_data, args.n_const, args.n_dims, args.n_valid, args.n_OoD,
                                   args.weight_type, train_cuts, multithread, bin_sizes, scaler)
    train_model(model, train_sample, valid_sample, args.OE_type, args.n_epochs, args.batch_size, args.beta,
                args.lamb, args.margin, args.lr, args.hist_file, args.model_in, args.model_out)
    model.load_weights(args.model_out)
if args. plotting == 'OFF' and args.apply_cut == 'OFF': sys.exit()


# MODEL PREDICTIONS ON VALIDATION SAMPLE
print('\n+'+36*'-'+'+\n+--- VALIDATION SAMPLE EVALUATION ---+\n+'+36*'-'+'+\n')
#DSIDs: 302321,310464,449929,450282,450283,450284
valid_sample = make_sample(bkg_data, sig_data, args.n_valid, args.n_sig, valid_cuts,
                           args.n_const, args.n_dims, DSIDs=None, adjust_weights=False)
y_true = np.where(valid_sample['JZW']==-1, 0, 1)
X_true = valid_sample['constituents'].copy()
if 'Geneva' in sig_data: valid_sample['weights'][y_true==0] /= 1e2 #Adjusting signal weights for Delphes samples
#from plots import sample_distributions
#sample_distributions(valid_sample,sig_data, args.output_dir, 'valid'); sys.exit()
if scaler is not None: X_true = apply_scaler(X_true, args.n_dims, scaler); print()
if args.n_iter > 1: print('\nEvaluating with', args.n_iter, 'iterations:')
X_pred = np.empty(X_true.shape+(args.n_iter,), dtype=np.float32)
for n in np.arange(args.n_iter): X_pred[...,n] = model.predict(X_true, batch_size=int(1e4), verbose=1)
X_pred = np.mean(X_pred, axis=2); print()
y_true, X_true, X_pred, valid_sample = filtering(y_true, X_true, X_pred, valid_sample)


# PLOTTING PERFORMANCE RESULTS
metric_list = ['Latent', 'MSE'] #+ ['MAE','X-S'] + ['JSD','EMD','KSD','KLD'] + ['Inputs']
loss_metric = 'Latent' if (args.OE_type=='KLD' and 'Latent' in metric_list) else 'MSE'
if args.plotting == 'ON':
    plot_results(y_true, X_true, X_pred, valid_sample, args.n_dims, model,
                 metric_list, loss_metric, sig_data, args.output_dir)
    if os.path.isfile(args.hist_file): plot_history(args.hist_file, args.output_dir)
if args.apply_cut == 'ON':
    apply_cuts(y_true, X_true, X_pred, valid_sample, args.n_dims, model,
               loss_metric, sig_data, args.cut_types, args.output_dir)
