# IMPORT PACKAGES AND FUNCTIONS
import numpy      as np
import tensorflow as tf
import multiprocessing as mp
import os, sys, pickle, time, h5py
from   argparse  import ArgumentParser
from   itertools import accumulate
from   tabulate  import tabulate
from   sklearn   import utils
from   copy      import deepcopy
from   models    import VariationalAutoEncoder, train_model
from   utils     import load_data, make_sample, Batch_Generator, fit_scaler, apply_scaler
from   utils     import get_file, filtering, grid_search
from   plots     import sample_distributions, apply_cuts, plot_results, plot_history


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--n_train'       , default = 1e6          , type = float         )
parser.add_argument( '--n_valid'       , default = 1e6          , type = float         )
parser.add_argument( '--n_OoD'         , default = 2e6          , type = float         )
parser.add_argument( '--n_sig'         , default = 1e6          , type = float         )
parser.add_argument( '--n_const'       , default = 20           , type = int           )
parser.add_argument( '--n_dims'        , default = 3            , type = int           )
parser.add_argument( '--batch_size'    , default = 1e4          , type = float         )
parser.add_argument( '--n_epochs'      , default = 50           , type = int           )
parser.add_argument( '--FC_layers'     , default = [80,40,20,10], type = int, nargs='+')
parser.add_argument( '--lr'            , default = 1e-3         , type = float         )
parser.add_argument( '--beta'          , default = 0            , type = float         )
parser.add_argument( '--lamb'          , default = 0            , type = float         )
parser.add_argument( '--n_iter'        , default = 1            , type = int           )
parser.add_argument( '--OE_type'       , default = 'KLD'                               )
parser.add_argument( '--weight_type'   , default = 'X-S'                               )
parser.add_argument( '--output_dir'    , default = 'outputs'                           )
parser.add_argument( '--model_in'      , default = ''                                  )
parser.add_argument( '--model_out'     , default = 'model.h5'                          )
parser.add_argument( '--scaler_in'     , default = ''                                  )
parser.add_argument( '--scaler_out'    , default = 'scaler.pkl'                        )
parser.add_argument( '--scaler_type'   , default = 'RobustScaler'                      )
parser.add_argument( '--hist_file'     , default = 'history.pkl'                       )
parser.add_argument( '--plotting'      , default = 'OFF'                               )
parser.add_argument( '--apply_cut'     , default = 'OFF'                               )
parser.add_argument( '--cut_types'     , default = ['bkg_eff','gain']       , nargs='+')
parser.add_argument( '--slurm_id'      , default = 0            , type = int           )
args = parser.parse_args()
#args.n_const = grid_search(n_const=[20, 40, 60, 80, 100])[args.array_id]
#args.output_dir += '/n_const'+str(int(args.n_const))
#args.beta, args.lamb = grid_search(beta=[0, 0.1, 1, 10], lamb=[0, 1, 10, 100])[args.array_id]
#args.output_dir += '/beta'+str(int(args.beta))+'_lamb'+str(int(args.lamb))
for key in ['n_train', 'n_valid', 'n_OoD', 'n_sig', 'batch_size']: vars(args)[key] = int(vars(args)[key])


# SAMPLES DEFINITIONS
bkg_data, OoD_data, sig_data = 'qcd-Geneva', 'H-OoD', 'top-Geneva'
#bkg_data, OoD_data, sig_data = 'qcd-Geneva', 'H-OoD', '2HDM-Geneva'
#bkg_data, OoD_data, sig_data = 'qcd-UFO'   , 'H-OoD', 'top-UFO'
sample_size  = len(list(h5py.File(get_file(bkg_data),'r').values())[0])
args.n_train = [0                         , min(args.n_train, sample_size - args.n_valid)]
args.n_valid = [sample_size - args.n_valid, sample_size                                  ]
print('\nPROGRAM ARGUMENTS:\n'+tabulate(vars(args).items(), tablefmt='psql'))


# SAMPLES CUTS
gen_cuts = ['(sample["pt"] >=    0)'] + ['(sample["weights"] <= 200)']
#gen_cuts = ['(sample["m" ] <=  200) | (sample["m"] >= 400)'] + gen_cuts
bkg_cuts = ['(sample["pt"] <= 3000)'] + gen_cuts
OoD_cuts = ['(sample["pt"] <= 3000)'] + gen_cuts
sig_cuts = ['(sample["pt"] <= 3000)'] + gen_cuts


# METRICS LIST
metrics = ['MSE','Latent'] #+ ['MAE','X-S'] + ['JSD','EMD','KSD','KLD'] + ['Inputs']


# LOADIND PRE-TRAINED WEIGHTS AND/OR CONSTITUENTS SCALER
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #Suppressing Tensorflow warnings
seed  = None if (args.n_epochs > 0 or args.n_iter > 1) else 0
model = VariationalAutoEncoder(args.FC_layers, args.n_dims*args.n_const, seed=seed)
args.model_in   = args.output_dir+'/'+args.model_in
args.model_out  = args.output_dir+'/'+args.model_out
args.scaler_in  = args.output_dir+'/'+args.scaler_in
args.scaler_out = args.output_dir+'/'+args.scaler_out
args.hist_file  = args.output_dir+'/'+args.hist_file
if args.model_in != args.output_dir+'/':
    if not os.path.isfile(args.model_in): sys.exit()
    print('\nLoading pre-trained weights from:', args.model_in,)
    sys.stdout = open(os.devnull, 'w') #Suppressing screen output
    sample = Batch_Generator(bkg_data, OoD_data, n_const=args.n_const, n_dims=args.n_dims)
    train_model(model, sample, sample)
    sys.stdout = sys.__stdout__        #Resuming screen output
    model.load_weights(args.model_in)
if os.path.isfile(args.scaler_in):
    print('\nLoading scaler transform from:', args.scaler_in, '\n')
    scaler = pickle.load(open(args.scaler_in, 'rb'))
else:
    scaler = None
args.output_dir += '/plots'
for path in list(accumulate([folder+'/' for folder in args.output_dir.split('/')])):
    try: os.mkdir(path)
    except FileExistsError: pass


# MODEL TRAINING
if args.n_epochs > 0:
    if args.scaler_type.lower() != 'none' and scaler == None:
        train_sample = load_data(bkg_data, args.n_train, bkg_cuts, args.n_const, args.n_dims)
        scaler = fit_scaler(train_sample['constituents'], args.n_dims, args.scaler_out, args.scaler_type)
    bin_sizes = {'m':20,'pt':40} if args.weight_type.split('_')[0] in ['flat','OoD'] else {'m':10,'pt':20}
    train_sample = Batch_Generator(bkg_data, OoD_data, args.n_train, args.n_OoD, bkg_cuts, OoD_cuts,
                                   args.n_const, args.n_dims, args.weight_type, bin_sizes, scaler)
    plot_sample = train_sample[0]
    plot_sample = {key:np.concatenate([plot_sample[0][key], plot_sample[1][key]])
                   for key in ['m', 'pt', 'weights', 'JZW']}
    sample_distributions(plot_sample, OoD_data, args.output_dir, 'train', args.weight_type, bin_sizes)
    valid_sample = Batch_Generator(bkg_data, OoD_data, args.n_valid, args.n_OoD, bkg_cuts, OoD_cuts,
                                   args.n_const, args.n_dims, args.weight_type, bin_sizes, scaler)
    train_model(model, train_sample, valid_sample, args.OE_type, args.n_epochs, args.batch_size,
                args.beta, args.lamb, args.lr, args.hist_file, args.model_in, args.model_out)
    model.load_weights(args.model_out)


# MODEL PREDICTIONS ON VALIDATION DATA
if args. plotting == 'OFF' and args.apply_cut == 'OFF': sys.exit()
print('\n+'+36*'-'+'+\n+--- VALIDATION SAMPLE EVALUATION ---+\n+'+36*'-'+'+')
#DSIDs: 302321,310464,449929,450282,450283,450284
sample = make_sample(bkg_data, sig_data, args.n_valid, args.n_sig, bkg_cuts, sig_cuts,
                     args.n_const, args.n_dims, DSIDs=None, adjust_weights=True)
""" Defining labels """
y_true = np.where(sample['JZW']==-1, 0, 1)
""" Adjusting signal weights (Delphes samples)"""
if 'Geneva' in sig_data: sample['weights'][y_true==0] /= 1e3
#sample_distributions(sample,sig_data, args.output_dir, 'valid')
if scaler is not None: X_true = apply_scaler(sample['constituents'], args.n_dims, scaler)
else                 : X_true =              sample['constituents']
X_true = tf.cast(X_true, tf.float64)
if args.n_iter > 1: print('\nEvaluating with', args.n_iter, 'iterations:')
X_pred = np.empty(X_true.shape+(args.n_iter,), dtype=np.float32)
for n in np.arange(args.n_iter): X_pred[...,n] = model.predict(X_true, batch_size=int(1e4), verbose=1)
X_pred = np.mean(X_pred, axis=2); print()
y_true, X_true, X_pred, sample = filtering(y_true, X_true, X_pred, sample)


# BKG SUPPRESION AND MASS SCULPTING METRIC
main_metric = 'Latent' if (args.OE_type=='KLD' and 'Latent' in metrics) else 'MSE'


# APPLYING CUTS ON SAMPLES
if args.apply_cut == 'ON':
    apply_cuts(y_true, X_true, X_pred, sample, args.n_dims, model, main_metric,
               sig_data, args.cut_types, args.output_dir)


# PLOTTING PERFORMANCE RESULTS
if args.plotting == 'ON':
    plot_results(y_true, X_true, X_pred, sample, args.n_dims, model,
                 metrics, main_metric, sig_data, args.output_dir)
    if os.path.isfile(args.hist_file): plot_history(args.hist_file, args.output_dir)
