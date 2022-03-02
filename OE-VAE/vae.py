# IMPORT PACKAGES AND FUNCTIONS
import numpy      as np
import tensorflow as tf
import os, sys, pickle
from   argparse  import ArgumentParser
from   itertools import accumulate
from   tabulate  import tabulate
from   sklearn   import utils
from   copy      import deepcopy
from   models    import VariationalAutoEncoder, train_model
from   utils     import make_sample, make_datasets, fit_scaler, apply_scaler, upsampling
from   utils     import reweight_sample, apply_best_cut, separate_sample, bump_hunter
from   plots     import plot_distributions, plot_results, combined_plots


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--n_train'       , default = 1e6          , type = float         )
parser.add_argument( '--n_valid'       , default = 1e6          , type = float         )
parser.add_argument( '--n_test'        , default = 1e6          , type = float         )
parser.add_argument( '--n_OoD'         , default = 1e6          , type = float         )
parser.add_argument( '--n_sig'         , default = 1e6          , type = float         )
parser.add_argument( '--n_constituents', default = 20           , type = int           )
parser.add_argument( '--n_dims'        , default = 4            , type = int           )
parser.add_argument( '--batch_size'    , default = 5e3          , type = float         )
parser.add_argument( '--n_epochs'      , default = 50           , type = int           )
parser.add_argument( '--FC_layers'     , default = [80,40,20,10], type = int, nargs='+')
parser.add_argument( '--lr'            , default = 1e-3         , type = float         )
parser.add_argument( '--beta'          , default = 0            , type = float         )
parser.add_argument( '--lamb'          , default = 0            , type = float         )
parser.add_argument( '--n_iter'        , default = 1            , type = int           )
parser.add_argument( '--DSIDs'         , default = []           , type = int, nargs='+')
parser.add_argument( '--OE_type'       , default = 'KLD'                               )
parser.add_argument( '--weight_type'   , default = 'flat_pt'                           )
parser.add_argument( '--output_dir'    , default = 'outputs'                           )
parser.add_argument( '--model_in'      , default = ''                                  )
parser.add_argument( '--model_out'     , default = 'model.h5'                          )
parser.add_argument( '--scaler_in'     , default = ''                                  )
parser.add_argument( '--scaler_out'    , default = 'scaler.pkl'                        )
parser.add_argument( '--scaling'       , default = 'ON'                                )
parser.add_argument( '--plotting'      , default = 'OFF'                               )
parser.add_argument( '--apply_cut'     , default = 'OFF'                               )
parser.add_argument( '--bump_hunter'   , default = 'OFF'                               )
args = parser.parse_args()
for key in ['n_train', 'n_valid', 'n_test', 'n_OoD', 'n_sig', 'batch_size']:
    vars(args)[key] = int(vars(args)[key])
print('\nPROGRAM ARGUMENTS:\n'+tabulate(vars(args).items(), tablefmt='psql'))
#DSIDs: 302321,310464,449929,450282,450283,450284
#combined_plots(args.n_test, args.n_sig, args.output_dir+'/plots', plot_var='M')


# TRAINING AND TESTING SAMPLES
#bkg_data = 'qcd-UFO'
#OoD_data = 'W-OoD'
#sig_data = 'top-UFO'
bkg_data = 'qcd-Geneva'
OoD_data = 'H-OoD'
sig_data = 'top-Geneva'


# CUTS ON SIGNAL AND BACKGROUND SAMPLES
gen_cuts =            ['(sample["pt"] >= 0   )']
bkg_cuts = gen_cuts + ['(sample["pt"] <= 3000)']
OoD_cuts = gen_cuts + ['(sample["pt"] <= 3000)']
sig_cuts = gen_cuts + ['(sample["pt"] <= 3000)']


# SAMPLES RANGES
args.n_train = (0              , args.n_train                  )
args.n_valid = (args.n_train[1], args.n_train[1] + args.n_valid)
args.n_test  = (-args.n_test   , None                          )


# METRICS LIST
metrics = ['MSE', 'MAE'] + ['X-S'] + ['JSD', 'EMD', 'KSD', 'KLD'] + ['Inputs', 'Latent']


# LOADIND PRE-TRAINED WEIGHTS AND SCALER
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #Suppressing Tensorflow warnings
seed  = None if args.n_epochs > 0 else 0
model = VariationalAutoEncoder(args.FC_layers, args.n_dims*args.n_constituents, seed=seed)
args.model_in   = args.output_dir+'/'+args.model_in
args.model_out  = args.output_dir+'/'+args.model_out
args.scaler_in  = args.output_dir+'/'+args.scaler_in
args.scaler_out = args.output_dir+'/'+args.scaler_out
if args.model_in != args.output_dir+'/':
    if not os.path.isfile(args.model_in):
        sys.exit()
    print('\nLoading pre-trained weights from:', args.model_in)
    sys.stdout = open(os.devnull, 'w') #Suppressing screen output
    sample = make_sample(args.n_constituents, bkg_data, OoD_data, n_dims=args.n_dims)
    dataset = make_datasets(sample, sample)
    train_model(model, dataset.take(1), dataset.take(1))
    sys.stdout = sys.__stdout__        #Resuming screen output
    model.load_weights(args.model_in)
if args.scaling == 'ON' and os.path.isfile(args.scaler_in):
    print('\nLoading scaler transform from:', args.scaler_in)
    scaler = pickle.load(open(args.scaler_in, 'rb'))
args.output_dir += '/plots'
for path in list(accumulate([folder+'/' for folder in args.output_dir.split('/')])):
    try: os.mkdir(path)
    except FileExistsError: pass


# MODEL TRAINING
if args.n_epochs > 0:
    print('\nTRAINING SAMPLE:')
    train_sample = make_sample(args.n_constituents, bkg_data, OoD_data, args.n_train, args.n_OoD,
                               bkg_cuts, OoD_cuts, n_dims=args.n_dims)
    train_sample = {key:utils.shuffle(train_sample[key], random_state=0) for key in train_sample}
    plot_samples = [deepcopy(train_sample), train_sample] if 'flat' in args.weight_type else train_sample
    bin_sizes    = {'m':20,'pt':40}
    train_sample = reweight_sample(train_sample, bin_sizes, weight_type=args.weight_type)
    for var in ['m','pt']: plot_distributions(plot_samples, args.output_dir, bin_sizes, var,
                                              OoD_data, args.weight_type, file_name=var+'_train.png')
    #sys.exit()
    print('\nVALIDATION SAMPLE:')
    valid_sample = make_sample(args.n_constituents, bkg_data, OoD_data, args.n_valid, args.n_OoD,
                               bkg_cuts, OoD_cuts, n_dims=args.n_dims)
    valid_sample = reweight_sample(valid_sample, bin_sizes, weight_type=args.weight_type)
    if args.scaling == 'ON':
        if not os.path.isfile(args.scaler_in):
            JZW    = train_sample['JZW']
            scaler = fit_scaler(train_sample['constituents'][JZW!=-1], args.n_dims, args.scaler_out)
        print()
        train_sample['constituents'] = apply_scaler(train_sample['constituents'], args.n_dims, scaler)
        valid_sample['constituents'] = apply_scaler(valid_sample['constituents'], args.n_dims, scaler)
    train_sample, train_sample_OE = separate_sample(train_sample)
    valid_sample, valid_sample_OE = separate_sample(valid_sample)
    print('\nTraining   sample:', format(len(train_sample['weights']), '7.0f'), 'jets')
    print(  'Validation sample:', format(len(valid_sample['weights']), '7.0f'), 'jets')
    train_dataset = make_datasets(train_sample, train_sample_OE, args.batch_size)
    valid_dataset = make_datasets(valid_sample, valid_sample_OE, args.batch_size)
    train_model(model, train_dataset, valid_dataset, args.OE_type, args.n_epochs, args.beta, args.lamb, args.lr)
    print('\nSaving mode weights to:', args.model_out)
    model.save_weights(args.model_out)


# MODEL PREDICTIONS ON VALIDATION DATA
print('\n+'+30*'-'+'+\n+--- TEST SAMPLE EVALUATION ---+\n+'+30*'-'+'+')
sample = make_sample(args.n_constituents, bkg_data, sig_data, args.n_test, args.n_sig,
                     bkg_cuts, sig_cuts, n_dims=args.n_dims, adjust_weights=True, DSIDs=args.DSIDs)
sample = {key:utils.shuffle(sample[key], random_state=0) for key in sample}
""" Defining labels """
y_true = np.where(sample['JZW']==-1, 0, 1)
""" Adjusting signal weights (Delphes samples)"""
if sig_data == 'top-Geneva': sample['weights'][y_true==0] /= 10
#for var in ['m','pt']: plot_distributions(sample, args.output_dir, bin_sizes={'m':2.5,'pt':10},
#                                          plot_var=var, sig_tag=sig_data, file_name=var+'_valid.png')
if args.scaling=='ON': X_true = apply_scaler(sample['constituents'], args.n_dims, scaler)
else                 : X_true =              sample['constituents']
if args.n_iter > 1: print('\nEvaluating with', args.n_iter, 'iterations:')
X_pred = np.empty(X_true.shape+(args.n_iter,), dtype=np.float32)
for n in np.arange(args.n_iter): X_pred[...,n] = model.predict(X_true, batch_size=int(1e4), verbose=1)
X_pred = np.mean(X_pred, axis=2); print()


# BKG SUPPRESION AND MASS SCULPTING METRIC
wp_metric = 'Latent' if (args.OE_type=='KLD' and 'Latent' in metrics) else 'MSE'


# CUT ON RECONSTRUCTION LOSS
if args.apply_cut == 'ON' or args.bump_hunter == 'ON':
    for cut_type in ['gain', 'sigma']:
        cut_sample = apply_best_cut(y_true, X_true, X_pred, sample, args.n_dims, model, wp_metric, cut_type)
        if args.bump_hunter == 'ON': bump_hunter(cut_sample, args.output_dir, cut_type)
        plot_distributions([sample, cut_sample], args.output_dir, bin_sizes={'m':2.5,'pt':10},
                           plot_var='m', sig_tag=sig_data, file_name='bkg_supp-'+cut_type+'.png')


# PLOTTING RESULTS
if args.plotting == 'ON': plot_results(y_true, X_true, X_pred, sample, args.n_dims, model,
                                       metrics, wp_metric, sig_data, args.output_dir)
