# IMPORT PACKAGES AND FUNCTIONS
import numpy      as np
import tensorflow as tf
import os, sys, pickle
from   argparse  import ArgumentParser
from   itertools import accumulate
from   tabulate  import tabulate
from   sklearn   import utils
from   copy      import deepcopy
from   models    import VariationalAutoEncoder, build_model
from   utils     import make_sample, make_datasets, fit_scaler, apply_scaler, upsampling
from   utils     import reweight_sample, apply_best_cut, separate_sample, bump_hunter
from   plots     import plot_distributions, plot_results, combined_plots


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--n_train'       , default = 1e6          , type = float         )
parser.add_argument( '--n_valid'       , default = 1e6          , type = float         )
parser.add_argument( '--n_test'        , default = 1e6          , type = float         )
parser.add_argument( '--n_W'           , default = 1e6          , type = float         )
parser.add_argument( '--n_top'         , default = 1e6          , type = float         )
parser.add_argument( '--n_constituents', default = 20           , type = int           )
parser.add_argument( '--n_dims'        , default = 4            , type = int           )
parser.add_argument( '--batch_size'    , default = 5e3          , type = float         )
parser.add_argument( '--n_epochs'      , default = 50           , type = int           )
parser.add_argument( '--FC_layers'     , default = [80,40,20,10], type = int, nargs='+')
parser.add_argument( '--lr'            , default = 1e-3         , type = float         )
parser.add_argument( '--beta'          , default = 0            , type = float         )
parser.add_argument( '--lamb'          , default = 0            , type = float         )
parser.add_argument( '--n_iter'        , default = 1            , type = int           )
parser.add_argument( '--OE_type'       , default = 'KLD'                               )
parser.add_argument( '--weight_type'   , default = 'flat_pt'                           )
parser.add_argument( '--plotting'      , default = 'OFF'                               )
parser.add_argument( '--apply_cut'     , default = 'OFF'                               )
parser.add_argument( '--output_dir'    , default = 'outputs'                           )
parser.add_argument( '--model_in'      , default = ''                                  )
parser.add_argument( '--model_out'     , default = 'model.h5'                          )
parser.add_argument( '--scaling'       , default = 'ON'                                )
parser.add_argument( '--scaler_in'     , default = ''                                  )
parser.add_argument( '--scaler_out'    , default = 'scaler.pkl'                        )
parser.add_argument( '--bump_hunter'   , default = 'OFF'                               )
args = parser.parse_args()
for key in ['n_train', 'n_valid', 'n_test', 'n_W', 'n_top', 'batch_size']:
    vars(args)[key] = int(vars(args)[key])
#combined_plots(args.n_test, args.n_top, args.output_dir+'/plots', plot_var='M')


'''
import h5py
data_path = '/opt/tmp/godin/AD_data'
#file_name = 'Atlas_topo-dijet.h5'
#file_name = 'Atlas_topo-ttbar.h5'
#file_name = 'Atlas_UFO-dijet.h5'
file_name = 'Atlas_UFO-ttbar.h5'
#file_name = 'Atlas_BSM.h5'
data      = h5py.File(data_path+'/'+file_name,"r")
for key in data: print( key, data[key].shape, data[key].dtype )
print( np.sum(data['weights']) )
#print( np.min(data['JZW']), np.max(data['JZW']) )
#print(data['DSID'][:] )
sys.exit()
'''


# METRICS LIST
metrics = ['MSE', 'MAE'] + ['X-S'] + ['JSD', 'EMD', 'KSD', 'KLD'] + ['Inputs', 'Latent']


# CUTS ON SIGNAL AND BACKGROUND SAMPLES
cuts  = ['(sample["pt"] >= 0)', '(sample["weights"] <= 200)']
sig_cuts = cuts + ['(sample["pt"] <= 2000)']
bkg_cuts = cuts + ['(sample["pt"] <= 6000)']


# SAMPLES SIZES
args.n_train = (0              , args.n_train                  )
args.n_valid = (args.n_train[1], args.n_train[1] + args.n_valid)
args.n_test  = (args.n_valid[1], args.n_valid[1] + args.n_test )


# LOADIND PRE-TRAINED WEIGHTS AND SCALER
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #Suppressing Tensorflow warnings
seed  = None if args.n_epochs > 0 else 0
model = VariationalAutoEncoder(args.n_dims*args.n_constituents, args.FC_layers, seed=seed)
print('\nPROGRAM ARGUMENTS:\n'+tabulate(vars(args).items(), tablefmt='psql'))
args.model_in   = args.output_dir+'/'+args.model_in
args.model_out  = args.output_dir+'/'+args.model_out
args.scaler_in  = args.output_dir+'/'+args.scaler_in
args.scaler_out = args.output_dir+'/'+args.scaler_out
if args.model_in != args.output_dir+'/':
    if not os.path.isfile(args.model_in):
        sys.exit()
    print('\nLoading pre-trained weights from:', args.model_in)
    sys.stdout = open(os.devnull, 'w') #Suppressing screen output
    sample = make_sample(args.n_dims, args.n_constituents, 'qcd-UFO', 'W', bkg_idx=1, sig_idx=1)
    dataset = make_datasets(sample, sample)
    build_model(model, dataset.take(1), dataset.take(1))
    sys.stdout = sys.__stdout__        #Resuming screen output
    model.load_weights(args.model_in)
if args.scaling == 'ON' and os.path.isfile(args.scaler_in):
    print('\nLoading scaler transform  from:', args.scaler_in)
    scaler = pickle.load(open(args.scaler_in, 'rb'))
args.output_dir += '/plots'
for path in list(accumulate([folder+'/' for folder in args.output_dir.split('/')])):
    try: os.mkdir(path)
    except FileExistsError: pass


# MODEL TRAINING
if args.n_epochs > 0:
    print('\nTRAINING SAMPLE:')
    train_sample = make_sample(args.n_dims, args.n_constituents, 'qcd-UFO', 'W', args.n_train, args.n_W,
                               bkg_cuts, sig_cuts, adjust_weights=False)
    train_sample = {key:utils.shuffle(train_sample[key], random_state=0) for key in train_sample}
    plot_samples = [deepcopy(train_sample), train_sample] if args.weight_type=='flat_pt' else train_sample
    sig_bins, bkg_bins = 100, 100
    train_sample = reweight_sample(train_sample, sig_bins, bkg_bins, weight_type=args.weight_type)
    plot_var = 'M' if args.weight_type=='M' else 'pt'
    plot_distributions(plot_samples, args.output_dir, plot_var, sig_bins, bkg_bins, sig_tag='W')#; sys.exit()
    print('\nVALIDATION SAMPLE:')
    valid_sample = make_sample(args.n_dims, args.n_constituents, 'qcd-UFO', 'W', args.n_valid, args.n_W,
                               bkg_cuts, sig_cuts, adjust_weights=False)
    valid_sample = reweight_sample(valid_sample, sig_bins=100, bkg_bins=100, weight_type=args.weight_type)
    if args.scaling == 'ON':
        if not os.path.isfile(args.scaler_in):
            JZW    = train_sample['JZW']
            scaler = fit_scaler(train_sample['constituents'][JZW!=-1], args.n_dims, args.scaler_out)
        train_sample['constituents'] = apply_scaler(train_sample['constituents'], args.n_dims, scaler)
        valid_sample['constituents'] = apply_scaler(valid_sample['constituents'], args.n_dims, scaler)
    train_sample, train_sample_OE = separate_sample(train_sample)
    valid_sample, valid_sample_OE = separate_sample(valid_sample)
    print('\nTraining     sample:', format(len(train_sample['weights']), '7.0f'), 'jets')
    print(  'Valididation sample:', format(len(valid_sample['weights']), '7.0f'), 'jets')
    train_dataset = make_datasets(train_sample, train_sample_OE, args.batch_size)
    valid_dataset = make_datasets(valid_sample, valid_sample_OE, args.batch_size)
    build_model(model, train_dataset, valid_dataset, args.OE_type, args.n_epochs, args.beta, args.lamb, args.lr)
    print('\nSaving mode weights to:', args.model_out)
    model.save_weights(args.model_out)


# MODEL PREDICTIONS ON VALIDATION DATA
print('\n+'+30*'-'+'+\n+--- TEST SAMPLE EVALUATION ---+\n+'+30*'-'+'+')
#dsids = ['302321', '302326', '302331']
sample = make_sample(args.n_dims, args.n_constituents, 'qcd-UFO', 'BSM', args.n_test, args.n_top,
                     bkg_cuts, sig_cuts, dsids=None, adjust_weights=True)
sample = {key:utils.shuffle(sample[key], random_state=0) for key in sample}
y_true = np.where(sample['JZW']==-1, 0, 1)
#plot_distributions(sample, args.output_dir, plot_var='pt', sig_bins=200, bkg_bins=600, sig_tag='top'); sys.exit()
if args.scaling=='ON': X_true = apply_scaler(sample['constituents'], args.n_dims, scaler)
else                 : X_true = sample['constituents']
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
        samples = [sample, cut_sample]
        plot_distributions(samples, args.output_dir, sig_bins=200, bkg_bins=200, plot_var='M', sig_tag='top',
                           file_name='bkg_suppr-'+cut_type+'.png')


# PLOTTING RESULTS
if args.plotting == 'ON':
    plot_results(y_true, X_true, X_pred, sample, args.n_dims, model, metrics, wp_metric, args.output_dir)
