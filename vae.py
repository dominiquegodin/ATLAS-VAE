# IMPORT PACKAGES AND FUNCTIONS
import numpy           as np
import tensorflow      as tf
import os, sys, pickle
from   argparse  import ArgumentParser
from   itertools import accumulate
from   tabulate  import tabulate
from   sklearn   import model_selection, utils
from   models    import create_model, callback
from   utils     import make_sample, fit_scaler, apply_scaling, reweight_sample, apply_best_cut
from   plots     import var_distributions, plot_results


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--n_train'       , default =     1e6, type = float         )
parser.add_argument( '--n_valid'       , default =     1e6, type = float         )
parser.add_argument( '--n_test'        , default =     1e6, type = float         )
parser.add_argument( '--n_W'           , default =       0, type = float         )
parser.add_argument( '--n_top'         , default =     1e6, type = float         )
parser.add_argument( '--n_constituents', default =      20, type = int           )
parser.add_argument( '--batch_size'    , default =     5e3, type = float         )
parser.add_argument( '--n_epochs'      , default =     500, type = int           )
parser.add_argument( '--FCN_neurons'   , default = [40,20], type = int, nargs='+')
parser.add_argument( '--latent_dim'    , default =      10, type = int           )
parser.add_argument( '--lr'            , default =    1e-3, type = float         )
parser.add_argument( '--beta'          , default =       1, type = float         )
parser.add_argument( '--patience'      , default =      15, type = int           )
parser.add_argument( '--n_iter'        , default =       1, type = int           )
parser.add_argument( '--n_gpus'        , default =       1, type = int           )
parser.add_argument( '--weight_type'   , default = None                          )
parser.add_argument( '--scaling'       , default = 'ON'                          )
parser.add_argument( '--apply_cut'     , default = 'OFF'                         )
parser.add_argument( '--plotting'      , default = 'OFF'                         )
parser.add_argument( '--output_dir'    , default = 'outputs'                     )
parser.add_argument( '--model_in'      , default = ''                            )
parser.add_argument( '--model_out'     , default = 'model.h5'                    )
parser.add_argument( '--scaler_in'     , default = ''                            )
parser.add_argument( '--scaler_out'    , default = 'scaler.pkl'                  )
args = parser.parse_args()
for key in ['n_train', 'n_valid', 'n_test', 'n_W', 'n_top', 'batch_size']:
    vars(args)[key] = int(vars(args)[key])


# PT CUTS ON SIGNAL AND BACKGROUND SAMPLES
sig_cuts = '(sample["pt"] >= 0) & (sample["pt"] <= 2000)'
bkg_cuts = '(sample["pt"] >= 0) & (sample["pt"] <= 6000) & (sample["weights"] <= 1e4)'


# SAMPLES SIZES
args.n_train = (0              , args.n_train                  )
args.n_valid = (args.n_train[1], args.n_train[1] + args.n_valid)
args.n_test  = (args.n_valid[1], args.n_valid[1] + args.n_test )


# LOADIND PRE-TRAINED WEIGHTS AND SCALER
n_gpus = min(args.n_gpus, len(tf.config.experimental.list_physical_devices('GPU')))
model  = create_model(4*args.n_constituents, args.FCN_neurons, args.latent_dim, args.lr, args.beta, n_gpus)
print('PROGRAM ARGUMENTS:\n'+tabulate(vars(args).items(), tablefmt='psql')+'\n')
args.model_in  = args.output_dir+'/'+args.model_in ; args.model_out  = args.output_dir+'/'+args.model_out
args.scaler_in = args.output_dir+'/'+args.scaler_in; args.scaler_out = args.output_dir+'/'+args.scaler_out
if os.path.isfile(args.model_in):
    print('Loading pre-trained weights from:', args.model_in)
    model.load_weights(args.model_in)
if args.scaling == 'ON' and os.path.isfile(args.scaler_in):
    print('Loading quantile transform  from:', args.scaler_in)
    scaler = pickle.load(open(args.scaler_in, 'rb'))
args.output_dir += '/plots'


# MODEL TRAINING
if args.n_epochs > 0:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #suppress Tensorflow warnings
    for path in list(accumulate([folder+'/' for folder in args.output_dir.split('/')])):
        try: os.mkdir(path)
        except FileExistsError: pass
    sig_bins, bkg_bins = 50, 100; log = args.weight_type!='flat_pt'
    print('TRAINING SAMPLE:')
    train_sample = make_sample(args.n_train, args.n_W, bkg_cuts, sig_cuts, bkg='qcd', sig='W')
    train_sample = {key:utils.shuffle(train_sample[key], random_state=0) for key in train_sample}
    print('VALIDATION SAMPLE:')
    valid_sample = make_sample(args.n_valid, args.n_W, bkg_cuts, sig_cuts, bkg='qcd', sig='W')
    train_sample = reweight_sample(train_sample, sig_bins, bkg_bins, args.weight_type)
    var_distributions(train_sample, args.output_dir, sig_bins, bkg_bins, var='pt', log=log)
    train_X, valid_X = train_sample['jets'], valid_sample['jets']
    if args.scaling == 'ON':
        if not os.path.isfile(args.scaler_in): scaler = fit_scaler(train_X, args.scaler_out)
        train_X, valid_X = apply_scaling(train_X, scaler), apply_scaling(valid_X, scaler); print()
    print('Train sample:'   , format(len(train_X), '8.0f')  , 'jets'    )
    print('Valid sample:'   , format(len(valid_X), '8.0f')  , 'jets\n'  )
    print('Using TensorFlow', tf.__version__, 'with', n_gpus, 'GPU(s)\n')
    callbacks = callback(args.model_out, args.patience, 'val_loss')
    training  = model.fit(train_X, train_X, validation_data=(valid_X,valid_X),
                          callbacks=callbacks, sample_weight=train_sample['weights'],
                          batch_size=args.batch_size, epochs=args.n_epochs )
    model.load_weights(args.model_out)


# MODEL PREDICTIONS ON VALIDATION DATA
print('\n+'+30*'-'+'+\n+--- TEST SAMPLE EVALUATION ---+\n+'+30*'-'+'+')
sample = make_sample(args.n_test, args.n_top, bkg_cuts, sig_cuts, bkg='qcd', sig='top')
y_true = np.int_(np.concatenate([np.zeros(np.sum(sample['JZW']==-1)), np.ones(np.sum(sample['JZW']>=0))]))
#var_distributions(sample, args.output_dir, sig_bins=200, bkg_bins=600, var='pt'); sys.exit()
if args.scaling == 'ON': X_true = apply_scaling(sample['jets'], scaler); print()
else                   : X_true =              sample['jets']
if args.n_iter > 1: print('Evaluating with', args.n_iter, 'iterations:')
X_pred = np.empty(X_true.shape+(args.n_iter,), dtype=np.float32)
for n in np.arange(args.n_iter):
    X_pred[...,n] = np.float32(model.predict(X_true, batch_size=max(1,n_gpus)*int(1e4), verbose=1))
X_pred = np.mean(X_pred, axis=2); print()


# CUT ON RECONSTRUCTION LOSS
if args.apply_cut == 'ON':
    cut_sample = apply_best_cut(y_true, X_true, X_pred, sample, metric='JSD')
    samples    = [sample, cut_sample]
    var_distributions(samples, args.output_dir, sig_bins=200, bkg_bins=600, var='M', normalize=False)


# PLOTTING RESULTS
if args.plotting == 'ON':
    if not os.path.isdir(args.output_dir): os.mkdir(args.output_dir)
    metrics = ['JSD'] + ['EMD', 'MSE', 'MAE', 'KLD', 'X-S'] + ['B-1', 'B-2']; #sample['weights'] = None
    plot_results(y_true, X_true, X_pred, sample, metrics, args.output_dir)
