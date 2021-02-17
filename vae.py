# COMMAND LINES FOR TRAINING AND VALIDATION
# python vae.py --n_train=1e6 --n_W=0 --n_valid=1e6 --n_top=1e6 --batch_size=5000 --n_epochs=500 --lr=1e-4
# COMMAND LINE FOR VALIDATION WITH PRE-TRAINED MODEL AND SCALER
# python vae.py --n_train=1e6 --n_valid=1e6 --n_top=1e6 --n_epochs=0 --model_in='model.h5' --scaler_in='scaler.pkl'


# IMPORT PACKAGES AND FUNCTIONS
import numpy           as np
import tensorflow      as tf
import multiprocessing as mp
import os, sys, pickle
from argparse  import ArgumentParser
from itertools import accumulate
from sklearn   import model_selection, utils
from utils     import make_sample, fit_scaler, apply_scaler, reweight_sample, loss_function, apply_cut
from models    import create_model, callback
from plots     import var_distributions, pt_reconstruction, loss_distributions, mass_correlation, ROC_curves


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--n_train'       , default =     1e6, type = float         )
parser.add_argument( '--n_valid'       , default =     1e6, type = float         )
parser.add_argument( '--n_W'           , default =       0, type = float         )
parser.add_argument( '--n_top'         , default =     1e6, type = float         )
parser.add_argument( '--n_constituents', default =      20, type = int           )
parser.add_argument( '--batch_size'    , default =     5e3, type = float         )
parser.add_argument( '--n_epochs'      , default =     500, type = int           )
parser.add_argument( '--FCN_neurons'   , default = [40,20], type = int, nargs='+')
parser.add_argument( '--latent_dim'    , default =      10, type = int           )
parser.add_argument( '--lr'            , default =    1e-4, type = float         )
parser.add_argument( '--beta'          , default =       1, type = float         )
parser.add_argument( '--patience'      , default =      15, type = int           )
parser.add_argument( '--apply_cut'     , default =       0, type = int           )
parser.add_argument( '--output_dir'    , default = 'outputs'                     )
parser.add_argument( '--model_in'      , default = ''                            )
parser.add_argument( '--model_out'     , default = 'model.h5'                    )
parser.add_argument( '--scaler_in'     , default = ''                            )
parser.add_argument( '--scaler_out'    , default = 'scaler.pkl'                  )
args = parser.parse_args()


# CUTS ON SIGNAL AND BACKGROUND SAMPLES
sig_cuts = '(sample["pt"] >= 0) & (sample["pt"] <= 2000)'
bkg_cuts = '(sample["pt"] >= 0) & (sample["pt"] <= 6000)'


# LOADIND PRE-TRAINED WEIGHTS AND SCALER
model = create_model(4*args.n_constituents, args.FCN_neurons, args.latent_dim, args.lr, args.beta)
args.model_in  = args.output_dir+'/'+args.model_in ; args.model_out  = args.output_dir+'/'+args.model_out
args.scaler_in = args.output_dir+'/'+args.scaler_in; args.scaler_out = args.output_dir+'/'+args.scaler_out
if os.path.isfile(args.model_in):
    print('Loading pre-trained weights from', args.model_in)
    model.load_weights(args.model_in)
if os.path.isfile(args.scaler_in):
    print('Loading scaler              from', args.scaler_in)
    scaler = pickle.load(open(args.scaler_in, 'rb'))


# LOADING TRAINING DATA AND TRAINING LOOP
if args.n_epochs > 0:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #suppress warning messages
    for path in list(accumulate([folder+'/' for folder in args.output_dir.split('/')])):
        try: os.mkdir(path)
        except FileExistsError: pass
    sample = make_sample(args.n_train, args.n_W, bkg_cuts, sig_cuts, bkg='qcd', sig='W')
    sample = reweight_sample(sample, sig_bins=50, bkg_bins=100, weights='flat_pt')
    #var_distributions(sample, args.output_dir, var='pt', sig_bins=50, bkg_bins=100); sys.exit()
    sample = {key:utils.shuffle(sample[key], random_state=0) for key in sample}
    train_X, valid_X  = model_selection.train_test_split(np.float32(sample['jets']), test_size=0.2, random_state=0)
    sample_weights, _ = model_selection.train_test_split(sample['weights']         , test_size=0.2, random_state=0)
    if not os.path.isfile(args.scaler_in): scaler = fit_scaler(train_X, args.scaler_out)
    train_X, valid_X = apply_scaler(train_X, scaler), apply_scaler(valid_X, scaler); print()
    callbacks = callback(args.model_out, args.patience, 'val_loss')
    training  = model.fit(train_X, train_X, validation_data=(valid_X,valid_X),
                          callbacks=callbacks, sample_weight=sample_weights,
                          batch_size=int(args.batch_size), epochs=args.n_epochs )
    model.load_weights(args.model_out); print()


# LOADING VALIDATION DATA AND MODEL PREDICTIONS
args.n_train = (args.n_train, args.n_train+args.n_valid)
sample = make_sample(args.n_train, args.n_top, bkg_cuts, sig_cuts, bkg='qcd', sig='top')
y_true = np.int_(np.concatenate([np.zeros(np.sum(sample['JZW']==-1)), np.ones(np.sum(sample['JZW']>=0))]))
#var_distributions(sample, args.output_dir, var='M', sig_bins=200, bkg_bins=600); sys.exit()
X_true = apply_scaler(sample['jets'], scaler); print()
X_pred = model.predict(X_true, batch_size=10000, verbose=1); print()


# APPLY CUT ON RECONSTRUCTION LOSS
if args.apply_cut:
    sample, y_true = apply_cut(y_true, X_true, X_pred, sample, metric='JSD')
    var_distributions(sample, args.output_dir, var='M', sig_bins=200, bkg_bins=600); sys.exit()


# PLOTTING RESULTS
metrics_list = ['JSD', 'MSE', 'KLD'] + ['EMD']; #sample['weights'] = None
processes = [mp.Process(target=pt_reconstruction, args=(X_true, X_pred, y_true, sample['weights'], args.output_dir))]
for metric in metrics_list:
    X_loss     = loss_function(X_true, X_pred, metric)
    processes += [mp.Process(target=loss_distributions, args=(y_true,X_loss,metric,sample['weights'],args.output_dir))]
    processes += [mp.Process(target=mass_correlation  , args=(y_true,X_loss,metric,sample['M']      ,args.output_dir))]
for job in processes: job.start()
for job in processes: job.join()
ROC_curves(X_true, X_pred, y_true, sample['weights'], args.output_dir, metrics_list, wps=[1,10])
