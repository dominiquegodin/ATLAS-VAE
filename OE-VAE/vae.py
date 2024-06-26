# IMPORT PACKAGES AND FUNCTIONS
import numpy      as np
import tensorflow as tf
import os, sys, pickle
from   argparse  import ArgumentParser
from   itertools import accumulate
from   tabulate  import tabulate
from   sklearn   import utils
from   models    import VariationalAutoEncoder, build_model
from   utils     import make_sample, make_datasets, fit_scaler, scaling
from   utils     import reweight_sample, apply_best_cut, bump_hunter
from   plots     import var_distributions, plot_results


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--n_train'       , default =        1e6, type = float         )
parser.add_argument( '--n_valid'       , default =        1e6, type = float         )
parser.add_argument( '--n_test'        , default =        1e6, type = float         )
parser.add_argument( '--n_W'           , default =        1e6, type = float         )
parser.add_argument( '--n_top'         , default =        1e6, type = float         )
parser.add_argument( '--n_constituents', default =         20, type = int           )
parser.add_argument( '--n_dims'        , default =          3, type = int           )
parser.add_argument( '--batch_size'    , default =        5e3, type = float         )
parser.add_argument( '--n_epochs'      , default =        500, type = int           )
parser.add_argument( '--FC_layers'     , default = [40,20,10], type = int, nargs='+')
#parser.add_argument( '--FC_layers'     , default = [40,10], type = int, nargs='+')
parser.add_argument( '--lr'            , default =       1e-3, type = float         )
parser.add_argument( '--beta'          , default =          0, type = float         )
parser.add_argument( '--lamb'          , default =          0, type = float         )
parser.add_argument( '--n_iter'        , default =          1, type = int           )
parser.add_argument( '--weight_type'   , default = 'flat_pt'                        )
parser.add_argument( '--plotting'      , default = 'OFF'                            )
parser.add_argument( '--apply_cut'     , default = 'OFF'                            )
parser.add_argument( '--output_dir'    , default = 'outputs'                        )
parser.add_argument( '--model_in'      , default = ''                               )
parser.add_argument( '--model_out'     , default = 'model.h5'                       )
parser.add_argument( '--scaling'       , default = 'ON'                             )
parser.add_argument( '--scaler_in'     , default = ''                               )
parser.add_argument( '--scaler_out'    , default = 'scaler.pkl'                     )
parser.add_argument( '--bump_hunter'   , default = 'OFF'                            )
args = parser.parse_args()
for key in ['n_train', 'n_valid', 'n_test', 'n_W', 'n_top', 'batch_size']:
    vars(args)[key] = int(vars(args)[key])


# METRICS LIST
metrics = ['MSE', 'MAE'] + ['X-S', 'JSD'] + ['EMD', 'KSD', 'KLD'] + ['Inputs']


# PT CUTS ON SIGNAL AND BACKGROUND SAMPLES
cuts  = ['(sample["pt"] >= 0)', '(sample["weights"] <= 200)']
#cuts += ['(sample["M"] >= 150)', '(sample["M"] <= 190)']
sig_cuts, bkg_cuts = cuts+['(sample["pt"] <= 2000)'], cuts+['(sample["pt"] <= 6000)']


# SAMPLES SIZES
args.n_train = (0              , args.n_train                  )
args.n_valid = (args.n_train[1], args.n_train[1] + args.n_valid)
args.n_test  = (args.n_valid[1], args.n_valid[1] + args.n_test )


# LOADIND PRE-TRAINED WEIGHTS AND SCALER
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #suppressing Tensorflow warnings
seed  = None if args.n_epochs > 0 else 0
model = VariationalAutoEncoder(args.n_dims*args.n_constituents, args.FC_layers, seed=seed)
print('\nPROGRAM ARGUMENTS:\n'+tabulate(vars(args).items(), tablefmt='psql'))
args.model_in    = args.output_dir+'/'+args.model_in
args.model_out   = args.output_dir+'/'+args.model_out
args.scaler_in   = args.output_dir+'/'+args.scaler_in
args.scaler_out  = args.output_dir+'/'+args.scaler_out
if os.path.isfile(args.model_in):
    print('\nLoading pre-trained weights from:', args.model_in)
    sys.stdout = open(os.devnull, 'w')
    sample = make_sample(args.n_dims, args.n_constituents, 'qcd', 'W', bkg_idx=1, sig_idx=1)
    train_dataset, valid_dataset = make_datasets(sample, sample, sample)
    build_model(model, train_dataset.take(1), valid_dataset.take(1))
    sys.stdout = sys.__stdout__
    model.load_weights(args.model_in)
if args.scaling == 'ON' and os.path.isfile(args.scaler_in):
    print('Loading quantile transform  from:', args.scaler_in)
    scaler = pickle.load(open(args.scaler_in, 'rb'))
args.output_dir += '/plots'


# MODEL TRAINING
def separate_samples(sample):
    JZW        = sample['JZW']
    qcd_sample = {key:val[JZW!=-1] for key,val in sample.items()}
    oe_sample  = {key:val[JZW==-1] for key,val in sample.items()}
    return qcd_sample, oe_sample
if args.n_epochs > 0:
    for path in list(accumulate([folder+'/' for folder in args.output_dir.split('/')])):
        try: os.mkdir(path)
        except FileExistsError: pass
    print('\nTRAINING SAMPLE:')
    sig_bins, bkg_bins = 200, 600
    train_sample = make_sample(args.n_dims, args.n_constituents, 'qcd', 'W',
                               args.n_train, args.n_W, bkg_cuts, sig_cuts)
    train_sample = {key:utils.shuffle(train_sample[key], random_state=0) for key in train_sample}
    train_sample = reweight_sample(train_sample, sig_bins, bkg_bins, args.weight_type)
    #var_distributions(train_sample, args.output_dir, sig_bins, bkg_bins, var='pt', log=True)
    print('\nVALIDATION SAMPLE:')
    valid_sample = make_sample(args.n_dims, args.n_constituents, 'qcd', 'W',
                               args.n_valid, args.n_W, bkg_cuts, sig_cuts)
    valid_sample = reweight_sample(valid_sample, sig_bins, bkg_bins, args.weight_type)
    if args.scaling == 'ON':
        if not os.path.isfile(args.scaler_in):
            JZW    = train_sample['JZW']
            scaler = fit_scaler(train_sample['constituents'][JZW!=-1], args.n_dims, args.scaler_out)
        train_sample['constituents'] = scaling(train_sample['constituents'], args.n_dims, scaler)
        valid_sample['constituents'] = scaling(valid_sample['constituents'], args.n_dims, scaler)
    train_sample, train_sample_OE = separate_samples(train_sample)
    valid_sample, _               = separate_samples(valid_sample)
    print('\nTrain sample:', format(len(train_sample['weights']), '8.0f'), 'jets'  )
    print(  'Valid sample:', format(len(valid_sample['weights']), '8.0f'), 'jets\n')

    train_dataset, valid_dataset = make_datasets(train_sample, train_sample_OE, valid_sample, args.batch_size)
    build_model(model, train_dataset, valid_dataset, args.n_epochs, args.beta, args.lamb, args.lr)
    model.save_weights(args.model_out)


# MODEL PREDICTIONS ON VALIDATION DATA
if args.apply_cut == 'OFF' and args.plotting == 'OFF': sys.exit()
print('\n+'+30*'-'+'+\n+--- TEST SAMPLE EVALUATION ---+\n+'+30*'-'+'+')
sample = make_sample(args.n_dims, args.n_constituents, 'qcd', 'top',
                     args.n_test, args.n_top, bkg_cuts, sig_cuts)
sample = {key:utils.shuffle(sample[key], random_state=0) for key in sample}
y_true = np.where(sample['JZW']==-1, 0, 1)
#var_distributions(sample, args.output_dir, sig_bins=200, bkg_bins=600, var='pt', log=True); sys.exit()
if args.bump_hunter == 'ON':
    bump_hunter(y_true, sample, args.output_dir); sys.exit()
if args.scaling=='ON': X_true = scaling(sample['constituents'], args.n_dims, scaler)
else                 : X_true = sample['constituents']
if args.n_iter > 1: print('\nEvaluating with', args.n_iter, 'iterations:')
X_pred = np.empty(X_true.shape+(args.n_iter,), dtype=np.float32)
for n in np.arange(args.n_iter): X_pred[...,n] = model.predict(X_true, batch_size=int(1e4), verbose=1)
X_pred = np.mean(X_pred, axis=2); print()


# CUT ON RECONSTRUCTION LOSS
if args.apply_cut == 'ON':
    cut_sample = apply_best_cut(y_true, X_true, X_pred, sample, args.n_dims, metric='X-S', cut_type='gain')
    samples    = [sample, cut_sample]
    if args.bump_hunter == 'ON':
        bump_hunter(np.where(cut_sample['JZW']==-1,0,1), cut_sample, args.output_dir); sys.exit()
    var_distributions(samples, args.output_dir, sig_bins=200, bkg_bins=200, var='M', normalize=False)


# PLOTTING RESULTS
if args.plotting == 'ON':
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    plot_results(y_true, X_true, X_pred, sample, args.n_dims, metrics, model, args.output_dir)
