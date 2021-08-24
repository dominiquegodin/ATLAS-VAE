# IMPORT PACKAGES AND FUNCTIONS
import numpy      as np
import tensorflow as tf
import os, sys, pickle
from   argparse  import ArgumentParser
from   itertools import accumulate
from   tabulate  import tabulate
from   sklearn   import model_selection, utils
from   models    import create_model, callback
from   utils     import make_sample, fit_scaler, scaling, s_scaling
from   utils     import reweight_sample, apply_best_cut, bump_hunter
from   plots     import var_distributions, plot_results


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--n_train'       , default =        1e6, type = float         )
parser.add_argument( '--n_valid'       , default =        1e6, type = float         )
parser.add_argument( '--n_test'        , default =        1e6, type = float         )
parser.add_argument( '--n_W'           , default =          0, type = float         )
parser.add_argument( '--n_top'         , default =        1e6, type = float         )
parser.add_argument( '--n_constituents', default =         20, type = int           )
parser.add_argument( '--n_dims'        , default =          3, type = int           )
parser.add_argument( '--batch_size'    , default =        5e3, type = float         )
parser.add_argument( '--n_epochs'      , default =        500, type = int           )
parser.add_argument( '--FC_layers'     , default = [40,20,10], type = int, nargs='+')
parser.add_argument( '--lr'            , default =       1e-3, type = float         )
parser.add_argument( '--beta'          , default =          0, type = float         )
parser.add_argument( '--lamb'          , default =          0, type = float         )
parser.add_argument( '--patience'      , default =         20, type = int           )
parser.add_argument( '--n_iter'        , default =          1, type = int           )
parser.add_argument( '--n_gpus'        , default =          1, type = int           )
parser.add_argument( '--weight_type'   , default = 'flat_pt'                        )
parser.add_argument( '--plotting'      , default = 'OFF'                            )
parser.add_argument( '--apply_cut'     , default = 'OFF'                            )
parser.add_argument( '--output_dir'    , default = 'outputs'                        )
parser.add_argument( '--model_in'      , default = ''                               )
parser.add_argument( '--model_out'     , default = 'model.h5'                       )
parser.add_argument( '--scaling'       , default = 'ON'                             )
parser.add_argument( '--scaler_in'     , default = ''                               )
parser.add_argument( '--scaler_out'    , default = 'scaler.pkl'                     )
parser.add_argument( '--s_scaler_in'   , default = ''                               )
parser.add_argument( '--s_scaler_out'  , default = 's_scaler.pkl'                   )
parser.add_argument( '--encoder'       , default = 'oe_vae'                         )
args = parser.parse_args()
for key in ['n_train', 'n_valid', 'n_test', 'n_W', 'n_top', 'batch_size']:
    vars(args)[key] = int(vars(args)[key])


# TRAINING VARIABLES
train_var = ['rljet_pt_calo' , 'rljet_ECF3'     , 'rljet_C2'      , 'rljet_D2'       ,
             'rljet_Tau1_wta', 'rljet_Tau2_wta' , 'rljet_Tau3_wta', 'rljet_Tau32_wta',
             'rljet_Split12' , 'rljet_Split23'  , 'rljet_Qw'                         ]
train_var = ['constituents']
if args.n_W != 0: train_var = ['constituents']


# METRICS LIST
metrics = ['JSD', 'MSE', 'MAE', 'KLD', 'X-S'] + ['B-1', 'B-2'] #+ ['DNN', 'FCN']
if train_var == ['constituents']: metrics += ['EMD']


# PT CUTS ON SIGNAL AND BACKGROUND SAMPLES
cuts  = ['(sample["pt"] >= 0)', '(sample["weights"] <= 200)']
cuts += ['(sample["rljet_ECF3"] <= 1e18)', '(sample["rljet_C2"       ] >= 0)',
         '(sample["rljet_D2"  ] >= 0   )', '(sample["rljet_Tau32_wta"] >= 0)']
if 'DNN' in metrics: cuts += ['(sample["DNN"]) != -1000']
#cuts += ['(sample["M"] >= 150)', '(sample["M"] <= 190)']
sig_cuts, bkg_cuts = cuts+['(sample["pt"] <= 2000)'], cuts+['(sample["pt"] <= 6000)']


# SAMPLES SIZES
args.n_train = (0              , args.n_train                  )
args.n_valid = (args.n_train[1], args.n_train[1] + args.n_valid)
args.n_test  = (args.n_valid[1], args.n_valid[1] + args.n_test )


# LOADIND PRE-TRAINED WEIGHTS AND SCALER
n_gpus = min(args.n_gpus, len(tf.config.experimental.list_physical_devices('GPU')))
seed   = None if args.n_epochs > 0 else 0
model  = create_model(args.n_dims*args.n_constituents, train_var, args.FC_layers,
                      args.lr, args.beta, args.lamb, seed, args.encoder, n_gpus)
print('PROGRAM ARGUMENTS:\n'+tabulate(vars(args).items(), tablefmt='psql'))
args.model_in    = args.output_dir+'/'+args.model_in   ; args.model_out    = args.output_dir+'/'+args.model_out
args.scaler_in   = args.output_dir+'/'+args.scaler_in  ; args.scaler_out   = args.output_dir+'/'+args.scaler_out
args.s_scaler_in = args.output_dir+'/'+args.s_scaler_in; args.s_scaler_out = args.output_dir+'/'+args.s_scaler_out
if os.path.isfile(args.model_in):
    print('Loading pre-trained weights from:', args.model_in)
    model.load_weights(args.model_in)
if args.scaling == 'ON':
    if os.path.isfile(args.scaler_in) and 'constituents' in train_var:
        print('Loading quantile transform  from:', args.scaler_in)
        scaler = pickle.load(open(args.scaler_in, 'rb'))
    if os.path.isfile(args.s_scaler_in) and len(set(train_var)-{'constituents'}) != 0:
        print('Loading quantile transform  from:', args.s_scaler_in)
        s_scaler = pickle.load(open(args.s_scaler_in, 'rb'))
args.output_dir += '/plots'


# MODEL TRAINING
if args.n_epochs > 0:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #suppress Tensorflow warnings
    for path in list(accumulate([folder+'/' for folder in args.output_dir.split('/')])):
        try: os.mkdir(path)
        except FileExistsError: pass
    print('\nTRAINING SAMPLE:')
    sig_bins, bkg_bins = 200, 600; log = args.weight_type!='flat_pt'
    train_sample = make_sample(args.n_train, args.n_W, args.n_dims, metrics, train_var,
                               args.n_constituents, bkg_cuts, sig_cuts, bkg='qcd', sig='W')
    train_sample = {key:utils.shuffle(train_sample[key], random_state=0) for key in train_sample}
    train_sample = reweight_sample(train_sample, sig_bins, bkg_bins, args.weight_type)
    var_distributions(train_sample, args.output_dir, sig_bins, bkg_bins, var='pt', log=True)
    print('\nVALIDATION SAMPLE:')
    valid_sample = make_sample(args.n_valid, args.n_W, args.n_dims, metrics, train_var,
                               args.n_constituents, bkg_cuts, sig_cuts, bkg='qcd', sig='W')
    if args.scaling == 'ON':
        if 'constituents' in train_var:
            if not os.path.isfile(args.scaler_in):
                JZW    = train_sample['JZW']
                scaler = fit_scaler(train_sample['constituents'][JZW!=-1], args.n_dims, args.scaler_out)
            train_sample['constituents'] = scaling(train_sample['constituents'], args.n_dims, scaler)
            valid_sample['constituents'] = scaling(valid_sample['constituents'], args.n_dims, scaler)
        if len(set(train_var)-{'constituents'}) != 0:
            if not os.path.isfile(args.s_scaler_in):
                s_scaler = fit_scaler(train_sample['scalars'], args.n_dims, args.s_scaler_out, reshape=False)
            train_sample['scalars'] = s_scaling(train_sample['scalars'], s_scaler)
            valid_sample['scalars'] = s_scaling(valid_sample['scalars'], s_scaler)
    def separate_samples(sample):
        JZW = sample['JZW']
        qcd_sample = {key:val[JZW!=-1] for key,val in sample.items()}
        oe_sample  = {key:val[JZW==-1] for key,val in sample.items()}
        return qcd_sample, oe_sample
    train_sample, oe_train_sample = separate_samples(train_sample)
    valid_sample, oe_valid_sample = separate_samples(valid_sample)
    print('\nTrain sample:'   , format(len(train_sample['weights']), '8.0f'), 'jets'  )
    print(  'Valid sample:'   , format(len(valid_sample['weights']), '8.0f'), 'jets\n')
    print(  'Using TensorFlow', tf.__version__, 'with', n_gpus, 'GPU(s)\n')
    callbacks = callback(args.model_out, args.patience, 'val_loss')
    #FOR STAND-ALONE ENCODER
    if args.encoder in ['ae', 'vae']:
        sample_weights = train_sample['weights']
        if len(set(train_var)-{'constituents'}) == 0:
            train_X = train_sample['constituents']
            valid_X = valid_sample['constituents']
        elif 'constituents' not in train_var:
            train_X = train_sample['scalars']
            valid_X = valid_sample['scalars']
        else:
            train_X = np.concatenate([train_sample['constituents'], train_sample['scalars']], axis=1)
            valid_X = np.concatenate([valid_sample['constituents'], valid_sample['scalars']], axis=1)
        train_Y = train_X
        valid_Y = valid_X
    #FOR DUAL ENCODER
    if args.encoder == 'dual_ae':
        sample_weights = [train_sample['weights'],train_sample['weights']]
        train_X = [train_sample['constituents'], train_sample['scalars']]
        valid_X = [valid_sample['constituents'], valid_sample['scalars']]
        train_Y = train_X
        valid_Y = valid_X
    #FOR OE ENCODER
    if args.encoder == 'oe_vae':
        sample_weights = train_sample['weights']
        train_X = [train_sample['constituents'], oe_train_sample['constituents']]
        valid_X = [valid_sample['constituents'], oe_valid_sample['constituents']]
        train_Y = train_sample['constituents']
        valid_Y = valid_sample['constituents']
    training = model.fit(train_X, train_Y, validation_data=(valid_X,valid_Y), callbacks=callbacks,
                         batch_size=args.batch_size, epochs=args.n_epochs, sample_weight=sample_weights)
    model.load_weights(args.model_out)


# MODEL PREDICTIONS ON VALIDATION DATA
if args.apply_cut == 'OFF' and args.plotting == 'OFF': sys.exit()
print('\n+'+30*'-'+'+\n+--- TEST SAMPLE EVALUATION ---+\n+'+30*'-'+'+')
sample = make_sample(args.n_test, args.n_top, args.n_dims, metrics, train_var,
                     args.n_constituents, bkg_cuts, sig_cuts, bkg='qcd', sig='top')
sample = {key:utils.shuffle(sample[key], random_state=0) for key in sample}
y_true = np.where(sample['JZW']==-1, 0, 1)
#var_distributions(sample, args.output_dir, sig_bins=200, bkg_bins=600, var='pt', log=True); sys.exit()
#bump_hunter(y_true, sample, args.output_dir); sys.exit()
if 'constituents' in train_var:
    j_X_true = scaling(sample['constituents'], args.n_dims, scaler) if args.scaling=='ON' else sample['constituents']
if len(set(train_var)-{'constituents'}) != 0:
    s_X_true = s_scaling(sample['scalars'], s_scaler) if args.scaling=='ON' else sample['scalars']
if len(set(train_var)-{'constituents'}) == 0: X_true = j_X_true
elif 'constituents' not in train_var        : X_true = s_X_true
else                                        : X_true = np.concatenate([j_X_true, s_X_true], axis=1)
#FOR DUAL ENCODER
if args.encoder == 'dual_ae':
    X_pred = model.predict([j_X_true, s_X_true], batch_size=max(1,n_gpus)*int(1e4), verbose=1)[0]
    X_true = j_X_true
#FOR STAND-ALONE ENCODER
else:
    if args.n_iter > 1: print('Evaluating with', args.n_iter, 'iterations:')
    X_pred = np.empty(X_true.shape+(args.n_iter,), dtype=np.float32)
    for n in np.arange(args.n_iter):
        if args.encoder == 'oe_vae':
            X_pred[...,n] = model.predict([X_true, X_true], batch_size=max(1,n_gpus)*int(1e4), verbose=1)
        else:
            X_pred[...,n] = model.predict(X_true, batch_size=max(1,n_gpus)*int(1e4), verbose=1)
    X_pred = np.mean(X_pred, axis=2); print()


# CUT ON RECONSTRUCTION LOSS
if args.apply_cut == 'ON':
    cut_sample = apply_best_cut(y_true, X_true, X_pred, sample, args.n_dims, metric='X-S', cut_type='gain')
    samples    = [sample, cut_sample]
    #bump_hunter(np.where(cut_sample['JZW']==-1,0,1), cut_sample, args.output_dir); sys.exit()
    var_distributions(samples, args.output_dir, sig_bins=200, bkg_bins=200, var='M', normalize=False)


# PLOTTING RESULTS
if args.plotting == 'ON':
    if not os.path.isdir(args.output_dir): os.mkdir(args.output_dir)
    plot_results(y_true, X_true, X_pred, sample, train_var, args.n_dims,
                 metrics, model, args.encoder, args.output_dir)



'''
from sklearn.manifold import TSNE
from tensorflow.keras import models
import matplotlib.pyplot as plt
import time
#codings_layer = models.Model(inputs=model.inputs, outputs=model.get_layer('model').get_layer('codings').output)
codings_layer = models.Model(inputs=model.inputs, outputs=model.get_layer('model').get_layer('mean').output)
codings       = codings_layer(np.float32(X_true))
print(codings.shape)
start_time = time.time()
tsne = TSNE(n_jobs=-1, random_state=0)
codings_2D = tsne.fit_transform(codings)
print(codings_2D.shape)
#plt.scatter(codings_2D[:,0], codings_2D[:,1], c=['tab:orange', 'tab:blue'], s=10, label=['Tops', 'QCD'], alpha=0.5)

plt.figure(figsize=(12,8))
labels = [r'$t\bar{t}$', 'QCD']
colors = ['tab:orange', 'tab:blue']
for n in set(y_true):
    x = codings_2D[:,0][y_true==n]
    y = codings_2D[:,1][y_true==n]
    plt.scatter(x, y, color=colors[n], s=10, label=labels[n], alpha=0.1)
leg = plt.legend(loc='upper right', fontsize=18)
for lh in leg.legendHandles: lh.set_alpha(1)
plt.savefig(args.output_dir+'/'+'scatter.png')
print('(', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
'''
