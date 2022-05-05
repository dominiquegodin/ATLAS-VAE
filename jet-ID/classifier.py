# IMPORT PACKAGES AND FUNCTIONS
import tensorflow      as tf
import numpy           as np
import os, sys, h5py, pickle
from   argparse  import ArgumentParser
from   tabulate  import tabulate
from   itertools import accumulate
from   utils     import validation, make_sample, merge_samples, sample_composition
from   utils     import compo_matrix, get_sample_weights, get_class_weight, gen_weights
from   utils     import cross_valid, valid_results, sample_histograms, Batch_Generator
from   utils     import feature_removal, feature_ranking, fit_scaler, apply_scaler, fit_t_scaler, apply_t_scaler
from   models    import callback, create_model


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--n_train'        , default =  1e6,  type = float )
parser.add_argument( '--n_eval'         , default =    0,  type = float )
parser.add_argument( '--n_valid'        , default =  1e6,  type = float )
parser.add_argument( '--batch_size'     , default =  5e3,  type = float )
parser.add_argument( '--n_epochs'       , default =  100,  type = int   )
parser.add_argument( '--n_classes'      , default =    2,  type = int   )
parser.add_argument( '--n_tracks'       , default =    5,  type = int   )
parser.add_argument( '--bkg_ratio'      , default =    2,  type = float )
parser.add_argument( '--n_folds'        , default =    1,  type = int   )
parser.add_argument( '--n_gpus'         , default =    1,  type = int   )
parser.add_argument( '--verbose'        , default =    1,  type = int   )
parser.add_argument( '--patience'       , default =   15,  type = int   )
parser.add_argument( '--sbatch_var'     , default =    0,  type = int   )
parser.add_argument( '--node_dir'       , default = ''                  )
parser.add_argument( '--host_name'      , default = 'lps'               )
parser.add_argument( '--l2'             , default = 1e-7,  type = float )
parser.add_argument( '--dropout'        , default =  0.1,  type = float )
parser.add_argument( '--FCN_neurons'    , default = [500,500], type = int, nargs='+')
parser.add_argument( '--weight_type'    , default = 'none'              )
parser.add_argument( '--train_cuts'     , default = ''                  )
parser.add_argument( '--valid_cuts'     , default = ''                  )
parser.add_argument( '--NN_type'        , default = 'CNN'               )
parser.add_argument( '--images'         , default = 'ON'                )
parser.add_argument( '--scalars'        , default = 'ON'                )
parser.add_argument( '--scaling'        , default = 'ON'                )
parser.add_argument( '--t_scaling'      , default = 'ON'                )
parser.add_argument( '--plotting'       , default = 'ON'                )
parser.add_argument( '--generator'      , default = 'OFF'               )
parser.add_argument( '--sep_bkg'        , default = 'OFF'               )
#parser.add_argument( '--metrics'        , default = 'val_accuracy'      )
parser.add_argument( '--metrics'        , default = 'loss'      )
parser.add_argument( '--eta_region'     , default = '0.0-2.5'           )
parser.add_argument( '--output_dir'     , default = 'outputs'           )
parser.add_argument( '--model_in'       , default = ''                  )
parser.add_argument( '--model_out'      , default = 'model.h5'          )
parser.add_argument( '--scaler_in'      , default = ''                  )
parser.add_argument( '--scaler_out'     , default = 'scaler.pkl'        )
parser.add_argument( '--t_scaler_in'    , default = ''                  )
parser.add_argument( '--t_scaler_out'   , default = 't_scaler.pkl'      )
parser.add_argument( '--results_in'     , default = ''                  )
parser.add_argument( '--results_out'    , default = ''                  )
parser.add_argument( '--runDiffPlots'   , default = 0, type = int       )
parser.add_argument( '--feature_removal', default = 'OFF'               )
parser.add_argument( '--correlations'   , default = 'OFF'               )
args = parser.parse_args()


# VERIFYING ARGUMENTS
for key in ['n_train', 'n_eval', 'n_valid', 'batch_size']: vars(args)[key] = int(vars(args)[key])
if args.weight_type not in ['bkg_ratio', 'flattening', 'match2s', 'match2b', 'match2class', 'match2max', 'none']:
    print('\nweight_type', args.weight_type, 'not recognized --> resetting it to none\n')
    args.weight_type = 'none'
if '.h5' not in args.model_in and args.n_epochs < 1 and args.n_folds==1:
    print('\nERROR: weights file required with n_epochs < 1 --> aborting\n'); sys.exit()


# CNN PARAMETERS
CNN = {(56,11):{'maps':[100,100], 'kernels':[ (3,5) , (3,5) ], 'pools':[ (4,1) , (2,1) ]},
        (7,11):{'maps':[100,100], 'kernels':[ (3,5) , (3,5) ], 'pools':[ (1,1) , (1,1) ]},
        #(7,11):{'maps':[100,100], 'kernels':[(3,5,3),(3,5,3)], 'pools':[(1,1,1),(1,1,1)]},
      'tracks':{'maps':[200,200], 'kernels':[ (1,1) , (1,1) ], 'pools':[ (1,1) , (1,1) ]}}


# TRAINING VARIABLES
#scalars  = ['rljet_m_comb', 'rljet_pt_calo', 'rljet_ECF3', 'rljet_C2', 'rljet_D2', 'rljet_Tau1_wta',
#            'rljet_Tau2_wta', 'rljet_Tau3_wta', 'rljet_Tau32_wta', 'rljet_Split12', 'rljet_Split23', 'rljet_Qw']
scalars  = ['constituents']
images   = []
others   = ['JZW', 'weights', 'rljet_topTag_DNN19_qqb_score', 'rljet_pt_calo']
data_files = [
              '/opt/tmp/godin/AD_data/formatted_converted_20210430_ttbar_allhad_pT_450_1200_nevents_1M.h5',
              '/opt/tmp/godin/AD_data/formatted_converted_20210629_QCDjj_pT_450_1200_nevents_10M.h5',
              '/opt/tmp/godin/AD_data/formatted_delphes_H_HpHm_generation_mh2_5000_mhc_500_nevents_1M.h5',
              '/opt/tmp/godin/AD_data/H_HpHm_generation_merged_with_masses_20_40_60_80_reformatted_nghia.h5'
              ]
keys       = set().union(*[h5py.File(data_file,'r').keys() for data_file in data_files])
images     = [key for key in images  if key in keys or key=='tracks']
scalars    = [key for key in scalars if key in keys or key=='tracks']
others     = [key for key in others  if key in keys]
if args.scalars != 'ON': scalars=[]
if args.images  != 'ON': images =[]
if args.feature_removal == 'ON':
    groups = [('em_barrel_Lr1','em_barrel_Lr1_fine'), ('em_barrel_Lr0','em_barrel_Lr2','em_barrel_Lr3')]
    scalars, images, removed_feature = feature_removal(scalars, images, groups=[], index=args.sbatch_var)
    args.output_dir += '/'+removed_feature
if images == []: args.NN_type = 'FCN'
train_data = {'scalars':scalars, 'images':images}
input_data = {**train_data, 'others':others}


# SAMPLES SIZES AND APPLIED CUTS ON PHYSICS VARIABLES
sample_size  = sum([len(h5py.File(data_file,'r')['constituents']) for data_file in data_files])
#sample_size  = sum([len(h5py.File(data_file,'r')['eventNumber']) for data_file in data_files])
args.n_train = [0, min(sample_size, args.n_train)]
args.n_valid = [args.n_train[1], min(args.n_train[1]+args.n_valid, sample_size)]
if args.n_valid[0] == args.n_valid[1]: args.n_valid = args.n_train
if args.n_eval != 0: args.n_eval = [args.n_valid[0], min(args.n_valid[1],args.n_valid[0]+args.n_eval)]
else               : args.n_eval = args.n_valid
#args.train_cuts = '(abs(sample["eta"]) > 0.8) & (abs(sample["eta"]) < 1.15)'
#args.valid_cuts = '(sample["p_et_calo"] > 4.5) & (sample["p_et_calo"] < 20)'
#args.train_cuts = '((sample["mcChannelNumber"]==361106) | (sample["mcChannelNumber"]==423300)) & (sample["pt"]>=15)'
#args.valid_cuts = '((sample["mcChannelNumber"]==361106) | (sample["mcChannelNumber"]==423300)) & (sample["pt"]>=15)'


# OBTAINING PERFORMANCE FROM EXISTING VALIDATION RESULTS
if os.path.isfile(args.output_dir+'/'+args.results_in) or os.path.islink(args.output_dir+'/'+args.results_in):
    if args.eta_region in ['0.0-1.3', '1.3-1.6', '1.6-2.5']:
        eta_1, eta_2 = args.eta_region.split('-')
        valid_cuts   = '(abs(sample["eta"]) >= '+str(eta_1)+') & (abs(sample["eta"]) <= '+str(eta_2)+')'
        if args.valid_cuts == '': args.valid_cuts  = valid_cuts
        else                    : args.valid_cuts  = valid_cuts + '& ('+args.valid_cuts+')'
    inputs = {'scalars':scalars, 'images':[], 'others':others}
    validation(args.output_dir, args.results_in, args.plotting, args.n_valid, data_files,
               inputs, args.valid_cuts, args.sep_bkg, args.runDiffPlots)
elif args.results_in != '': print('\nOption --results_in not matching any file --> aborting\n')
if   args.results_in != '': sys.exit()


# MODEL CREATION AND MULTI-GPU DISTRIBUTION
n_gpus = min(args.n_gpus, len(tf.config.experimental.list_physical_devices('GPU')))
train_batch_size = max(1,n_gpus) * args.batch_size
valid_batch_size = max(1,n_gpus) * max(args.batch_size, int(5e3))
sample = make_sample(data_files[0], [0,1], input_data, args.n_tracks, args.n_classes)[0]
model  = create_model(args.n_classes, sample, args.NN_type, args.FCN_neurons, CNN,
                      args.l2, args.dropout, train_data, n_gpus)


# ARGUMENTS AND VARIABLES SUMMARY
args.scaling = args.scaling == 'ON' and list(set(scalars)-{'constituents'}) != []
args.t_scaling = args.t_scaling == 'ON' and 'constituents' in scalars+images
if args.NN_type == 'CNN':
    print('\nCNN ARCHITECTURE:')
    for shape in [shape for shape in CNN if shape in [sample[key].shape[1:] for key in sample]]:
        print(format(str(shape),'>8s')+':', str(CNN[shape]))
print('\nPROGRAM ARGUMENTS:'); print(tabulate(vars(args).items(), tablefmt='psql'))
print('\nTRAINING VARIABLES:')
headers = [           key  for key in train_data if train_data[key]!=[]]
table   = [train_data[key] for key in train_data if train_data[key]!=[]]
length  = max([len(n) for n in table])
table   = list(map(list, zip(*[n+(length-len(n))*[''] for n in table])))
print(tabulate(table, headers=headers, tablefmt='psql')); print()
args.model_in   = args.output_dir+'/'+args.model_in;   args.model_out   = args.output_dir+'/'+args.model_out
args.scaler_in  = args.output_dir+'/'+args.scaler_in;  args.scaler_out  = args.output_dir+'/'+args.scaler_out
args.t_scaler_in = args.output_dir+'/'+args.t_scaler_in ; args.t_scaler_out = args.output_dir+'/'+args.t_scaler_out
args.results_in = args.output_dir+'/'+args.results_in; args.results_out = args.output_dir+'/'+args.results_out


# GENERATING VALIDATION SAMPLE AND LOADING PRE-TRAINED WEIGHTS
if os.path.isfile(args.model_in):
    print('Loading pre-trained weights from', args.model_in, '\n')
    model.load_weights(args.model_in)
if args.scaling and os.path.isfile(args.scaler_in):
    print('Loading quantile transform from', args.scaler_in, '\n')
    scaler = pickle.load(open(args.scaler_in, 'rb'))
else: scaler = None
if args.t_scaling and os.path.isfile(args.t_scaler_in):
    print('Loading tracks scaler from ', args.t_scaler_in, '\n')
    t_scaler = pickle.load(open(args.t_scaler_in, 'rb'))
else: t_scaler = None
print('LOADING VALIDATION SAMPLES')
inputs = {'scalars':scalars, 'images':[], 'others':others} if args.generator == 'ON' else input_data
#valid_sample, valid_labels, _ = merge_samples(data_files, args.n_valid, inputs, args.n_tracks, args.n_classes,
#                                              args.valid_cuts, None if args.generator=='ON' else scaler)

from sklearn import utils
from utils import weights_factors
sample_1, labels_1, _ = merge_samples([data_files[0]], [800_000, 982_823], inputs, args.n_tracks, args.n_classes,
           args.valid_cuts, None if args.generator=='ON' else scaler, None if args.generator=='ON' else t_scaler)
labels_1 = np.zeros_like(labels_1)
sample_1['labels'] = labels_1
#sample_1['weights'] = sample_1['weights']*weights_factors(sample_1['JZW'], data_files[0])
sample_2, labels_2, _ = merge_samples([data_files[1]], [8000_000, 8542_774], inputs, args.n_tracks, args.n_classes,
           args.valid_cuts, None if args.generator=='ON' else scaler, None if args.generator=='ON' else t_scaler)
labels_2 = np.ones_like(labels_2)
sample_2['labels'] = labels_2
#sample_2['weights'] = sample_2['weights']*weights_factors(sample_2['JZW'], data_files[1])
valid_sample = {key:np.concatenate([n[key] for n in [sample_1, sample_2]]) for key in sample_1}
valid_labels = np.concatenate([n for n in [labels_1, labels_2]])
#DNN_tag = valid_sample['rljet_topTag_DNN19_qqb_score']
#cut = np.logical_and( np.isfinite(DNN_tag), DNN_tag!=-1000 )
#valid_sample = {key:valid_sample[key][cut] for key in valid_sample}
#valid_labels = valid_labels[cut]
#for key in valid_sample: print(key, valid_sample[key].shape)
#from plots_DG import plot_weights
#plot_weights(valid_sample, valid_labels, args.output_dir)
#sys.exit()

#for key in valid_sample:
#    print(key, valid_sample[key].shape)
#sys.exit()


# TRAINING LOOP
if args.n_epochs > 0:
    for path in list(accumulate([folder+'/' for folder in args.output_dir.split('/')])):
        try: os.mkdir(path)
        except FileExistsError: pass
    print('\nUsing TensorFlow', tf.__version__                               )
    print(  'Using'           , n_gpus, 'GPU(s)'                             )
    print(  'Using'           , args.NN_type, 'architecture with', end=' '   )
    print([key for key in train_data if train_data[key] != []], '\n'         )
    print('LOADING TRAINING SAMPLES'            )
    #train_sample, train_labels, weight_idx = merge_samples(data_files, args.n_train, inputs, args.n_tracks,
    #                                                       args.n_classes, args.train_cuts, scaler=None)
    sample_1, labels_1, _ = merge_samples([data_files[0]], [0, 800_000], inputs, args.n_tracks, args.n_classes,
           args.train_cuts, scaler=None, t_scaler=None)
    labels_1 = np.zeros_like(labels_1)
    #sample_1['weights'] = sample_1['weights']*weights_factors(sample_1['JZW'], data_files[0])
    sample_2, labels_2, _ = merge_samples([data_files[1]], [0, 8000_000], inputs, args.n_tracks, args.n_classes,
           args.train_cuts, scaler=None, t_scaler=None)
    labels_2 = np.ones_like(labels_2)
    #sample_2['weights'] = sample_2['weights']*weights_factors(sample_2['JZW'], data_files[1])
    train_sample = {key:utils.shuffle(np.concatenate([n[key] for n in [sample_1,sample_2]]), random_state=0)
                    for key in sample_1}
    train_labels = utils.shuffle( np.concatenate([n for n in [labels_1,labels_2]]), random_state=0 )
    #train_sample = sample_2
    #train_labels = labels_2
    #DNN_tag = train_sample['rljet_topTag_DNN19_qqb_score']
    #cut = np.logical_and( np.isfinite(DNN_tag), DNN_tag!=-1000 )
    #train_sample = {key:train_sample[key][cut] for key in train_sample}
    #train_labels = train_labels[cut]

    if args.scaling:
        if not os.path.isfile(args.scaler_in):
            scaler = fit_scaler(train_sample, scalars, args.scaler_out)
            if args.generator != 'ON': valid_sample = apply_scaler(valid_sample, scalars, scaler, verbose='ON')
        if args.generator != 'ON': train_sample = apply_scaler(train_sample, scalars, scaler, verbose='ON')

    if args.t_scaling:
        if not os.path.isfile(args.t_scaler_in):
            t_scaler = fit_t_scaler(train_sample, args.t_scaler_out)
            if args.generator != 'ON': valid_sample = apply_t_scaler(valid_sample, t_scaler, verbose='ON')
        if args.generator != 'ON': train_sample = apply_t_scaler(train_sample, t_scaler, verbose='ON')

    #sample_composition(train_sample)
    print(); compo_matrix(valid_labels, train_labels=train_labels); print()
    #train_weights, bins = get_sample_weights(train_sample, train_labels, args.weight_type, args.bkg_ratio, hist='pt')
    train_weights = train_sample['weights']
    #sample_histograms(valid_sample, valid_labels, train_sample, train_labels, train_weights, bins, args.output_dir)
    callbacks = callback(args.model_out, args.patience, args.metrics)
    if args.generator == 'ON':
        del(train_sample)
        if np.all(train_weights) != None: train_weights = gen_weights(args.n_train, weight_idx, train_weights)
        print('\nLAUNCHING GENERATOR FOR', np.diff(args.n_train)[0], 'TRAINING SAMPLES')
        eval_gen  = Batch_Generator(data_files, args.n_eval, input_data, args.n_tracks, args.n_classes,
                                    valid_batch_size, args.valid_cuts, scaler, t_scaler, shuffle='OFF')
        train_gen = Batch_Generator(data_files, args.n_train, input_data, args.n_tracks, args.n_classes,
                                    train_batch_size, args.train_cuts, scaler, t_scaler, train_weights, shuffle='ON')
        training  = model.fit( train_gen, validation_data=eval_gen, max_queue_size=100*max(1,n_gpus),
                               callbacks=callbacks, workers=1, epochs=args.n_epochs, verbose=args.verbose )
    else:
        training  = model.fit( train_sample, train_labels, validation_data=(valid_sample,valid_labels),
                               callbacks=callbacks, sample_weight=train_weights, batch_size=train_batch_size,
                               epochs=args.n_epochs, verbose=args.verbose )
    model.load_weights(args.model_out); print()
else: train_labels = []; training = None


# RESULTS AND PLOTTING SECTION
if args.n_folds > 1:
    valid_probs = cross_valid(valid_sample, valid_labels, scalars, args.output_dir, args.n_folds, data_files,
                              args.n_valid, input_data, args.n_tracks, args.valid_cuts, model, args.generator)
else:
    print('Validation sample', args.n_valid, 'class predictions:')
    if args.generator == 'ON':
        valid_gen   = Batch_Generator(data_files, args.n_valid, input_data, args.n_tracks, args.n_classes,
                                      valid_batch_size, args.valid_cuts, scaler, t_scaler, shuffle='OFF')
        valid_probs = model.predict(valid_gen, verbose=args.verbose)
    else:
        valid_probs = model.predict(valid_sample, batch_size=valid_batch_size, verbose=args.verbose)
bkg_rej = valid_results(valid_sample, valid_labels, valid_probs, train_labels, training,
                        args.output_dir, args.plotting, args.sep_bkg, args.runDiffPlots)
if '.pkl' in args.results_out:
    if args.feature_removal == 'ON':
        args.results_out = args.output_dir[0:args.output_dir.rfind('/')]+'/'+args.results_out.split('/')[-1]
        try: pickle.dump({removed_feature:bkg_rej}, open(args.results_out,'ab'))
        except IOError: print('FILE ACCESS CONFLICT FOR', removed_feature, '--> SKIPPING FILE ACCESS\n')
        feature_ranking(args.output_dir, args.results_out, scalars, images, groups=[])
    else:
        #if args.n_folds > 1 and False: valid_data = (np.float16(valid_probs),)
        #else: valid_data = ({key:valid_sample[key] for key in others}, valid_labels, valid_probs)
        #pickle.dump(valid_data, open(args.results_out,'wb'), protocol=4)
        valid_data = valid_probs[valid_labels==0][:,0]
        pickle.dump(valid_data, open(args.output_dir+'/'+'FCN_tagger_ttbar.pkl','wb'), protocol=4)
        valid_data = valid_probs[valid_labels==1][:,0]
        pickle.dump(valid_data, open(args.output_dir+'/'+'FCN_tagger_dijet.pkl','wb'), protocol=4)
    print('\nValidation results saved to:', args.results_out, '\n')
