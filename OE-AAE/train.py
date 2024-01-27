# IMPORT PACKAGES AND FUNCTIONS
import numpy as np
import os, sys, pickle, h5py
from   argparse import ArgumentParser
from   pathlib  import Path
from   tabulate import tabulate
from   aae      import create_model, train_AAE
from   utils    import get_file, load_data, make_sample, Batch_Generator, get_data
from   utils    import grid_search, fit_scaler, apply_scaler, adjust_weights
from   plots    import sample_distributions, plot_results, plot_history


# PROGRAM ARGUMENTS
parser = ArgumentParser()
parser.add_argument( '--n_train'      , default = 1e6          , type = float         )
parser.add_argument( '--n_valid'      , default = 1e6          , type = float         )
parser.add_argument( '--n_OoD'        , default = 10e6         , type = float         )
parser.add_argument( '--n_sig'        , default = 1e6          , type = float         )
parser.add_argument( '--n_const'      , default = 20           , type = int           )
parser.add_argument( '--n_dims'       , default = 3            , type = int           )
parser.add_argument( '--batch_size'   , default = 5e3          , type = float         )
parser.add_argument( '--n_epochs'     , default = 100          , type = int           )
parser.add_argument( '--layers_sizes' , default = [100,100,100], type = int, nargs='+')
parser.add_argument( '--lr'           , default = 1e-3         , type = float         )
parser.add_argument( '--beta'         , default = 0            , type = float         )
parser.add_argument( '--lamb'         , default = 0            , type = float         )
parser.add_argument( '--margin'       , default = 1            , type = float         )
parser.add_argument( '--n_iter'       , default = 1            , type = int           )
parser.add_argument( '--OE_type'      , default = 'KLD'                               )
parser.add_argument( '--weight_type'  , default = 'X-S'                               )
parser.add_argument( '--model_in'     , default = ''                                  )
parser.add_argument( '--model_out'    , default = 'model.h5'                          )
parser.add_argument( '--const_scaler_type', default = ''                              )
parser.add_argument( '--const_scaler_in'  , default = ''                              )
parser.add_argument( '--const_scaler_out' , default = ''                              )
parser.add_argument( '--HLV_scaler_type'  , default = ''                              )
parser.add_argument( '--HLV_scaler_in'    , default = ''                              )
parser.add_argument( '--HLV_scaler_out'   , default = ''                              )
parser.add_argument( '--hist_file'    , default = 'history.pkl'                       )
parser.add_argument( '--output_dir'   , default = 'outputs'                           )
parser.add_argument( '--plotting'     , default = 'ON'                                )
parser.add_argument( '--apply_cuts'   , default = 'OFF'                               )
parser.add_argument( '--normal_loss'  , default = 'ON'                                )
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
#from plots import deco_example
#deco_example(args.output_dir) ; sys.exit()


# SAMPLES SELECTIONS
bkg_data, OoD_data = 'QCD-Geneva', 'OoD-H'
HLV_list = ['rljet_Tau1_wta', 'rljet_Tau2_wta', 'rljet_Tau3_wta', 'rljet_eta', 'rljet_ECF3',
            'ECF2', 'd12', 'd23', 'pt', 'm', 'tau21', 'tau32']
if args.constituents == 'ON'  and args.HLVs == 'ON' : input_size = args.n_dims*args.n_const + len(HLV_list)
if args.constituents == 'ON'  and args.HLVs == 'OFF': input_size = args.n_dims*args.n_const
if args.constituents == 'OFF' and args.HLVs == 'ON' : input_size =                            len(HLV_list)
bkg_size  = len(list(h5py.File(get_file(bkg_data),'r').values())[0])
OoD_size  = len(list(h5py.File(get_file(OoD_data),'r').values())[0])
args.n_train = [0, min(args.n_train, max(int(1e6), bkg_size-args.n_valid))]
args.n_valid = [max(args.n_train[1],bkg_size-args.n_valid), bkg_size   ]
args.n_OoD   = min(OoD_size, args.n_OoD)
gen_cuts   =            ['(sample["m" ] >=   20)']
train_cuts = gen_cuts + ['(sample["pt"] <= 5000)']
valid_cuts = gen_cuts + ['(sample["pt"] <= 5000)']
for n in range(len(train_cuts)): vars(args)['train cuts ('+str(n+1)+')'] = train_cuts[n]
for n in range(len(valid_cuts)): vars(args)['valid cuts ('+str(n+1)+')'] = valid_cuts[n]
print('\nPROGRAM ARGUMENTS:\n'+tabulate(vars(args).items(), tablefmt='psql'))
#for key,val in h5py.File(get_file(bkg_data),"r").items(): print(key, val.shape, val.dtype)


# LOADIND PRE-TRAINED WEIGHTS AND/OR CONSTITUENTS SCALER
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Suppressing Tensorflow infos and warnings
model = create_model(input_size, args.layers_sizes)
multithread = True ; const_scaler = None ; HLV_scaler = None
if args.model_in != args.output_dir[0:args.output_dir.rfind('/')]+'/':
    if not os.path.isfile(args.model_in): sys.exit()
    print('\nLoading pre-trained weights from: ' + args.model_in)
    sys.stdout = open(os.devnull, 'w') #Stopping screen output
    sample = Batch_Generator(bkg_data, args.n_const, args.n_dims, n_bkg=[0,1],
                             constituents=args.constituents, HLVs=args.HLVs, HLV_list=HLV_list)
    sys.stdout = sys.__stdout__        #Resuming screen output
    model.load_weights(args.model_in)
    multithread = False
if args.const_scaler_type.lower() != '' and os.path.isfile(args.const_scaler_in):
    print('\nLoading scaler/transformer from: ' + args.const_scaler_in)
    const_scaler = pickle.load(open(args.const_scaler_in, 'rb'))
if args.HLV_scaler_type.lower()   != '' and os.path.isfile(args.HLV_scaler_in):
    print('\nLoading scaler/transformer from: ' + args.HLV_scaler_in)
    HLV_scaler = pickle.load(open(args.HLV_scaler_in, 'rb'))


# MODEL TRAINING
if args.n_epochs > 0:
    if (args.constituents == 'ON' and args.const_scaler_type.lower() != '' and const_scaler == None)\
    or (args.HLVs         == 'ON' and args.HLV_scaler_type.lower()   != '' and   HLV_scaler == None):
        print('\nLoading QCD training sample'.upper())
        n_jets = min(args.n_train[1], int(1e9*30/args.n_const/args.n_dims/4))
        train_sample = load_data(bkg_data, n_jets, train_cuts, args.n_const, args.n_dims,
                                 args.constituents, args.HLVs, HLV_list)
        if args.constituents == 'ON':
            const_scaler = fit_scaler(train_sample['constituents'], args.n_dims,
                                      args.const_scaler_out, args.const_scaler_type)
        if args.HLVs == 'ON':
            HLV_scaler = fit_scaler(train_sample['HLVs'], args.n_dims,
                                    args.HLV_scaler_out, args.HLV_scaler_type)
    print('\nLoading outlier sample'.upper())
    OoD_sample = load_data(OoD_data, args.n_OoD, train_cuts, args.n_const, args.n_dims,
                           args.constituents, args.HLVs, HLV_list)
    if 'constituents' in OoD_sample:
        OoD_sample['constituents'] = apply_scaler(OoD_sample['constituents'], args.n_dims, const_scaler, 'OoD')
    if 'HLVs' in OoD_sample:
        OoD_sample['HLVs'        ] = apply_scaler(OoD_sample['HLVs'        ], args.n_dims,   HLV_scaler, 'OoD')
    bin_sizes = {'m':20,'pt':40} if args.weight_type.split('_')[0] in ['flat','OoD'] else {'m':10,'pt':20}
    train_sample = Batch_Generator(bkg_data, args.n_const, args.n_dims, args.n_train, OoD_sample,
                                   args.weight_type, train_cuts, multithread, args.constituents, args.HLVs,
                                   HLV_list, bin_sizes, HLV_scaler, const_scaler, args.output_dir)
    train_AAE(model, train_sample, args.n_epochs, args.batch_size, args.output_dir, args.model_out)
    #_, _, AAE = model ; AAE.load_weights(args.model_out)
if args. plotting == 'OFF' and args.apply_cuts == 'OFF': sys.exit()


#Autoencoder, Discriminator, AAE = model
#Autoencoder  .load_weights(args.output_dir[0:args.output_dir.rfind('/')]+'/'+'autoencodeur.h5')
#Discriminator.load_weights(args.output_dir[0:args.output_dir.rfind('/')]+'/'+'discriminator.h5')
_, _, AAE = model
AAE.load_weights(args.output_dir[0:args.output_dir.rfind('/')]+'/'+'AAE.h5')
Autoencoder   = AAE.get_layer(name='AUTOENCODER')
Discriminator = AAE.get_layer(name='DISCRIMINATOR')
#print('\n'); Autoencoder.summary()
#Discriminator.trainable = True
#print('\n'); Discriminator.summary()
#sys.exit()


# MODEL PREDICTIONS ON VALIDATION SAMPLE
print('\n+'+36*'-'+'+\n+--- VALIDATION SAMPLE EVALUATION ---+\n+'+36*'-'+'+\n')
valid_data = {sig_data:get_data(Autoencoder, Discriminator, bkg_data, sig_data, args.n_valid, args.n_sig, valid_cuts,
                                args.n_const, args.n_dims, args.constituents, args.HLVs, HLV_list,
                                const_scaler, HLV_scaler, args.normal_loss, args.decorrelation)
              for sig_data in ['2HDM_500GeV']}
              #for sig_data in ['top-Geneva']}
              #for sig_data in ['VZ-Geneva']}


# PLOTTING PERFORMANCE RESULTS
if args.plotting == 'ON':
    if os.path.isfile(args.hist_file): plot_history(args.hist_file, args.output_dir)
    plot_results(valid_data, args.output_dir, args.apply_cuts)
