import numpy      as np
import tensorflow as tf
import time, sys, os, pickle
from sklearn   import utils
from functools import partial
from tensorflow.keras import layers, models, Input, initializers, regularizers, optimizers, callbacks
from tensorflow.keras import losses


def Euclidean_dist(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square  (y_pred-y_true), axis=1))
def MAE_dist(y_true, y_pred):
    return         tf.reduce_mean(tf.math.abs(y_pred-y_true), axis=1)
def get_accuracy(y_true, y_pred, weights):
    class_pred = np.argmax(y_pred, axis=1)
    #return np.sum(class_pred==y_true)/len(y_true)
    return (class_pred==y_true)@weights/(np.sum(weights))


def encoder_model(input_size, layers_sizes, activation, kernel, batchNorm, name):
    encoder_inputs = layers.Input(shape=input_size); x = encoder_inputs
    for size in layers_sizes[:-1]:
        x = layers.Dense(size, kernel_initializer=kernel)(x)
        if batchNorm: x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
    encoder_outputs = layers.Dense(layers_sizes[-1], activation='relu', kernel_initializer=kernel)(x)
    encoder = models.Model(inputs=encoder_inputs, outputs=encoder_outputs, name=name)
    return encoder


def decoder_model(output_size, layers_sizes, activation, kernel, batchNorm, name):
    decoder_inputs = layers.Input(shape=layers_sizes[-1]); x = decoder_inputs
    for size in layers_sizes[:-1][::-1]:
        x = layers.Dense(size, kernel_initializer=kernel)(x)
        if batchNorm: x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
    decoder_outputs = layers.Dense(output_size, activation='relu', kernel_initializer=kernel)(x)
    decoder = models.Model(inputs=decoder_inputs, outputs=decoder_outputs, name=name)
    return decoder


def autoencoder_model(input_size, layers_sizes, activation, kernel, batchNorm):
    encoder = encoder_model(input_size, layers_sizes, activation, kernel, batchNorm, name='ENCODER')
    decoder = decoder_model(input_size, layers_sizes, activation, kernel, batchNorm, name='DECODER')
    autoencoder = models.Sequential([encoder, decoder], name='AUTOENCODER')
    return autoencoder


def discriminator_model(input_size, layers_sizes, activation, kernel, batchNorm, name='DISCRIMINATOR'):
    discriminator_inputs = layers.Input(shape=input_size); x = discriminator_inputs
    for size in layers_sizes[:-1]:
        x = layers.Dense(size, kernel_initializer=kernel)(x)
        if batchNorm: x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
    discriminator_outputs = layers.Dense(layers_sizes[-1], activation='softmax', kernel_initializer=kernel)(x)
    discriminator = models.Model(inputs=discriminator_inputs, outputs=discriminator_outputs, name=name)
    return discriminator


def OoD_loss(loss_bkg, loss_OoD):
    def get_loss(y_true, y_pred):
        return tf.keras.activations.sigmoid(loss_bkg - loss_OoD)
        #return -loss_OoD
    return get_loss


def make_noise(bkg_sample):
    range_tuples = zip(np.min(bkg_sample,axis=0), np.max(bkg_sample,axis=0))
    noise = [np.random.default_rng().uniform(val1,val2,len(bkg_sample)) for val1,val2 in range_tuples]
    return np.hstack([array[:,np.newaxis] for array in noise])


def create_model(input_size, autoencoder_layers, beta, lamb, activation='relu', kernel='glorot_uniform'):
    #optimizer = optimizers.RMSprop(learning_rate=1e-4)
    optimizer = optimizers.Adam(lr=1e-6, amsgrad=False)
    discriminator_loss = 'sparse_categorical_crossentropy' ; discriminator_layers = [100, 100, 3]
    #discriminator_loss = 'sparse_categorical_crossentropy' ; discriminator_layers = [100, 100, 4]

    # AE
    Autoencoder = autoencoder_model(input_size, autoencoder_layers, activation, kernel, batchNorm=False)
    AE_bkg_inputs  = layers.Input(shape=input_size)
    AE_OoD_inputs  = layers.Input(shape=input_size)
    AE_bkg_outputs = Autoencoder(AE_bkg_inputs)
    AE_OoD_outputs = Autoencoder(AE_OoD_inputs)
    AE = models.Model([AE_bkg_inputs, AE_OoD_inputs], [AE_bkg_outputs, AE_OoD_outputs], name='AE')
    AE_bkg_MAE = MAE_dist(AE_bkg_inputs, AE_bkg_outputs)
    AE_OoD_MAE = MAE_dist(AE_OoD_inputs, AE_OoD_outputs)
    AE.compile(loss=['mean_absolute_error', OoD_loss(AE_bkg_MAE,AE_OoD_MAE)],
               loss_weights=[1,lamb], optimizer=optimizer, experimental_run_tf_function=False,
               weighted_metrics=[['mean_absolute_error'],['mean_absolute_error']])
    print('\n'); AE.summary()

    # DISCRIMINATOR
    Discriminator = discriminator_model(input_size, discriminator_layers, activation, kernel, batchNorm=False)
    Discriminator.compile(loss=discriminator_loss, optimizer=optimizer, weighted_metrics=['accuracy'])
    print('\n'); Discriminator.summary()

    # AAE
    Discriminator.trainable = False
    AAE_bkg_inputs        = layers.Input(shape=input_size)
    AAE_OoD_inputs        = layers.Input(shape=input_size)
    AAE_all_inputs        = layers.Input(shape=input_size)
    AAE_bkg_outputs       = Autoencoder  (AAE_bkg_inputs )
    AAE_OoD_outputs       = Autoencoder  (AAE_OoD_inputs )
    AAE_all_outputs       = Autoencoder  (AAE_all_inputs )
    discriminator_outputs = Discriminator(AAE_all_outputs)
    AAE_bkg_MAE = MAE_dist(AAE_bkg_inputs, AAE_bkg_outputs)
    AAE_OoD_MAE = MAE_dist(AAE_OoD_inputs, AAE_OoD_outputs)
    AAE = models.Model([AAE_bkg_inputs , AAE_OoD_inputs , AAE_all_inputs        ],
                       [AAE_bkg_outputs, AAE_OoD_outputs, discriminator_outputs ], name='AAE')
    AAE.compile(loss=['mean_absolute_error', OoD_loss(AAE_bkg_MAE,AAE_OoD_MAE), discriminator_loss],
                loss_weights=[1,lamb,beta], optimizer=optimizer, experimental_run_tf_function=False,
                weighted_metrics=[['mean_absolute_error'],['mean_absolute_error'],['accuracy']])
    print('\n'); AAE.summary()

    return AE, Discriminator, AAE


def train_AAE(model, train_generator, n_cycles, batch_size, output_dir, model_out, hist_file, AE_weights, lamb):
    epoch_dict = {'AE':np.full(n_cycles,0), 'Disc':np.full(n_cycles,5), 'AAE':np.full(n_cycles,5)}
    epoch_dict['AE'][0] = 100 ; epoch_dict['Disc'][0] = 5 ; epoch_dict['AAE'][0] = 5

    AE, Discriminator, AAE = model
    sample = train_generator[0]
    bkg_sample, bkg_weight = sample['bkg']['HLVs'], sample['bkg']['weights']
    OoD_sample, OoD_weight = sample['OoD']['HLVs'], sample['OoD']['weights']
    #bkg_sample, bkg_weight = sample['bkg']['constituents'], sample['bkg']['weights']
    #OoD_sample, OoD_weight = sample['OoD']['constituents'], sample['OoD']['weights']
    n_batches = int(np.ceil(len(bkg_sample)/batch_size))
    epoch_counter = 0
    loss_history = {'QCD-AE Loss':[], 'OoD-AE Loss'  :[], 'OE Loss'      :[],
                        'AE Loss' :[],  'Disc Loss'  :[], 'Disc Accuracy':[]}
    AE_weights = output_dir if AE_weights=='' else output_dir+'/'+AE_weights
    if os.path.isfile(AE_weights):
        print('\nLoading pre-trained AE file from:', AE_weights)
        AE.load_weights(AE_weights)
        epoch_dict['AE'][0] = epoch_dict['AE'][1]

    # CYCLE TRAINING
    for cycle in range(n_cycles):

        # AUTOENCODER TRAINING
        print('\n*** CYCLE %d/%d ***'%(cycle+1,n_cycles))
        n_epochs = epoch_dict['AE'][cycle]
        if n_epochs != 0: print('training autoencoder'.upper())
        start_time = time.time()
        for epoch in range(n_epochs):
            print('Epoch %d/%d:'%(epoch+1, n_epochs))
            idx_batch = utils.shuffle(np.arange(n_batches), random_state=None)
            for n in range(len(idx_batch)):
                idx_data = idx_batch[n]*batch_size, min((idx_batch[n]+1)*batch_size, len(bkg_sample))
                bkg_batch_sample = utils.shuffle(bkg_sample[idx_data[0]:idx_data[1]], random_state=0)
                bkg_batch_weight = utils.shuffle(bkg_weight[idx_data[0]:idx_data[1]], random_state=0)
                OoD_batch_sample = utils.shuffle(OoD_sample[idx_data[0]:idx_data[1]], random_state=0)
                OoD_batch_weight = utils.shuffle(OoD_weight[idx_data[0]:idx_data[1]], random_state=0)
                #autoencoder_loss = losses.MeanAbsoluteError()(batch_sample, AE(batch_sample))
                #autoencoder_loss = np.mean(MAE_dist(batch_sample, AE(batch_sample)))
                AE_hist = AE.train_on_batch([bkg_batch_sample, OoD_batch_sample],
                                            [bkg_batch_sample, OoD_batch_sample],
                                            [bkg_batch_weight, OoD_batch_weight])
                #print(AE.metrics_names); print(AE_hist); sys.exit()
                loss_dict = {'QCD-AE Loss':AE_hist[1]}
                if lamb != 0:
                    loss_dict['OoD-AE Loss'] = AE_hist[4]
                    loss_dict[    'OE Loss'] = AE_hist[2]
                loss_dict['AE Loss'] = AE_hist[0]
                print_batch(n, n_batches, loss_dict)
            print('(', '\b' + format(time.time() - start_time, '.1f'), '\b' + 's)')
            if epoch+1 != n_epochs: print(3*'\033[A')
            epoch_counter += 1
            for key in loss_dict: loss_history[key] += [(cycle+1, epoch_counter, loss_dict[key])]
        if cycle == 0 and n_epochs != 0 and not os.path.isfile(AE_weights):
            if AE_hist[0] < 100:
                print('Saving pre-trained AE file to:', AE_weights)
                AE.save_weights(output_dir+'/'+'AE_weights.h5')
            else: sys.exit()
        #for key,val in loss_history.items(): print(key, val)
        #sys.exit()

        # DISCRIMINATOR TRAINING
        n_epochs = epoch_dict['Disc'][cycle]
        if n_epochs != 0: print('training discriminator'.upper())
        start_time = time.time()
        Discriminator.trainable = True
        for epoch in range(n_epochs):
            print('Epoch %d/%d:'%(epoch+1, n_epochs))
            idx_batch = utils.shuffle(np.arange(n_batches), random_state=None)
            for n in range(len(idx_batch)):
                idx_data = idx_batch[n]*batch_size, min((idx_batch[n]+1)*batch_size, len(bkg_sample))
                bkg_batch_sample = utils.shuffle(bkg_sample[idx_data[0]:idx_data[1]], random_state=0)
                bkg_batch_weight = utils.shuffle(bkg_weight[idx_data[0]:idx_data[1]], random_state=0)
                OoD_batch_sample = utils.shuffle(OoD_sample[idx_data[0]:idx_data[1]], random_state=0)
                OoD_batch_weight = utils.shuffle(OoD_weight[idx_data[0]:idx_data[1]], random_state=0)
                bkg_batch_fake, OoD_batch_fake = AE([bkg_batch_sample, OoD_batch_sample])

                batch_sample = np.concatenate([bkg_batch_sample, bkg_batch_fake  , OoD_batch_sample], axis=0)
                batch_weight = np.concatenate([bkg_batch_weight, bkg_batch_weight, OoD_batch_weight], axis=0)
                batch_labels = np.concatenate([np.full_like(bkg_batch_weight,0), np.full_like(bkg_batch_weight,1),
                                               np.full_like(OoD_batch_weight,2)])
                #batch_sample = np.concatenate([bkg_batch_sample, bkg_batch_fake ,
                #                               OoD_batch_sample, OoD_batch_fake], axis=0)
                #batch_weight = np.concatenate([bkg_batch_weight, bkg_batch_weight ,
                #                               OoD_batch_weight, OoD_batch_weight], axis=0)
                #batch_labels = np.concatenate([np.full_like(bkg_batch_weight,0), np.full_like(bkg_batch_weight,1),
                #                               np.full_like(OoD_batch_weight,2), np.full_like(OoD_batch_weight,3)])

                batch_sample = utils.shuffle(batch_sample, random_state=0)
                batch_labels = utils.shuffle(batch_labels, random_state=0)
                batch_weight = utils.shuffle(batch_weight, random_state=0)
                Discriminator_hist = Discriminator.train_on_batch(batch_sample, batch_labels, batch_weight,
                                                                  reset_metrics=False)
                #print(Discriminator.metrics_names); print(Discriminator_hist); sys.exit()
                loss_dict = {'Disc Loss':Discriminator_hist[0], 'Disc Accuracy':Discriminator_hist[1]}
                print_batch(n, n_batches, loss_dict)
            print('(', '\b' + format(time.time() - start_time, '.1f'), '\b' + 's)')
            if epoch+1 != n_epochs: print(3*'\033[A')
            epoch_counter += 1
            for key in loss_dict: loss_history[key] += [(cycle+1, epoch_counter, loss_dict[key])]
            Discriminator.reset_metrics()
        #for key,val in loss_history.items(): print(key, val)
        #sys.exit()

        # AAE TRAINING
        n_epochs = epoch_dict['AAE'][cycle]
        if n_epochs != 0: print('training AAE'.upper())
        start_time = time.time()
        Discriminator.trainable = False
        for epoch in range(n_epochs):
            print('Epoch %d/%d:'%(epoch+1, n_epochs))
            idx_batch = utils.shuffle(np.arange(n_batches), random_state=None)
            for n in range(len(idx_batch)):
                idx_data = idx_batch[n]*batch_size, min((idx_batch[n]+1)*batch_size, len(bkg_sample))
                bkg_batch_sample = utils.shuffle(bkg_sample[idx_data[0]:idx_data[1]], random_state=0)
                OoD_batch_sample = utils.shuffle(OoD_sample[idx_data[0]:idx_data[1]], random_state=0)
                bkg_batch_weight = utils.shuffle(bkg_weight[idx_data[0]:idx_data[1]], random_state=0)
                OoD_batch_weight = utils.shuffle(OoD_weight[idx_data[0]:idx_data[1]], random_state=0)
                all_batch_sample = np.concatenate([bkg_batch_sample, OoD_batch_sample], axis=0)
                all_batch_weight = np.concatenate([bkg_batch_weight, OoD_batch_weight], axis=0)

                all_batch_labels = np.concatenate([np.full_like(bkg_batch_weight, 0),
                                                   np.full_like(OoD_batch_weight, 1)])
                #all_batch_labels = np.concatenate([np.full_like(bkg_batch_weight, 0),
                #                                   np.full_like(OoD_batch_weight, 3)])

                bkg_batch_sample = np.concatenate([bkg_batch_sample, bkg_batch_sample], axis=0)
                bkg_batch_weight = np.concatenate([bkg_batch_weight, bkg_batch_weight], axis=0)
                OoD_batch_sample = np.concatenate([OoD_batch_sample, OoD_batch_sample], axis=0)
                OoD_batch_weight = np.concatenate([OoD_batch_weight, OoD_batch_weight], axis=0)
                AAE_hist = AAE.train_on_batch([bkg_batch_sample, OoD_batch_sample, all_batch_sample],
                                              [bkg_batch_sample, OoD_batch_sample, all_batch_labels],
                                              [bkg_batch_weight, OoD_batch_weight, all_batch_weight],
                                              reset_metrics=False)
                #print(AAE.metrics_names); print(AAE_hist); sys.exit()
                # Discriminator loss and accuracy
                bkg_pred     , OoD_pred      = Discriminator(bkg_batch_sample), Discriminator(OoD_batch_sample)
                bkg_fake     , OoD_fake      = AE([bkg_batch_sample, OoD_batch_sample])
                bkg_fake_pred, OoD_fake_pred = Discriminator(bkg_fake), Discriminator(OoD_fake)

                y_true  = np.concatenate([np.full_like(bkg_batch_weight,0), np.full_like(bkg_batch_weight,1),
                                          np.full_like(OoD_batch_weight,2)])
                y_pred  = tf.convert_to_tensor(np.concatenate([bkg_pred, bkg_fake_pred, OoD_pred], axis=0))
                weights = np.concatenate([bkg_batch_weight, bkg_batch_weight, OoD_batch_weight], axis=0)
                #y_true  = np.concatenate([np.full_like(bkg_batch_weight,0), np.full_like(bkg_batch_weight,1),
                #                          np.full_like(OoD_batch_weight,2), np.full_like(OoD_batch_weight,3)])
                #y_pred  = tf.convert_to_tensor(np.concatenate([bkg_pred, bkg_fake_pred, OoD_pred, OoD_fake_pred], axis=0))
                #weights = np.concatenate([bkg_batch_weight, bkg_batch_weight, OoD_batch_weight, OoD_batch_weight], axis=0)

                disc_loss     = losses.SparseCategoricalCrossentropy()(y_true, y_pred, weights).numpy()
                disc_accuracy = get_accuracy(y_true, y_pred, weights)
                #reconstructed, y_pred = AAE(batch_sample)
                #AE_loss = losses.MeanAbsoluteError()(batch_sample, reconstructed)
                #AE_loss = np.mean(MAE_dist(batch_sample, reconstructed))
                loss_dict = {'QCD-AE Loss':AAE_hist[1]}
                if lamb != 0:
                    loss_dict['OoD-AE Loss'] = AAE_hist[5]
                    loss_dict[    'OE Loss'] = AAE_hist[2]
                loss_dict['AE Loss'   ] = AAE_hist[1] + lamb*AAE_hist[2]
                loss_dict['Disc Loss'    ] = disc_loss
                loss_dict['Disc Accuracy'] = disc_accuracy
                # Values pertaining to the autoencoder training (i.e. attempt to fool the discriminator)
                AAE_dict = {'AAE Loss':AAE_hist[0], 'D_Loss':AAE_hist[3], 'D_Accuracy':AAE_hist[6]}
                print_batch(n, n_batches, {**loss_dict, **AAE_dict})
            print('(', '\b' + format(time.time() - start_time, '.1f'), '\b' + 's)')
            if epoch+1 != n_epochs: print(3*'\033[A')
            epoch_counter += 1
            for key in loss_dict: loss_history[key] += [(cycle+1, epoch_counter, loss_dict[key])]
            AAE.reset_metrics()
    for key,val in loss_history.items():
        print()
        print(format(key,'22'), val)
    pickle.dump(loss_history, open(hist_file,'wb'))
    AAE.save_weights(output_dir+'/'+'AAE.h5')


def print_batch(n, n_batches, loss_dict):
    if (n+1)%10 == 0 or n+1 == n_batches:
        print('Batch %3d/%d'%(n+1, n_batches), end='  -->  ', flush=True)
        for key,val in loss_dict.items():
            condition = (n+1 != n_batches)
            val_format = format(100*val,'4.1f')+'%' if 'Accuracy' in key else format(val,'4.3e')
            end, flush = ('\r', False)  if condition and key==list(loss_dict.keys())[-1] else ('  ', True)
            print(key + ': ' + val_format, end=end, flush=True)
