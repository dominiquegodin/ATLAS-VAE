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


def reinitialize(model):
    for l in model.layers:
        if hasattr(l,"kernel_initializer"):
            l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
        if hasattr(l,"bias_initializer"):
            l.bias.assign(l.bias_initializer(tf.shape(l.bias)))
        if hasattr(l,"recurrent_initializer"):
            l.recurrent_kernel.assign(l.recurrent_initializer(tf.shape(l.recurrent_kernel)))
    return model


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
    discriminator_inputs = layers.Input(shape=input_size, name='test'); x = discriminator_inputs
    for size in layers_sizes[:-1]:
        x = layers.Dense(size, kernel_initializer=kernel)(x)
        if batchNorm: x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
    discriminator_outputs = layers.Dense(layers_sizes[-1], activation='softmax', kernel_initializer=kernel)(x)
    discriminator = models.Model(inputs=discriminator_inputs, outputs=discriminator_outputs, name=name)
    return discriminator


def get_OoD_loss(loss_bkg, loss_OoD):
    def get_loss(y_true, y_pred):
        return tf.keras.activations.sigmoid(loss_bkg - loss_OoD)
    return get_loss


def create_model(input_size, autoencoder_layers, activation='relu', kernel='glorot_uniform', beta=5, epsilon=5):
    #optimizer = optimizers.RMSprop(learning_rate=1e-4)
    optimizer = optimizers.Adam(lr=1e-6, amsgrad=False)
    #discriminator_loss = 'binary_crossentropy'             ; discriminator_layers = [100,100,1]
    discriminator_loss = 'sparse_categorical_crossentropy' ; discriminator_layers = [100,100,2]

    Autoencoder = autoencoder_model(input_size, autoencoder_layers, activation, kernel, batchNorm=False)
    #Autoencoder.compile(loss='mean_absolute_error', optimizer=optimizer, weighted_metrics=['mean_absolute_error'])
    #print('\n'); Autoencoder.summary()

    AE_inputs_bkg = layers.Input(shape=input_size)
    AE_inputs_OoD = layers.Input(shape=input_size)
    AE_outputs_bkg = Autoencoder(AE_inputs_bkg)
    AE_outputs_OoD = Autoencoder(AE_inputs_OoD)
    AE = models.Model([AE_inputs_bkg, AE_inputs_OoD], [AE_outputs_bkg, AE_outputs_OoD], name='AE')
    loss_bkg = MAE_dist(AE_inputs_bkg, AE_outputs_bkg)
    loss_OoD = MAE_dist(AE_inputs_OoD, AE_outputs_OoD)
    OoD_loss = get_OoD_loss(loss_bkg, loss_OoD)
    AE.compile(loss=['mean_absolute_error', OoD_loss], loss_weights=[1,beta], experimental_run_tf_function=False,
               optimizer=optimizer, weighted_metrics=[['mean_absolute_error'],['mean_absolute_error']])
    print('\n'); AE.summary()

    Discriminator = discriminator_model(input_size, discriminator_layers, activation, kernel, batchNorm=False)
    Discriminator.compile(loss=discriminator_loss, optimizer=optimizer, weighted_metrics=['accuracy'])
    print('\n'); Discriminator.summary()

    AAE_inputs = layers.Input(shape=input_size)
    autoencoder_outputs = Autoencoder(AAE_inputs)
    Discriminator.trainable = False
    discriminator_outputs = Discriminator(autoencoder_outputs)
    AAE = models.Model(AAE_inputs, [autoencoder_outputs, discriminator_outputs], name='AAE')
    AAE.compile(loss=['mean_absolute_error', discriminator_loss], loss_weights=[1,epsilon],
                optimizer=optimizer, weighted_metrics=[['mean_absolute_error'],['accuracy']])
    print('\n'); AAE.summary()

    #return Autoencoder, Discriminator, AAE
    return AE, Discriminator, AAE


def train_AAE(model, train_generator, n_cycles, batch_size, output_dir, model_out,
              AE_pretrained='', beta=5, epsilon=5):
    epoch_dict = {'AE':np.full(n_cycles,5), 'Discriminator':np.full(n_cycles,5), 'AAE':np.full(n_cycles,5)}
    epoch_dict['AE'][0] = 10 ; epoch_dict['Discriminator'][0] = 0 ; epoch_dict['AAE'][0] = 0

    Autoencoder, Discriminator, AAE = model
    sample = train_generator[0]
    train_data, train_weights = sample['bkg']['HLVs'], sample['bkg']['weights']
    OoD_data  , OoD_weights   = sample['OoD']['HLVs'], sample['OoD']['weights']
    #train_data, train_weights = sample['bkg']['constituents'], sample['bkg']['weights']
    #OoD_data  , OoD_weights   = sample['OoD']['constituents'], sample['OoD']['weights']
    n_batches = int(np.ceil(len(train_data)/batch_size))
    epoch_counter = 0
    loss_history = {'AAE Loss'          :[], 'Autoencoder Loss'      :[],
                    'MAE Loss'          :[], 'OoD Loss'              :[],
                    'Discriminator Loss':[], 'Discriminator Accuracy':[]}

    output_dir = output_dir[0:output_dir.rfind('/')]
    AE_pretrained = output_dir+'/'+AE_pretrained
    if os.path.isfile(AE_pretrained):
        print('\nLoading pre-trained AE file from:', AE_pretrained)
        Autoencoder.load_weights(AE_pretrained)
        epoch_dict['AE'][0] = epoch_dict['AE'][1]

    for cycle in range(n_cycles):
        # AUTOENCODER TRAINING
        print('\n*** CYCLE %d/%d ***'%(cycle+1,n_cycles))
        n_epochs = epoch_dict['AE'][cycle]
        if n_epochs != 0: print('training autoencoder'.upper())
        else            : continue
        start_time = time.time()
        for epoch in range(n_epochs):
            print('Epoch %d/%d:'%(epoch+1, n_epochs))
            idx = utils.shuffle(np.arange(n_batches), random_state=None)
            for n in range(len(idx)):
                idx_data      = idx[n]*batch_size, min((idx[n]+1)*batch_size, len(train_data))
                bkg_batch_sample  = utils.shuffle(train_data   [idx_data[0]:idx_data[1]], random_state=0)
                bkg_batch_weights = utils.shuffle(train_weights[idx_data[0]:idx_data[1]], random_state=0)
                OoD_batch_sample  = utils.shuffle(  OoD_data   [idx_data[0]:idx_data[1]], random_state=0)
                OoD_batch_weights = utils.shuffle(  OoD_weights[idx_data[0]:idx_data[1]], random_state=0)
                #autoencoder_loss = losses.MeanAbsoluteError()(batch_sample, Autoencoder(batch_sample))
                #autoencoder_loss = np.mean(MAE_dist(batch_sample, Autoencoder(batch_sample)))
                #autoencoder_hist = Autoencoder.train_on_batch(batch_sample, batch_sample, batch_weights)
                autoencoder_hist = Autoencoder.train_on_batch([bkg_batch_sample , OoD_batch_sample ],
                                                              [bkg_batch_sample , OoD_batch_sample ],
                                                              [bkg_batch_weights, OoD_batch_weights])
                #print(Autoencoder.metrics_names) ; print(autoencoder_hist) ; sys.exit()
                loss_dict = {'Autoencoder Loss':autoencoder_hist[0],
                             'MAE Loss'        :autoencoder_hist[1],
                             'OoD Loss'        :autoencoder_hist[2]}
                print_batch(n, n_batches, loss_dict)
            print('(', '\b' + format(time.time() - start_time, '.1f'), '\b' + 's)')
            if epoch+1 != n_epochs: print(3*'\033[A')
            epoch_counter += 1
            for key in loss_dict: loss_history[key] += [(cycle+1, epoch_counter, loss_dict[key])]
        if cycle == 0 and not os.path.isfile(AE_pretrained):
            if autoencoder_hist[0] < 100:
                print('\nSaving pre-trained AE file to:', AE_pretrained)
                Autoencoder.save_weights(output_dir+'/'+'AE_pretrained.h5')
            else: sys.exit()
        #for key,val in loss_history.items(): print(key, val)
        #sys.exit()

        # DISCRIMINATOR TRAINING
        n_epochs = epoch_dict['Discriminator'][cycle]
        if n_epochs != 0: print('training discriminator'.upper())
        else            : continue
        start_time = time.time()
        Discriminator.trainable = True
        for epoch in range(n_epochs):
            print('Epoch %d/%d:'%(epoch+1, n_epochs))
            idx = utils.shuffle(np.arange(n_batches))
            for n in range(len(idx)):
                idx_data  = idx[n]*batch_size, min((idx[n]+1)*batch_size, len(train_data))
                if False:
                    #rng      = np.random.default_rng()
                    #idx_real = rng.choice(np.arange(idx_data[0],idx_data[1]), np.diff(idx_data)//2, replace=False)
                    #idx_fake = list(set(np.arange(idx_data[0],idx_data[1])) - set(idx_real))
                    idx_real = utils.shuffle(np.arange(idx_data[0],idx_data[1]), random_state=None)
                    idx_fake = utils.shuffle(np.arange(idx_data[0],idx_data[1]), random_state=None)
                else:
                    idx_real = np.arange(idx_data[0],idx_data[1])
                    idx_fake = np.arange(idx_data[0],idx_data[1])
                data_real     =              np.take(train_data, idx_real, axis=0)
                data_fake, _  = Autoencoder([np.take(train_data, idx_fake, axis=0),
                                             np.take(train_data, idx_fake, axis=0)])
                weights_real  = np.take(train_weights, idx_real, axis=0)
                weights_fake  = np.take(train_weights, idx_fake, axis=0)
                batch_sample  = np.concatenate([   data_real,    data_fake], axis=0)
                batch_weights = np.concatenate([weights_real, weights_fake], axis=0)
                batch_labels  = np.concatenate([np.ones_like(weights_fake), np.zeros_like(weights_real)])
                batch_sample  = utils.shuffle(batch_sample , random_state=0)
                batch_weights = utils.shuffle(batch_weights, random_state=0)
                batch_labels  = utils.shuffle(batch_labels , random_state=0)
                Discriminator_hist = Discriminator.train_on_batch(batch_sample, batch_labels,
                                                                  batch_weights, reset_metrics=True)
                loss_dict = {'Discriminator Loss'    :Discriminator_hist[0],
                             'Discriminator Accuracy':Discriminator_hist[1]}
                print_batch(n, n_batches, loss_dict)
            print('(', '\b' + format(time.time() - start_time, '.1f'), '\b' + 's)')
            if epoch+1 != n_epochs: print(3*'\033[A')
            epoch_counter += 1
            for key in loss_dict: loss_history[key] += [(cycle+1, epoch_counter, loss_dict[key])]
            #Discriminator.reset_metrics()
        # AAE TRAINING
        n_epochs = epoch_dict['AAE'][cycle]
        if n_epochs != 0: print('training AAE'.upper())
        else            : continue
        start_time = time.time()
        Discriminator.trainable = False
        for epoch in range(n_epochs):
            print('Epoch %d/%d:'%(epoch+1, n_epochs))
            idx = utils.shuffle(np.arange(n_batches))
            for n in range(len(idx)):
                idx_data  = idx[n]*batch_size, min((idx[n]+1)*batch_size, len(train_data))
                batch_sample  = train_data   [idx_data[0]:idx_data[1]]
                batch_weights = train_weights[idx_data[0]:idx_data[1]]
                batch_labels  = np.ones_like(batch_weights)
                #y_pred = Discriminator(Autoencoder(batch_sample))
                #print( losses.SparseCategoricalCrossentropy()(batch_labels, y_pred, batch_weights) )
                AAE_hist = AAE.train_on_batch(batch_sample, [batch_sample, batch_labels],
                                              [batch_weights, batch_weights], reset_metrics=True)
                # Discriminator loss and accuracy
                y_pred_real = Discriminator(            batch_sample )
                y_pred_fake = Discriminator(Autoencoder([batch_sample, batch_sample])[0])
                y_true  = np.concatenate([np.ones_like(batch_weights), np.zeros_like(batch_weights)])
                y_pred  = tf.convert_to_tensor(np.concatenate([y_pred_real, y_pred_fake], axis=0))
                weights = np.concatenate([batch_weights, batch_weights], axis=0)
                discriminator_loss = losses.SparseCategoricalCrossentropy()(y_true, y_pred, weights).numpy()
                discriminator_accuracy = get_accuracy(y_true, y_pred, weights)
                #reconstructed, y_pred = AAE(batch_sample)
                #autoencoder_loss = losses.MeanAbsoluteError()(batch_sample, reconstructed)
                #autoencoder_loss = np.mean(MAE_dist(batch_sample, reconstructed))
                loss_dict = {'Autoencoder Loss'      :AAE_hist[1]           ,
                             'Discriminator Loss'    :discriminator_loss    ,
                             'Discriminator Accuracy':discriminator_accuracy,
                             'AAE Loss'              :AAE_hist[0]           }
                print_batch(n, n_batches, {**loss_dict, 'D_Loss':epsilon*AAE_hist[2], 'D_Accuracy':AAE_hist[3]})
            print('(', '\b' + format(time.time() - start_time, '.1f'), '\b' + 's)')
            if epoch+1 != n_epochs: print(3*'\033[A')
            epoch_counter += 1
            for key in loss_dict: loss_history[key] += [(cycle+1, epoch_counter, loss_dict[key])]
            #AAE.reset_metrics()
    for key,val in loss_history.items():
        print()
        print(format(key,'22'), val)
    #Autoencoder  .save_weights(output_dir+'/'+'autoencodeur.h5')
    #Discriminator.save_weights(output_dir+'/'+'discriminator.h5')
    AAE.save_weights(output_dir+'/'+'AAE.h5')


def print_batch(n, n_batches, loss_dict):
    if (n+1)%10 == 0 or n+1 == n_batches:
        print('Batch %3d/%d'%(n+1, n_batches), end='  -->  ', flush=True)
        for key,val in loss_dict.items():
            condition = (n+1 != n_batches)
            val_format = format(100*val,'4.1f')+'%' if 'Accuracy' in key else format(val,'4.3e')
            end, flush = ('\r', False)  if condition and key==list(loss_dict.keys())[-1] else ('  ', True)
            print(key + ': ' + val_format, end=end, flush=True)
