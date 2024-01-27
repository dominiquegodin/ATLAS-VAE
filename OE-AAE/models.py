import numpy      as np
import tensorflow as tf
import time, sys, os, pickle
from   tensorflow.keras import layers, models, Input, initializers, regularizers, optimizers, callbacks
#from   tensorflow.keras.layers import Flatten, Dense, concatenate, Dropout, BatchNormalization, LeakyReLU
#from   tensorflow.keras        import Input, regularizers, models, callbacks, optimizers


class Sampling(layers.Layer):
    """ Using (z_mean, z_log_var) to sample z """
    def call(self, inputs, seed):
        z_mean, z_log_var = inputs
        sigma = tf.exp(z_log_var/2)
        sigma = clip_values(sigma, max_val=1e6)
        tf.random.set_seed(seed)
        #return tf.random.normal(tf.shape(z_mean), seed=seed)*sigma + z_mean
        return tf.random.normal(tf.shape(z_mean), mean=z_mean, stddev=sigma, seed=seed)


class Encoder(layers.Layer):
    """ Mapping inputs to (z_mean, z_log_var, z) """
    def __init__(self, FC_layers, activation, seed=None, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.FC_layers = FC_layers[:-1]; self.seed = seed
        self.denses = [layers.Dense(n_neurons, activation=activation,
                                    kernel_initializer=initializers.he_normal(),
                                    bias_initializer=tf.random.normal)
                       for n_neurons in self.FC_layers]
        self.dense_mean    = layers.Dense(FC_layers[-1])
        self.dense_log_var = layers.Dense(FC_layers[-1])
        self.sampling = Sampling()
    def call(self, x):
        for dense in self.denses:
            x = dense(x)
        z_mean    = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z         = self.sampling([z_mean, z_log_var], seed=self.seed)
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """ Converting the encoded digit vector z back to input space """
    def __init__(self, FC_layers, output_dim, activation, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.FC_layers = FC_layers[:-1][::-1]
        self.denses = [layers.Dense(n_neurons, activation=activation,
                                    kernel_initializer=initializers.he_normal(),
                                    bias_initializer=tf.random.normal)
                       for n_neurons in self.FC_layers]
        self.dense_output = layers.Dense(output_dim, activation='linear')
    def call(self, x):
        for dense in self.denses:
            x = dense(x)
        return self.dense_output(x)


class VariationalAutoEncoder(models.Model):
    """ Combining the encoder and decoder into a VAE model for training """
    def __init__(self, FC_layers, input_dim, activation='relu', seed=None, name="autoencoder", **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.encoder   = Encoder(FC_layers, activation, seed=seed)
        self.decoder   = Decoder(FC_layers, input_dim, activation)
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        """ Clipping bad reconstruction """
        reconstructed = clip_values(reconstructed, max_val=1e6)
        """ Adding KLD regularization loss """
        self.add_loss( get_KLD_loss(z_mean, z_log_var) )
        return reconstructed


def get_reconstruction_loss(batch_input, batch_output, OE_type):
    if OE_type == 'MSE' or OE_type == 'MSE-margin':
        return tf.keras.losses.MSE(batch_input, batch_output)
    if OE_type == 'MAE' or OE_type == 'MAE-margin' or OE_type == 'KLD':
        #mape = tf.keras.losses.MeanAbsolutePercentageError()
        #return mape(batch_input, batch_output)/tf.constant(100.)
        return tf.keras.losses.MAE(batch_input, batch_output)


def get_KLD_loss(z_mean, z_log_var):
    """ Calculating latent layer KLD loss """
    z_exp_log_var = tf.exp(z_log_var)
    z_exp_log_var = clip_values(z_exp_log_var, max_val=1e6)
    return -tf.reduce_mean(1 + z_log_var - z_exp_log_var - tf.square(z_mean), axis=-1)/2


def get_OE_loss(vae, batch_X, batch_X_OE, OE_type, margin):
    """ Calculating outlier KLD loss """
    if OE_type == 'KLD':
        z_mean   , z_log_var   , _ = vae.encoder(batch_X   )
        z_mean_OE, z_log_var_OE, _ = vae.encoder(batch_X_OE)
        loss_KLD    = get_KLD_loss(z_mean   , z_log_var   )
        loss_KLD_OE = get_KLD_loss(z_mean_OE, z_log_var_OE)
        #return tf.keras.activations.relu(loss_KLD*factor - loss_KLD_OE)
        return tf.keras.activations.relu(loss_KLD - loss_KLD_OE + margin)
    """ Calculating outlier MSE loss """
    reconstructed    = vae(batch_X   )
    reconstructed_OE = vae(batch_X_OE)
    loss_MSE    = get_reconstruction_loss(batch_X   , reconstructed   , OE_type)
    loss_MSE_OE = get_reconstruction_loss(batch_X_OE, reconstructed_OE, OE_type)
    if OE_type == 'MSE' or OE_type == 'MAE':
        return tf.keras.activations.sigmoid(loss_MSE - loss_MSE_OE         )
    if OE_type == 'MSE-margin' or OE_type == 'MAE-margin':
        return tf.keras.activations.relu   (loss_MSE - loss_MSE_OE + margin)


def get_losses(vae, data, OE_type, beta, lamb, margin):
    bkg_sample , OoD_sample  = data
    if 'constituents' in bkg_sample and 'HLVs' in bkg_sample:
        bkg_batch_X = np.hstack([bkg_sample['constituents'], bkg_sample['HLVs']])
        OoD_batch_X = np.hstack([OoD_sample['constituents'], OoD_sample['HLVs']])
    elif 'constituents' in bkg_sample:
        bkg_batch_X = bkg_sample['constituents']
        OoD_batch_X = OoD_sample['constituents']
    elif 'HLVs' in bkg_sample:
        bkg_batch_X = bkg_sample['HLVs']
        OoD_batch_X = OoD_sample['HLVs']
    bkg_weights = bkg_sample['weights']
    OoD_weights = OoD_sample['weights']
    """ MSE reconstruction loss """
    reconstructed = vae(bkg_batch_X)
    loss_MSE = get_reconstruction_loss(bkg_batch_X, reconstructed, OE_type)
    loss_MSE = tf.math.multiply(loss_MSE, bkg_weights)
    """ KLD regularization loss """
    loss_KLD = sum(vae.losses)
    loss_KLD = tf.math.multiply(loss_KLD, bkg_weights) * beta
    """ OE decorrelation loss   """
    loss_OoD = get_OE_loss(vae, bkg_batch_X, OoD_batch_X, OE_type, margin)
    loss_OoD = tf.math.multiply(loss_OoD, OoD_weights) * lamb
    """ Total training loss     """
    return loss_MSE, loss_KLD, loss_OoD, loss_MSE+loss_KLD+loss_OoD


def train_model(vae, train_sample, valid_sample, OE_type='KLD', n_epochs=1, batch_size=5000,
                beta=0, lamb=0, margin=0, lr=1e-3, hist_file=None, model_in=None, model_out=None):
    """ Using subclassing Tensoflow API to build model """
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    try   : train_dataset_len = tf.data.experimental.cardinality(train_sample).numpy()
    except: train_dataset_len = len(train_sample)
    try   : valid_dataset_len = tf.data.experimental.cardinality(valid_sample).numpy()
    except: valid_dataset_len = len(valid_sample)
    if train_dataset_len == 1:
        try   : train_sample = [train_sample[0]]
        except: pass
    if valid_dataset_len == 1:
        try   : valid_sample = [valid_sample[0]]
        except: pass
    metric_MSE   = tf.keras.metrics.Mean()
    metric_KLD   = tf.keras.metrics.Mean()
    metric_OE    = tf.keras.metrics.Mean()
    metric_train = tf.keras.metrics.Mean()
    metric_valid = tf.keras.metrics.Mean()
    """ Iterating over epochs """
    print('STARTING TRAINING (generator '+('OFF' if train_dataset_len==1 else 'ON') +')')
    history = {'MSE':[]}
    if beta != 0: history['KLD'] = []
    if lamb != 0: history['OE' ] = []
    history = {**history, 'Train loss':[], 'Valid loss':[]}
    if hist_file is not None and os.path.isfile(hist_file) and os.path.isfile(model_in):
        history = pickle.load(open(hist_file, 'rb'))
    total_batches = 0; count = 0
    for epoch in range(n_epochs):
        sum_batches = 0; start_time = time.time()
        print('\nEpoch %d/%d:'%(epoch+1,n_epochs))
        metric_MSE  .reset_states()
        metric_KLD  .reset_states()
        metric_OE   .reset_states()
        metric_train.reset_states()
        metric_valid.reset_states()
        """ Iterating over loads """
        for train_idx, train_data in enumerate(train_sample):
            train_size = len(train_data[0]['weights'])
            n_batches  = int(np.ceil(train_size/batch_size))
            if epoch == 0: total_batches += n_batches
            """ Iterating over batches """
            for batch_idx in range(n_batches):
                sum_batches += 1
                idx  = batch_idx*batch_size, min((batch_idx+1)*batch_size, train_size)
                data = [{key:sample[key][idx[0]:idx[1]] for key in sample} for sample in train_data]
                with tf.GradientTape() as tape:
                    loss_MSE, loss_KLD, loss_OE, loss_train = get_losses(vae, data, OE_type, beta, lamb, margin)
                grads = tape.gradient(loss_train, vae.trainable_weights)
                """ Clipping bad gradients """
                grads = [clip_values(val, max_val=1e6) for val in grads]
                optimizer.apply_gradients( zip(grads, vae.trainable_weights) )
                metric_MSE  (loss_MSE  )
                metric_KLD  (loss_KLD  )
                metric_OE   (loss_OE   )
                metric_train(loss_train)
                if sum_batches % 10 == 0 or batch_idx+1 == n_batches:
                    losses = {'MSE':metric_MSE.result()}
                    if beta != 0: losses['KLD'] = metric_KLD.result()
                    if lamb != 0: losses['OE']  = metric_OE .result()
                    losses['Train loss'] = metric_train.result()
                    print('Batch %4d/%d: mean losses'%(sum_batches, total_batches), end='  -->  ', flush=True)
                    for key,val in losses.items():
                        condition = train_idx+1 != train_dataset_len or batch_idx+1 != n_batches
                        end, flush = ('\r', False)  if condition and key == 'Train loss' else ('  ', True)
                        print(key + ' = ' + format(val,'4.3e'), end=end, flush=True)
        for valid_data in valid_sample:
            valid_loss = get_losses(vae, valid_data, OE_type, beta, lamb, margin)[-1]
            metric_valid(valid_loss)
        losses['Valid loss'] = metric_valid.result()
        print('Valid loss = ' + format(losses['Valid loss'],'4.3e'), end='  ', flush=True)
        print('(', '\b' + format(time.time() - start_time, '.1f'), '\b' + 's)')
        for key in history: history[key] += [losses[key].numpy() if key in losses else 0]
        if hist_file is not None: pickle.dump(history, open(hist_file,'wb'))
        if epoch > 0:
            optimizer, count = model_checkpoint(vae, optimizer, history, model_out, count)
            if count is None: break


def model_checkpoint(vae, optimizer, history, model_out, count, metric='Train loss',
                     patience=3, factor=2, min_delta=1e-3, min_lr=1e-4):
    if history[metric][-1] < np.min(history[metric][:-1]) - min_delta:
        print(metric, 'improved from', format(np.min(history[metric][:-1]),'4.2f'), end=' to ', flush=True)
        print(format(history[metric][-1],'4.2f'), ' -->  saving model weights to',  model_out)
        vae.save_weights(model_out); count = 0
    elif history[metric][-1] > np.min(history[metric][-(patience+1):-1]) - min_delta:
        count += 1
    if count >= patience:
        print('No improvement for', count, 'epochs', end='  -->  ', flush=True)
        if optimizer.learning_rate < min_lr:
            print('terminating training')
            count = None
        else:
            print('reducing learning rate from', optimizer.learning_rate.numpy(), end=' to ', flush=True)
            optimizer.learning_rate = optimizer.learning_rate/factor
            print(optimizer.learning_rate.numpy())
            count = 0
    return optimizer, count


def clip_values(array, max_val=np.inf):
    array = tf.where(tf.math.is_finite(array), array, 0)
    array = tf.clip_by_value(array, -max_val, max_val)
    return array


def find_nearest(value, array):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]





def multi_FCN(sample, n_classes, FCN_neurons, l2, dropout, batchNorm=False):
    regularizer = regularizers.l2(l2)
    input_dict  = {key:Input(shape=sample[key].shape[1:], name=key) for key in sample}
    output_list = []
    # CONSTITUENTS
    if 'constituents' in sample:
        outputs = layers.Flatten()(input_dict['constituents'])
        for n_neurons in [200]:#[200, 200]:
            outputs = layers.Dense(n_neurons, kernel_regularizer=regularizer) (outputs)
            if batchNorm: outputs = layers.BatchNormalization()               (outputs)
            outputs = layers.LeakyReLU(alpha=0)                               (outputs)
            outputs = layers.Dropout(dropout)                                 (outputs)
        output_list += [outputs]
    # HLVs
    if 'HLVs' in sample:
        outputs = layers.Flatten()(input_dict['HLVs'])
        for n_neurons in [200]:#[200, 200]:
            outputs = layers.Dense(n_neurons, kernel_regularizer=regularizer) (outputs)
            if batchNorm: outputs = layers.BatchNormalization()               (outputs)
            outputs = layers.LeakyReLU(alpha=0)                               (outputs)
            outputs = layers.Dropout(dropout)                                 (outputs)
        output_list += [outputs]
    #CONCATENATION TO DOWNSTREAM FCN
    outputs = layers.concatenate(output_list) if len(output_list)>1 else output_list[0]
    for n_neurons in FCN_neurons:
        outputs = layers.Dense(n_neurons, kernel_regularizer=regularizer)     (outputs)
        if batchNorm: outputs = layers.BatchNormalization()                   (outputs)
        outputs = layers.LeakyReLU(alpha=0)                                   (outputs)
        outputs = layers.Dropout(dropout)                                     (outputs)
    outputs = layers.Dense(n_classes, activation='softmax', dtype='float32')  (outputs)
    return models.Model(inputs = list(input_dict.values()), outputs = outputs)


def QCD_tagger(sample, n_classes=2, FCN_neurons=[200,200], l2=1e-7, dropout=0.1, n_gpus=1):
    devices = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3','/gpu:4', '/gpu:5', '/gpu:6', '/gpu:7']
    tf.debugging.set_log_device_placement(False)
    #strategy = tf.distribute.MirroredStrategy(devices=devices[:n_gpus])
    #with strategy.scope():
    #    model = multi_FCN(sample, n_classes, FCN_neurons, l2, dropout)
    #    print('\nNEURAL NETWORK ARCHITECTURE'); model.summary()
    #    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model = multi_FCN(sample, n_classes, FCN_neurons, l2, dropout)
    print('\nNEURAL NETWORK ARCHITECTURE'); model.summary()
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def call_backs(model_out, patience, metric):
    calls  = [callbacks.ModelCheckpoint(model_out, save_best_only=True, monitor=metric, verbose=1)]
    calls += [callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_delta=1e-6, monitor=metric, verbose=1)]
    calls += [callbacks.EarlyStopping(patience=patience, restore_best_weights=True,
                                      min_delta=1e-6, monitor=metric, verbose=1)]
    return calls + [callbacks.TerminateOnNaN()]
