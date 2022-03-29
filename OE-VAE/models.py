import numpy      as np
import tensorflow as tf
import h5py, time, sys, os, pickle
from tensorflow.keras import layers, models


class Encoder(layers.Layer):
    """ Mapping inputs to (z_mean, z_log_var, z) """
    def __init__(self, FC_layers, seed=None, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.FC_layers = FC_layers[:-1]; self.seed = seed
        self.denses = [layers.Dense(n_neurons, activation="relu") for n_neurons in self.FC_layers]
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
    def __init__(self, FC_layers, output_dim, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.FC_layers = FC_layers[:-1][::-1]
        self.denses = [layers.Dense(n_neurons, activation="relu") for n_neurons in self.FC_layers]
        self.dense_output = layers.Dense(output_dim, activation='linear')
    def call(self, x):
        for dense in self.denses:
            x = dense(x)
        return self.dense_output(x)


class VariationalAutoEncoder(models.Model):
    """ Combining the encoder and decoder into a VAE model for training """
    def __init__(self, FC_layers, input_dim, seed=None, name="autoencoder", **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.encoder   = Encoder(FC_layers, seed=seed)
        self.decoder   = Decoder(FC_layers, input_dim)
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        """ Adding KLD regularization loss """
        self.add_loss( KLD_loss(z_mean, z_log_var) )
        return reconstructed


class Sampling(layers.Layer):
    """ Using (z_mean, z_log_var) to sample z """
    def call(self, inputs, seed):
        z_mean, z_log_var = inputs
        return tf.random.normal(tf.shape(z_mean), seed=seed) * tf.exp(z_log_var/2) + z_mean


def KLD_loss(z_mean, z_log_var):
    """ Calculating latent layer KLD loss """
    return -tf.reduce_mean(1 + z_log_var - tf.exp(z_log_var) - tf.square(z_mean), axis=-1)/2


def OE_loss(vae, batch_X, batch_X_OE, OE_type, margin=2):
    """ Calculating outlier KLD loss """
    if OE_type == 'KLD':
        z_mean   , z_log_var   , _ = vae.encoder(batch_X)
        z_mean_OE, z_log_var_OE, _ = vae.encoder(batch_X_OE)
        loss = KLD_loss(z_mean, z_log_var) - KLD_loss(z_mean_OE, z_log_var_OE) + margin
        return tf.keras.activations.relu(loss)
    """ Calculating outlier MSE loss """
    reconstructed    = vae(batch_X)
    reconstructed_OE = vae(batch_X_OE)
    loss_MSE    = tf.keras.losses.MSE(batch_X   , reconstructed)
    loss_MSE_OE = tf.keras.losses.MSE(batch_X_OE, reconstructed_OE)
    if OE_type == 'MSE':
        return tf.keras.activations.sigmoid(loss_MSE - loss_MSE_OE) #-1
    if OE_type == 'MSE-margin':
        return tf.keras.activations.relu(loss_MSE - loss_MSE_OE + margin)


def get_losses(vae, data, OE_type, beta, lamb):
    bkg_sample, OoD_sample = data
    batch_X   , weights    = bkg_sample['constituents'], bkg_sample['weights']
    batch_X_OE, weights_OE = OoD_sample['constituents'], OoD_sample['weights']
    #batch_X, weights, batch_X_OE, weights_OE = data
    reconstructed = vae(batch_X)
    """ MSE reconstruction loss """
    loss_MSE = tf.keras.losses.MSE(batch_X, reconstructed)
    loss_MSE = tf.math.multiply(loss_MSE, weights)
    """ KLD regularization loss """
    loss_KLD = beta * sum(vae.losses)
    loss_KLD = tf.math.multiply(loss_KLD, weights)
    """ OE decorrelation loss   """
    loss_OE  = lamb * OE_loss(vae, batch_X, batch_X_OE, OE_type)
    loss_OE  = tf.math.multiply(loss_OE, weights_OE)
    """ Total training loss     """
    loss_train = loss_MSE + loss_KLD + loss_OE
    return loss_MSE, loss_KLD, loss_OE, loss_train


def train_model(vae, train_sample, valid_sample, OE_type='KLD', n_epochs=1, batch_size=5000,
                beta=0, lamb=0, lr=1e-3, output_dir=None, model_in=None, model_out=None):
    """ Using subclassing Tensoflow API to build model """
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    try   : dataset_len = tf.data.experimental.cardinality(train_sample).numpy()
    except: dataset_len = len(train_sample)
    metric_MSE   = tf.keras.metrics.Mean()
    metric_KLD   = tf.keras.metrics.Mean()
    metric_OE    = tf.keras.metrics.Mean()
    metric_train = tf.keras.metrics.Mean()
    metric_valid = tf.keras.metrics.Mean()
    """ Iterating over epochs """
    print('\nSTARTING TRAINING (generator '+('OFF' if dataset_len==1 else 'ON') +')')
    if dataset_len == 1:
        try   : train_sample = [train_sample[0]]
        except: pass
    history = {'MSE':[],'KLD':[],'OE':[],'Train loss':[],'Valid loss':[]}
    if output_dir is not None:
        history_file = output_dir+'/'+'history.pkl'
        if os.path.isfile(history_file) and model_in != output_dir+'/':
            history = pickle.load(open(history_file, 'rb'))
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
            #train_size = len(train_data[0])
            train_size = len(train_data[0]['weights'])
            n_batches  = int(np.ceil(train_size/batch_size))
            if epoch == 0: total_batches += n_batches
            """ Iterating over batches """
            for batch_idx in range(n_batches):
                sum_batches += 1
                idx  = batch_idx*batch_size, min((batch_idx+1)*batch_size, train_size)
                #data = [n[idx[0]:idx[1]] for n in train_data]
                data = [{key:sample[key][idx[0]:idx[1]] for key in sample} for sample in train_data]
                with tf.GradientTape() as tape:
                    loss_MSE, loss_KLD, loss_OE, loss_train = get_losses(vae, data, OE_type, beta, lamb)
                #if np.sum(tf.math.is_finite(loss_train)) != len(loss_train):
                #    print(loss_train)
                #    print( tf.where(tf.math.is_nan(loss_train)) )
                #    print( tf.where(tf.math.is_inf(loss_train)) )
                grads = tape.gradient(loss_train, vae.trainable_weights)
                """ Clipping gradients """
                grads = [tf.where(tf.math.is_finite(val), val, 0) for val in grads]
                grads = [tf.clip_by_value(val, -1e6, 1e6)         for val in grads]
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
                        condition = train_idx+1 != dataset_len or batch_idx+1 != n_batches
                        end, flush = ('\r', False)  if condition and key == 'Train loss' else ('  ', True)
                        print(key + ' = ' + format(val,'4.3e'), end=end, flush=True)
        for valid_data in valid_sample:
            loss_valid = get_losses(vae, valid_data, OE_type, beta, lamb)[-1]
            metric_valid(loss_valid)
        losses['Valid loss'] = metric_valid.result()
        print('Valid loss = ' + format(losses['Valid loss'],'4.3e'), end='  ', flush=True)
        print('(', '\b' + format(time.time() - start_time, '.1f'), '\b' + 's)')
        for key in history: history[key] += [losses[key].numpy() if key in losses else 0]
        if epoch > 0: optimizer, count = model_checkpoint(vae, optimizer, history, model_out, count)
        if output_dir is not None: pickle.dump(history, open(history_file,'wb'))
    #return {key:(val if len(val)!=0 else len(history['Train loss'])*[0.]) for key,val in history.items()}


def model_checkpoint(vae, optimizer, history, model_out, count, metric='Valid loss', patience=3, factor=2):
    if history[metric][-1] < np.min(history[metric][:-1]):
        print(metric, 'improved from', format(np.min(history[metric][:-1]),'4.2f'), end=' to ', flush=True)
        print(format(history[metric][-1],'4.2f'), '--> saving model weights to',  model_out)
        vae.save_weights(model_out); count = 0
    elif history[metric][-1] > np.min(history[metric][-(patience+1):-1]):
        count += 1
    if count >= patience:
        print('No improvement for', count, 'epochs', end=' --> ', flush=True)
        print('reducing learning rate from', optimizer.learning_rate.numpy(), end=' to ', flush=True)
        optimizer.learning_rate = optimizer.learning_rate/factor
        print(optimizer.learning_rate.numpy()); count = 0
    return optimizer, count


def find_nearest(value, array):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
