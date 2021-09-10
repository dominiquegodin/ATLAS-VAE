import numpy      as np
import tensorflow as tf
import time, sys
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
        for dense in self.denses: x = dense(x)
        z_mean    = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z         = self.sampling([z_mean, z_log_var], seed=self.seed)
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """ Converting the encoded digit vector z back to input space """
    def __init__(self, output_dim, FC_layers, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.FC_layers = FC_layers[:-1][::-1]
        self.denses = [layers.Dense(n_neurons, activation="relu") for n_neurons in self.FC_layers]
        self.dense_output = layers.Dense(output_dim, activation='linear')
    def call(self, x):
        for dense in self.denses: x = dense(x)
        return self.dense_output(x)


class VariationalAutoEncoder(models.Model):
    """ Combining the encoder and decoder into a VAE model for training """
    def __init__(self, input_dim, FC_layers, seed=None, name="autoencoder", **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.encoder   = Encoder(FC_layers, seed=seed)
        self.decoder   = Decoder(input_dim, FC_layers)
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


def OE_loss(vae, batch_X, batch_X_OE, OE_type, margin=1):
    """ Calculating outlier jets KLD-OE loss """
    if OE_type == 1:
        z_mean   , z_log_var   , _ = vae.encoder(batch_X)
        z_mean_OE, z_log_var_OE, _ = vae.encoder(batch_X_OE)
        loss = KLD_loss(z_mean, z_log_var) - KLD_loss(z_mean_OE, z_log_var_OE) + margin
        return tf.keras.activations.relu(loss)
    """ Calculating outlier jets MSE-OE loss """
    reconstructed    = vae(batch_X)
    reconstructed_OE = vae(batch_X_OE)
    loss_MSE    = tf.keras.losses.MSE(batch_X   , reconstructed)
    loss_MSE_OE = tf.keras.losses.MSE(batch_X_OE, reconstructed_OE)
    if OE_type == 2:
        return tf.keras.activations.sigmoid(loss_MSE - loss_MSE_OE) + 1
    if OE_type == 3:
        return tf.keras.activations.relu(loss_MSE - loss_MSE_OE + margin)


def get_losses(vae, data, beta, lamb, OE_type=2):
    batch_X, weights, batch_X_OE, weights_OE = data
    reconstructed = vae(batch_X)
    """ MSE reconstruction loss """
    loss_MSE = tf.keras.losses.MSE(batch_X, reconstructed)
    loss_MSE = tf.math.multiply(loss_MSE, weights)
    """ KLD regularization loss """
    loss_KLD = beta*sum(vae.losses)
    loss_KLD = tf.math.multiply(loss_KLD, weights)
    """ OE decorrelation loss   """
    loss_OE  = lamb * OE_loss(vae, batch_X, batch_X_OE, OE_type)
    loss_OE  = tf.math.multiply(loss_OE, weights_OE)
    return loss_MSE, loss_KLD, loss_OE, loss_MSE + loss_KLD + loss_OE


def build_model(vae, train_dataset, valid_dataset, n_epochs=1, beta=0, lamb=0, lr=1e-3):
    """ Using subclassing Tensoflow API to build model """
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    train_len = tf.data.experimental.cardinality(train_dataset).numpy()
    valid_len = tf.data.experimental.cardinality(valid_dataset).numpy()
    metric_MSE   = tf.keras.metrics.Mean()
    metric_KLD   = tf.keras.metrics.Mean()
    metric_OE    = tf.keras.metrics.Mean()
    metric_train = tf.keras.metrics.Mean()
    metric_valid = tf.keras.metrics.Mean()
    """ ITERATING OVER EPOCHS """
    progress_step = find_nearest(train_len/20, [1,2,5,10,20,50,100,200,500,1000])
    print('\nSTARTING TRAINING')
    for epoch in range(n_epochs):
        start_time = time.time()
        print('Epoch %d/%d:'%(epoch+1,n_epochs))
        metric_MSE  .reset_states()
        metric_KLD  .reset_states()
        metric_OE   .reset_states()
        metric_train.reset_states()
        metric_valid.reset_states()
        """ ITERATING OVER BATCHES """
        for batch_idx, train_data in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                loss_MSE, loss_KLD, loss_OE, loss_train = get_losses(vae, train_data, beta, lamb)
            grads = tape.gradient(loss_train, vae.trainable_weights)
            optimizer.apply_gradients( zip(grads, vae.trainable_weights) )
            metric_MSE  (loss_MSE  )
            metric_KLD  (loss_KLD  )
            metric_OE   (loss_OE   )
            metric_train(loss_train)
            if (batch_idx+1) % progress_step == 0 or batch_idx+1==train_len:
                losses = {'MSE':metric_MSE.result()}
                if beta != 0: losses['KLD'] = metric_KLD.result()
                if lamb != 0: losses['OE']  = metric_OE .result()
                losses['Train loss'] = metric_train.result()
                print('Batch %3d/%d: mean losses'%(batch_idx+1, train_len), end='  -->  ', flush=True)
                for key,val in losses.items():
                    end   = '\r'  if key=='Train loss' and batch_idx+1!=train_len else '  '
                    flush = False if key=='Train loss' and batch_idx+1!=train_len else True
                    print(key + ' = ' + format(val,'4.3e'), end=end, flush=True)
        for valid_data in valid_dataset:
            loss_MSE, loss_KLD, loss_OE, loss_valid = get_losses(vae, valid_data, beta, lamb=0)
        metric_valid(loss_valid)
        print('Valid loss = ' + format(metric_valid.result(),'4.3e'), end='  ', flush=True)
        print('(', '\b' + format(time.time() - start_time, '.1f'), '\b' + 's)')


def find_nearest(value, array):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
