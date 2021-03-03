import numpy as np, tensorflow as tf, sys
from   tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, LeakyReLU, UpSampling2D
from   tensorflow.keras.layers import Flatten, Dense, concatenate, Reshape, Dropout, Activation, Layer
from   tensorflow.keras        import Input, regularizers, models, callbacks, mixed_precision, losses, optimizers


def create_model(input_dim, FCN_neurons, latent_dim, lr, beta, n_gpus):
    tf.debugging.set_log_device_placement(False)
    strategy = tf.distribute.MirroredStrategy(devices=['/gpu:'+str(n) for n in np.arange(n_gpus)])
    with strategy.scope():
        model = FCN_VAE(input_dim, FCN_neurons, latent_dim, beta)
        print('\nNEURAL NETWORK ARCHITECTURE'); model.summary(); print()
        model.compile(optimizer=optimizers.Adam(lr=lr, amsgrad=False), loss='binary_crossentropy')
        #model.compile(optimizer=optimizers.Adam(lr=lr, amsgrad=True), loss='mean_squared_error')
        #model.compile(optimizer=optimizers.RMSprop(lr=lr), loss='binary_crossentropy')
    return model


class Sampling(Layer):
    def call(self, inputs, seed=None):
        mean, log_var = inputs
        return tf.random.normal(tf.shape(mean), seed=seed) * tf.exp(log_var/2) + mean


def callback(model_out, patience, metrics):
    calls  = [callbacks.ModelCheckpoint(model_out, save_best_only=True, monitor=metrics, verbose=1)]
    calls += [callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_delta=1e-5, monitor=metrics, verbose=1)]
    calls += [callbacks.EarlyStopping(patience=patience, restore_best_weights=True,
                                      min_delta=1e-5, monitor=metrics, verbose=1)]
    return calls + [callbacks.TerminateOnNaN()]


def FCN_AE(input_dim, FCN_neurons, latent_dim):
    encoder_inputs = Input(shape=(input_dim,)); z = encoder_inputs
    for n_neurons in FCN_neurons: z = Dense(n_neurons, activation='selu')(z)
    z = Dense(latent_dim, activation='selu')(z)
    encoder = models.Model(inputs=encoder_inputs, outputs=z)
    decoder_inputs = Input(shape=latent_dim); z = decoder_inputs
    for n_neurons in FCN_neurons[::-1]: z = Dense(n_neurons, activation='selu')(z)
    z = Dense(input_dim, activation='sigmoid')(z)
    #z = Dense(shape[0], activation='softmax')(z)
    decoder = models.Model(inputs=decoder_inputs, outputs=z)
    encoding       = encoder(encoder_inputs)
    reconstruction = decoder(encoding)
    vae = models.Model(inputs=encoder_inputs, outputs=reconstruction)
    return vae


def FCN_VAE(input_dim, FCN_neurons, latent_dim, beta):
    encoder_inputs = Input(shape=(input_dim,)); z = encoder_inputs
    for n_neurons in FCN_neurons: z = Dense(n_neurons, activation='relu')(z)
    coding_mean    = Dense(latent_dim)(z)
    coding_log_var = Dense(latent_dim)(z)
    coding         = Sampling()([coding_mean, coding_log_var])
    encoder        = models.Model(inputs=encoder_inputs, outputs=coding)
    decoder_inputs = Input(shape=latent_dim); z = decoder_inputs
    for n_neurons in FCN_neurons[::-1]: z = Dense(n_neurons, activation='relu')(z)
    z = Dense(input_dim, activation='sigmoid')(z)
    decoder        = models.Model(inputs=decoder_inputs, outputs=z)
    coding         = encoder(encoder_inputs)
    reconstruction = decoder(coding)
    vae            = models.Model(inputs=encoder_inputs, outputs=reconstruction)
    kl_loss        = -0.5 * tf.reduce_sum(1 + coding_log_var - tf.exp(coding_log_var)
                                          - tf.square(coding_mean), axis=-1)
    vae.add_loss(beta*tf.reduce_mean(kl_loss)/input_dim)
    return vae


'''
def CNN_AE(input_dim, latent_dim):
    encoder_inputs = Input(shape=(input_dim,)); z = encoder_inputs
    z = Reshape([8,10,1])(z)
    z = Conv2D(32, kernel_size=(3,3), padding='same', activation='selu')(z)
    z = Conv2D(32, kernel_size=(3,3), padding='same', activation='selu')(z)
    z = Flatten()(z)
    z = Dense(32, activation='selu')(z)
    z = Dense(latent_dim, activation='selu')(z)
    z = Dense(32, activation='selu')(z)
    z = Dense(2240, activation='selu')(z)
    z = Reshape([5, 7, 64])(z)
    z = Conv2D(32, kernel_size=(3,4), padding='valid', activation='selu')(z)
    z = UpSampling2D((2,2))(z)
    z = Conv2D(32, kernel_size=(3,4), padding='valid', activation='selu')(z)
    z = UpSampling2D((2,2))(z)
    z = Conv2D(1, kernel_size=(3,3), padding='same', activation='selu')(z)
    z = Reshape((80,))(z)
    z = Activation('sigmoid')(z)
    return models.Model(inputs=encoder_inputs, outputs=z)
'''
