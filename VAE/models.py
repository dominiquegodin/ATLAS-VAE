import numpy as np, tensorflow as tf, sys
from   tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, ReLU, UpSampling2D, Reshape
from   tensorflow.keras.layers import Flatten, Dense, concatenate, Dropout, Activation, Layer, BatchNormalization
from   tensorflow.keras        import Input, regularizers, models, callbacks, mixed_precision, losses, optimizers


class Sampling(Layer):
    def call(self, inputs, seed):
        mean, log_var = inputs
        return tf.random.normal(tf.shape(mean), seed=seed) * tf.exp(log_var/2) + mean


def KL_loss(mean, log_var):
    return -0.5 * tf.reduce_mean(1 + log_var - tf.exp(log_var) - tf.square(mean), axis=-1)


def callback(model_out, patience, metrics):
    calls  = [callbacks.ModelCheckpoint(model_out, save_best_only=True, monitor=metrics, verbose=1)]
    calls += [callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_delta=1e-6, monitor=metrics, verbose=1)]
    calls += [callbacks.EarlyStopping(patience=patience, restore_best_weights=True,
                                      min_delta=1e-6, monitor=metrics, verbose=1)]
    return calls + [callbacks.TerminateOnNaN()]


def create_model(jets_dim, train_var, FC_layers, lr, beta, lamb, seed, encoder, n_gpus):
    if len(set(train_var)-{'constituents'}) == 0: input_dim = jets_dim
    elif 'constituents' not in train_var        : input_dim = len(train_var)
    else                                        : input_dim = jets_dim+len(train_var)-1
    tf.debugging.set_log_device_placement(False)
    strategy = tf.distribute.MirroredStrategy(devices=['/gpu:'+str(n) for n in np.arange(n_gpus)])
    with strategy.scope():
        #loss = 'mean_squared_error'
        loss = 'binary_crossentropy'
        if encoder == 'dual_ae':
            model = dual_AE(jets_dim, scalars_dim, FC_layers)
            model.compile(optimizer=optimizers.Adadelta(lr=lr), loss=loss, loss_weights=[1.0,1.0])
        else:
            if encoder == 'oe_vae': model = FC_VAE(input_dim, FC_layers, beta, lamb, seed)
            if encoder == 'vae'   : model = FC_VAE(input_dim, FC_layers, beta, seed)
            if encoder == 'ae'    : model = FC_AE (input_dim, FC_layers)
            model.compile(optimizer=optimizers.Adam(lr=lr, amsgrad=False), loss=loss)
            #model.compile(optimizer=optimizers.Adam(lr=lr, amsgrad=False), loss=loss, loss_weights=[1.0,1.0])
        print('\nNEURAL NETWORK ARCHITECTURE'); model.summary(); print()
    return model


def FC_VAE(input_dim, FC_layers, beta, lamb, seed, batchNorm=False):
    latent_dim = FC_layers[-1]; FC_layers = FC_layers[:-1]
    encoder_inputs = Input(shape=(input_dim,))
    oe_encoder_inputs = Input(shape=(input_dim,))
    x = BatchNormalization()(encoder_inputs) if batchNorm else encoder_inputs
    for n_neurons in FC_layers:
        x = Dense(n_neurons, kernel_initializer='he_normal')(x)
        if batchNorm: x = BatchNormalization()(x)
        x = Activation('relu')(x)
    qcd_mean       = Dense(latent_dim, name=  'mean' )(x)
    qcd_log_var    = Dense(latent_dim, name='log_var')(x)
    codings        = Sampling()([qcd_mean, qcd_log_var], seed)
    encoder        = models.Model(inputs=encoder_inputs, outputs=[codings, qcd_mean, qcd_log_var])
    decoder_inputs = Input(shape=latent_dim)
    x = BatchNormalization()(decoder_inputs) if batchNorm else decoder_inputs
    for n_neurons in FC_layers[::-1]:
        x = Dense(n_neurons, kernel_initializer='he_normal')(x)
        if batchNorm: x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = Dense(input_dim, activation='sigmoid')(x)
    decoder        = models.Model(inputs=decoder_inputs, outputs=x)
    codings, _, _  = encoder(encoder_inputs)
    _, oe_mean, oe_log_var = encoder(oe_encoder_inputs)
    reconstruction = decoder(codings)
    autoencoder    = models.Model(inputs=[encoder_inputs,oe_encoder_inputs], outputs=reconstruction)

    autoencoder.add_loss(beta*tf.reduce_mean(KL_loss(qcd_mean,qcd_log_var)))
    oe_loss = tf.keras.activations.relu( KL_loss(qcd_mean,qcd_log_var)-KL_loss(oe_mean,oe_log_var)+1. )
    autoencoder.add_loss( lamb*tf.reduce_mean(oe_loss) )

    return autoencoder


def FC_AE(input_dim, FC_layers, batchNorm=False):
    latent_dim = FC_layers[-1]; FC_layers = FC_layers[:-1]
    encoder_inputs = Input(shape=(input_dim,))
    x = encoder_inputs
    for n_neurons in FC_layers:
        x = Dense(n_neurons, kernel_initializer='he_normal')(x)
        if batchNorm: x = BatchNormalization()(x)
        x = Activation('relu')(x)
    codings = Dense(latent_dim, activation='relu', name='codings')(x)
    encoder = models.Model(inputs=encoder_inputs, outputs=codings)
    decoder_inputs = Input(shape=latent_dim)
    x = decoder_inputs
    for n_neurons in FC_layers[::-1]:
        x = Dense(n_neurons, kernel_initializer='he_normal')(x)
        if batchNorm: x = BatchNormalization()(x)
        x = Activation('relu')(x)
    #x = Dense(input_dim, activation='linear')(x)
    x = Dense(input_dim, activation='sigmoid')(x)
    decoder = models.Model(inputs=decoder_inputs, outputs=x)
    codings        = encoder(encoder_inputs)
    reconstruction = decoder(codings)
    autoencoder    = models.Model(inputs=encoder_inputs, outputs=reconstruction)
    return autoencoder


def dual_AE(jets_dim, scalars_dim, FC_layers, batchNorm=False):
    latent_dim = FC_layers[-1]; FC_layers = FC_layers[:-1]
    scalars_dim -= 1
    clusters_inputs = Input(shape=(jets_dim,))
    scalars_inputs  = Input(shape=(scalars_dim,))

    x = clusters_inputs
    for n_neurons in FC_layers:
        x = Dense(n_neurons, kernel_initializer='he_normal')(x)
        if batchNorm: x = BatchNormalization()(x)
        x = Activation('relu')(x)
    clusters_codings = Dense(latent_dim, activation='relu', name='codings')(x)
    clusters_encoder = models.Model(inputs=clusters_inputs, outputs=clusters_codings, name='clusters_encoder')

    clusters_decoder_inputs = Input(shape=latent_dim)
    x = clusters_decoder_inputs
    for n_neurons in FC_layers[::-1]:
        x = Dense(n_neurons, kernel_initializer='he_normal')(x)
        if batchNorm: x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = Dense(jets_dim, activation='sigmoid')(x)
    clusters_decoder = models.Model(inputs=clusters_decoder_inputs, outputs=x, name='clusters_decoder')

    clusters_codings = clusters_encoder(clusters_inputs)
    clusters_reconstruction = clusters_decoder(clusters_codings)

    y = scalars_inputs
    for n_neurons in FC_layers:
        y = Dense(n_neurons, kernel_initializer='he_normal')(y)
        if batchNorm: y = BatchNormalization()(y)
        y = Activation('relu')(y)
    scalars_codings = Dense(latent_dim, activation='relu', name='codings')(y)
    scalars_encoder = models.Model(inputs=scalars_inputs, outputs=scalars_codings, name='scalars_encoder')

    scalars_decoder_inputs = Input(shape=latent_dim)
    y = scalars_decoder_inputs
    for n_neurons in FC_layers[::-1]:
        y = Dense(n_neurons, kernel_initializer='he_normal')(y)
        if batchNorm: y = BatchNormalization()(y)
        y = Activation('relu')(y)
    y = Dense(scalars_dim, activation='sigmoid')(y)
    scalars_decoder = models.Model(inputs=scalars_decoder_inputs, outputs=y, name='scalars_decoder')

    scalars_codings = scalars_encoder(scalars_inputs)
    scalars_reconstruction = scalars_decoder(scalars_codings)

    autoencoder = models.Model(inputs=[clusters_inputs,scalars_inputs],
                               outputs=[clusters_reconstruction, scalars_reconstruction])
    return autoencoder


'''
def FC_VAE(input_dim, FC_layers, beta, seed, batchNorm=True):
    latent_dim = FC_layers[-1]; FC_layers = FC_layers[:-1]
    encoder_inputs = Input(shape=(input_dim,))
    x = BatchNormalization()(encoder_inputs) if batchNorm else encoder_inputs
    for n_neurons in FC_layers:
        x = Dense(n_neurons, kernel_initializer='he_normal')(x)
        if batchNorm: x = BatchNormalization()(x)
        x = Activation('relu')(x)
    mean           = Dense(latent_dim, name=  'mean' )(x)
    log_var        = Dense(latent_dim, name='log_var')(x)
    codings        = Sampling()([mean, log_var], seed)
    encoder        = models.Model(inputs=encoder_inputs, outputs=codings)
    decoder_inputs = Input(shape=latent_dim)
    x = BatchNormalization()(decoder_inputs) if batchNorm else decoder_inputs
    for n_neurons in FC_layers[::-1]:
        x = Dense(n_neurons, kernel_initializer='he_normal')(x)
        if batchNorm: x = BatchNormalization()(x)
        x = Activation('relu')(x)
    #x = Dense(input_dim, activation='linear')(x)
    x = Dense(input_dim, activation='sigmoid')(x)
    decoder        = models.Model(inputs=decoder_inputs, outputs=x)
    codings = encoder(encoder_inputs)
    reconstruction = decoder(codings)
    autoencoder    = models.Model(inputs=encoder_inputs, outputs=reconstruction)
    #autoencoder.add_loss(beta*tf.reduce_mean(KL_loss(mean,log_var))/input_dim)
    autoencoder.add_loss(beta*tf.reduce_mean(KL_loss(mean,log_var)))
    return autoencoder
'''
