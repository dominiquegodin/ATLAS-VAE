import numpy      as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, LeakyReLU
from tensorflow.keras.layers import Flatten, Dense, concatenate, Reshape, Dropout, BatchNormalization
from tensorflow.keras        import Input, regularizers, models, callbacks, mixed_precision, optimizers, metrics
import sys


def multi_CNN(n_classes, sample, NN_type, FCN_neurons, CNN, l2, dropout, scalars, images, batchNorm=False):
    regularizer = regularizers.l2(l2)
    input_dict  = {key:Input(shape=sample[key].shape[1:], name=key) for key in scalars+images}
    shape_set   = set([sample[key].shape[1:] for key in images])
    output_list = []
    #IMAGES CNN
    for shape in shape_set:
        inputs  = [Reshape(shape+(1,))(input_dict[key]) for key in images if sample[key].shape[1:]==shape]
        outputs = concatenate(inputs, axis=3) if len(inputs) > 1 else inputs[0]
        if batchNorm: outputs = BatchNormalization()(outputs)
        if NN_type == 'CNN':
            n_maps  = [CNN[shape]['maps'   ][layer] for layer in np.arange(len(CNN[shape]['maps']))]
            kernels = [CNN[shape]['kernels'][layer] for layer in np.arange(len(CNN[shape]['maps']))]
            pools   = [CNN[shape]['pools'  ][layer] for layer in np.arange(len(CNN[shape]['maps']))]
            if np.all(np.array([len(kernel) for kernel in kernels]) >= 3):
                kernels_dim = 3; outputs = Reshape(outputs.shape[1:]+(1,))                         (outputs)
            else: kernels_dim = 2
            kernels = [(kernel+(3-len(kernel))*(1,))[:kernels_dim] for kernel in kernels]
            pools   = [( pool +(3-len( pool ))*(1,))[:kernels_dim] for  pool  in  pools ]
            for layer in np.arange(len(CNN[shape]['maps'])):
                if len(kernels[layer]) == 2:
                    outputs = Conv2D(n_maps[layer], kernels[layer], kernel_regularizer=regularizer)(outputs)
                    outputs = MaxPooling2D(pools[layer], padding='same')                           (outputs)
                if len(kernels[layer]) == 3:
                    outputs = Conv3D(n_maps[layer], kernels[layer], kernel_regularizer=regularizer)(outputs)
                    outputs = MaxPooling3D(pools[layer], padding='same')                           (outputs)
                if batchNorm: outputs = BatchNormalization()                                       (outputs)
                outputs = LeakyReLU(alpha=0)                                                       (outputs)
                outputs = Dropout(dropout)                                                         (outputs)
        output_list += [Flatten()(outputs)]
    #TRACKS FCN
    if 'constituents' in scalars:
        outputs = Flatten()(input_dict['constituents'])
        for n_neurons in [200]:#[200, 200]:
            outputs = Dense(n_neurons, kernel_regularizer=regularizer)                             (outputs)
            if batchNorm: outputs = BatchNormalization()                                           (outputs)
            outputs = LeakyReLU(alpha=0)                                                           (outputs)
            outputs = Dropout(dropout)                                                             (outputs)
        output_list += [outputs]
    #SCALARS
    outputs = [Flatten()(input_dict[key]) for key in scalars if key!='constituents']
    if len(outputs) != 0:
        outputs = concatenate(outputs) if len(outputs)>1 else outputs[0]
        for n_neurons in [200]:#[200, 200]:
            outputs = Dense(n_neurons, kernel_regularizer=regularizer)                             (outputs)
            if batchNorm: outputs = BatchNormalization()                                           (outputs)
            outputs = LeakyReLU(alpha=0)                                                           (outputs)
            outputs = Dropout(dropout)                                                             (outputs)
        output_list += [outputs]
    #CONCATENATION TO FCN
    outputs = concatenate(output_list) if len(output_list)>1 else output_list[0]
    for n_neurons in FCN_neurons:
        outputs = Dense(n_neurons, kernel_regularizer=regularizer)                                 (outputs)
        if batchNorm: outputs = BatchNormalization()                                               (outputs)
        outputs = LeakyReLU(alpha=0)                                                               (outputs)
        outputs = Dropout(dropout)                                                                 (outputs)
    outputs = Dense(n_classes, activation='softmax', dtype='float32')                              (outputs)
    return models.Model(inputs = list(input_dict.values()), outputs = outputs)


def create_model(n_classes, sample, NN_type, FCN_neurons, CNN, l2, dropout, train_var, n_gpus):
    devices = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3','/gpu:4', '/gpu:5', '/gpu:6', '/gpu:7']
    tf.debugging.set_log_device_placement(False)
    strategy = tf.distribute.MirroredStrategy(devices=devices[:n_gpus])
    with strategy.scope():
        if tf.__version__ >= '2.1.0':
            mixed_precision.experimental.set_policy('mixed_float16')
        if 'tracks' in train_var['images']: CNN[sample['tracks'].shape[1:]] = CNN.pop('tracks')
        model = multi_CNN(n_classes, sample, NN_type, FCN_neurons, CNN, l2, dropout, **train_var)
        print('\nNEURAL NETWORK ARCHITECTURE'); descent_optimizers(); model.summary()#;sys.exit()
        model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        #model.compile(optimizer=optimizers.Adam(lr=1e-3), loss='sparse_categorical_crossentropy')
    return model


def descent_optimizers():
    optimizers.Adadelta(learning_rate=1e-3, rho=0.95, epsilon=1e-07, name='Adadelta')
    optimizers.Adagrad (learning_rate=1e-3, initial_accumulator_value=0.1, epsilon=1e-07, name='Adagrad')
    optimizers.Adam    (learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
    optimizers.Adamax  (learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax')
    optimizers.Nadam   (learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam')
    optimizers.RMSprop (learning_rate=1e-3, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name='RMSprop')
    optimizers.SGD     (learning_rate=1e-2, momentum=0.0, nesterov=False, name='SGD')


def callback(model_out, patience, metrics):
    calls  = [callbacks.ModelCheckpoint(model_out, save_best_only=True, monitor=metrics, verbose=1)]
    calls += [callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_delta=1e-6, monitor=metrics, verbose=1)]
    calls += [callbacks.EarlyStopping(patience=patience, restore_best_weights=True,
                                      min_delta=1e-6, monitor=metrics, verbose=1)]
    return calls + [callbacks.TerminateOnNaN()]
