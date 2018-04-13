"""
Customized Deep Autoencoder
Reference: https://blog.keras.io/building-autoencoders-in-keras.html

Recommened to use anaconda "carnd-term1"
"""
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import h5py


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def deepAutoEncoder_GPS():

    # Parameter
    nb_epoch = 200
    batch_size = 256

    # Dataset, Reshape data
    h5f = h5py.File('gps_standardized_data.h5', 'r')
    gps_data = h5f['data'][:]
    h5f.close()
    dim_single_input = len(gps_data[0]) * len(gps_data[0][0])
    gps_data = np.reshape(gps_data, (-1, dim_single_input))
    x_train = gps_data[0:int(0.8*len(gps_data))]
    x_test = gps_data[int(0.8*len(gps_data)):]
    print('Train samples: {}'.format(x_train.shape))
    print('Test samples: {}'.format(x_test.shape))

#    # Model 1 : 3 layer for encoder
#    input_img = Input(shape=(dim_single_input,))
#    hidden_layer1 = 30
#    hidden_layer2 = 20
#    hidden_layer3 = 10  # because you have 10 categories
#    encoded = Dense(hidden_layer1, activation='relu')(input_img)
#    encoded = Dense(hidden_layer2, activation='relu')(encoded)
#    encoded = Dense(hidden_layer3, activation='relu')(encoded)
#    decoded = Dense(hidden_layer2, activation='relu')(encoded)
#    decoded = Dense(hidden_layer1, activation='relu')(decoded)
#    decoded = Dense(33, activation='sigmoid')(decoded)
#    model = Model(input=input_img, output=decoded)
#
#    # Model 2 : 1 layer for encoder
#    input_img = Input(shape=(dim_single_input,))
#    hidden_layer1 = 10
#    encoded = Dense(hidden_layer1, activation='relu')(input_img)
#    decoded = Dense(hidden_layer1, activation='relu')(encoded)
#    decoded = Dense(33, activation='sigmoid')(decoded)
#    model = Model(input=input_img, output=decoded)

    # Model 3 : 1 layer for encoder
    input_img = Input(shape=(dim_single_input,))
    hidden_layer1 = 20
    hidden_layer2 = 10
    encoded = Dense(hidden_layer1, activation='relu')(input_img)
    encoded = Dense(hidden_layer2, activation='relu')(encoded)
    decoded = Dense(hidden_layer2, activation='relu')(encoded)
    decoded = Dense(hidden_layer1, activation='relu')(decoded)
    decoded = Dense(33, activation='sigmoid')(decoded)
    model = Model(input=input_img, output=decoded)

    # Learn
    model.compile(optimizer='adadelta',
                  loss='mean_squared_error',
#                  loss=root_mean_squared_error,
#                  loss='binary_crossentropy',
                  metrics=['mae'])
    model.fit(x_train, x_train,
              nb_epoch=nb_epoch,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(x_test, x_test))
    score = model.evaluate(x_test, x_test, verbose=1)

    print('/n')
    print('Test score before fine turning:', score[0])
    print('Test accuracy after fine turning:', score[1])


def deepAutoEncoder_Mnist():

    # Parameter
    nb_classes = 10
    nb_epoch = 5
    batch_size = 256
    hidden_layer1 = 128
    hidden_layer2 = 64
    hidden_layer3 = 32  # because you have 10 categories

    # Dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print('Train samples: {}'.format(x_train.shape[0]))
    print('Test samples: {}'.format(x_test.shape[0]))

    from keras.utils import np_utils
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    input_img = Input(shape=(784,))
    encoded = Dense(hidden_layer1, activation='relu')(input_img)
    encoded = Dense(hidden_layer2, activation='relu')(encoded)
    encoded = Dense(hidden_layer3, activation='relu')(encoded)
    decoded = Dense(hidden_layer2, activation='relu')(encoded)
    decoded = Dense(hidden_layer1, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)

    model = Model(input=input_img, output=decoded)
    model.compile(optimizer='adadelta',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, x_train,
              nb_epoch=nb_epoch,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(x_test, x_test))
    score = model.evaluate(x_test, x_test, verbose=1)

    print('/n')
    print('Test score before fine turning:', score[0])
    print('Test accuracy after fine turning:', score[1])

# Main Function
#deepAutoEncoder_Mnist()
deepAutoEncoder_GPS()

    
##
## # Load Input Data
## h5f = h5py.File('gps_standardized_data.h5', 'r')
## gps_data = h5f['data'][:]
## h5f.close()
## print(gps_data)
##
## # Reshape data
## dim_single_input = len(gps_data[0]) * len(gps_data[0][0])
## print(dim_single_input)
## gps_data = np.reshape(gps_data, (dim_single_input, -1))
## print(gps_data.shape)
#
#encoding_dim = 32
#input_img = Input(shape=(784,))
#encoded = Dense(encoding_dim, activation='relu')(input_img)
#decoded = Dense(784, activation='sigmoid')(encoded)
#autoencoder = Model(input_img, decoded)
#encoder = Model(input_img, encoded)
#
#encoded_input = Input(shape=(encoding_dim,))
#decoder_layer = autoencoder.layers[-1]
#decoder = Model(encoded_input, decoder_layer(encoded_input))
#
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#
#from keras.datasets import mnist
#import numpy as np
#
#(x_train, _), (x_test, _) = mnist.load_data()
#
#autoencoder.fit(x_train, x_train,
#                epochs=50,
#                batch_size=256,
#                shuffle=True,
#                validation_data=(x_test, x_test))
#
#encoded_imgs = encoder.predict(x_test)
#decoded_imgs = decoder.predict(encoded_imgs)
