"""
Generic setup of the data sources and the model training. 

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
and also on 
    https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

"""

import tensorflow.keras as keras
from tensorflow.keras.datasets       import mnist, cifar10
from tensorflow.keras.models         import Sequential
from tensorflow.keras.layers         import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks      import EarlyStopping, Callback
from tensorflow.keras.layers         import Conv2D, MaxPooling2D
from tensorflow.keras                import backend as K
import numpy as np
import os
import logging
from PIL import Image
# Helper: Early stopping.
early_stopper = EarlyStopping( monitor='val_loss', min_delta=0.1, patience=2, verbose=0, mode='auto' )

#patience=5)
#monitor='val_loss',patience=2,verbose=0
#In your case, you can see that your training loss is not dropping - which means you are learning nothing after each epoch. 
#It look like there's nothing to learn in this model, aside from some trivial linear-like fit or cutoff value.
def load_images(path, num_classes):
    # Load images

    print('Loading ' + str(num_classes) + ' classes')

    X_train = np.zeros([num_classes * 500, 3, 64, 64], dtype='uint8')
    y_train = np.zeros([num_classes * 500], dtype='uint8')

    trainPath = path + '/train'

    print('loading training images...');

    i = 0
    j = 0
    annotations = {}
    for sChild in os.listdir(trainPath):
        sChildPath = os.path.join(os.path.join(trainPath, sChild), 'images')
        annotations[sChild] = j
        for c in os.listdir(sChildPath):
            X = np.array(Image.open(os.path.join(sChildPath, c)))
            if len(np.shape(X)) == 2:
                X_train[i] = np.array([X, X, X])
            else:
                X_train[i] = np.transpose(X, (2, 0, 1))
            y_train[i] = j
            i += 1
        j += 1
        if (j >= num_classes):
            break

    print('finished loading training images')

    val_annotations_map = get_annotations_map()

    X_test = np.zeros([num_classes * 50, 3, 64, 64], dtype='uint8')
    y_test = np.zeros([num_classes * 50], dtype='uint8')

    print('loading test images...')

    i = 0
    testPath = path + '/val/images'
    for sChild in os.listdir(testPath):
        if val_annotations_map[sChild] in annotations.keys():
            sChildPath = os.path.join(testPath, sChild)
            X = np.array(Image.open(sChildPath))
            if len(np.shape(X)) == 2:
                X_test[i] = np.array([X, X, X])
            else:
                X_test[i] = np.transpose(X, (2, 0, 1))
            y_test[i] = annotations[val_annotations_map[sChild]]
            i += 1
        else:
            pass

    print('finished loading test images') + str(i)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    path = './tiny-imagenet-200'
    X_train, y_train, X_test, y_test = load_images(path, 2)

    fig1 = plt.figure()
    fig1.suptitle('Train data')
    ax1 = fig1.add_subplot(221)
    ax1.axis("off")
    ax1.imshow(np.transpose(X_train[0], (1, 2, 0)))
    print(y_train[0])
    ax2 = fig1.add_subplot(222)
    ax2.axis("off")
    ax2.imshow(np.transpose(X_train[499], (1, 2, 0)))
    print(y_train[499])
    ax3 = fig1.add_subplot(223)
    ax3.axis("off")
    ax3.imshow(np.transpose(X_train[500], (1, 2, 0)))
    print(y_train[500])
    ax4 = fig1.add_subplot(224)

    ax4.axis("off")
    ax4.imshow(np.transpose(X_train[999], (1, 2, 0)))
    print(y_train[999])

    plt.show()

    fig2 = plt.figure()
    fig2.suptitle('Test data')
    ax1 = fig2.add_subplot(221)
    ax1.axis("off")
    ax1.imshow(np.transpose(X_test[0], (1, 2, 0)))
    print(y_test[0])
    ax2 = fig2.add_subplot(222)
    ax2.axis("off")
    ax2.imshow(np.transpose(X_test[49], (1, 2, 0)))
    print(y_test[49])
    ax3 = fig2.add_subplot(223)
    ax3.axis("off")
    ax3.imshow(np.transpose(X_test[50], (1, 2, 0)))
    print(y_test[50])
    ax4 = fig2.add_subplot(224)
    ax4.axis("off")
    ax4.imshow(np.transpose(X_test[99], (1, 2, 0)))
    print(y_test[99])

    plt.show()
def get_cifar10_mlp():
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes  = 200 #dataset dependent
    batch_size  = 64
    epochs      = 4
    input_shape = (3072,) #because it's RGB

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test  = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test  = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs)

def get_tinyimagenet_cnn():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 200 #dataset dependent
    batch_size = 128
    epochs     = 4
    dir=input('enter tiny imagenet dataset directory:')
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = load_images(dir,nb_classes)
    
    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test  = to_categorical(y_test,  nb_classes)

    #x._train shape: (50000, 32, 32, 3)
    #input shape (32, 32, 3)
    input_shape = x_train.shape[1:]

    #print('x_train shape:', x_train.shape)
    #print(x_train.shape[0], 'train samples')
    #print(x_test.shape[0], 'test samples')
    #print('input shape', input_shape)
   
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs)
def get_annotations_map():
	valAnnotationsPath =input('please insert val annotation path:')
	valAnnotationsFile = open(valAnnotationsPath, 'r')
	valAnnotationsContents = valAnnotationsFile.read()
	valAnnotations = {}

	for line in valAnnotationsContents.splitlines():
		pieces = line.strip().split()
		valAnnotations[pieces[0]] = pieces[1]

	return valAnnotations
def get_mnist_mlp():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes  = 10 #dataset dependent 
    batch_size  = 64
    epochs      = 4
    input_shape = (784,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test  = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test  = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs)

def get_mnist_cnn():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 10 #dataset dependent 
    batch_size = 128
    epochs     = 4
    
    # Input image dimensions
    img_rows, img_cols = 28, 28

    # Get the data.
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    #x_train = x_train.reshape(60000, 784)
    #x_test  = x_test.reshape(10000, 784)
    
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    #print('x_train shape:', x_train.shape)
    #print(x_train.shape[0], 'train samples')
    #print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test  = to_categorical(y_test,  nb_classes)

    # convert class vectors to binary class matrices
    #y_train = keras.utils.to_categorical(y_train, nb_classes)
    #y_test = keras.utils.to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs)

def compile_model_mlp(genome, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers  = genome.geneparam['nb_layers' ]
    nb_neurons = genome.nb_neurons()
    activation = genome.geneparam['activation']
    optimizer  = genome.geneparam['optimizer' ]

    logging.info("Architecture:%s,%s,%s,%d" % (str(nb_neurons), activation, optimizer, nb_layers))

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons[i], activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons[i], activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout for each layer

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                    optimizer=optimizer,
                    metrics=['accuracy'])

    return model

def compile_model_cnn(genome, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        genome (dict): the parameters of the genome

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers  = genome.geneparam['nb_layers' ]
    nb_neurons = genome.nb_neurons()
    activation = genome.geneparam['activation']
    optimizer  = genome.geneparam['optimizer' ]

    logging.info("Architecture:%s,%s,%s,%d" % (str(nb_neurons), activation, optimizer, nb_layers))

    model = Sequential()

    # Add each layer.
    for i in range(0,nb_layers):
        # Need input shape for first layer.
        if i == 0:
            model.add(Conv2D(nb_neurons[i], kernel_size = (3, 3), activation = activation, padding='same', input_shape = input_shape))
        else:
            model.add(Conv2D(nb_neurons[i], kernel_size = (3, 3), activation = activation))
        
        if i < 2: #otherwise we hit zero
            model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Dropout(0.2))

    model.add(Flatten())
    # always use last nb_neurons value for dense layer
    model.add(Dense(nb_neurons[len(nb_neurons) - 1], activation = activation))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation = 'softmax'))

    #BAYESIAN CONVOLUTIONAL NEURAL NETWORKS WITH BERNOULLI APPROXIMATE VARIATIONAL INFERENCE
    #need to read this paper

    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

    return model

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def train_and_score(genome, dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    logging.info("Getting Keras datasets")

    if dataset   == 'cifar10_mlp':
        nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs = get_cifar10_mlp()
    elif dataset == 'tinyimagenet_cnn':
        nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs = get_tinyimagenet_cnn()
    elif dataset == 'mnist_mlp':
        nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs = get_mnist_mlp()
    elif dataset == 'mnist_cnn':
        nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs = get_mnist_cnn()

    logging.info("Compling Keras model")

    if dataset   == 'cifar10_mlp':
        model = compile_model_mlp(genome, nb_classes, input_shape)
    elif dataset == 'tinyimagenet_cnn':
        model = compile_model_cnn(genome, nb_classes, input_shape)
    elif dataset == 'mnist_mlp':
        model = compile_model_mlp(genome, nb_classes, input_shape)
    elif dataset == 'mnist_cnn':
        model = compile_model_cnn(genome, nb_classes, input_shape)

    history = LossHistory()

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,  
              # using early stopping so no real limit - don't want to waste time on horrible architectures
              verbose=1,
              validation_data=(x_test, y_test),
              #callbacks=[history])
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    K.clear_session()
    #we do not care about keeping any of this in memory - 
    #we just need to know the final scores and the architecture
    
    return score[1]  # 1 is accuracy. 0 is loss.
