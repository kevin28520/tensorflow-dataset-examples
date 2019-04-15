# 1.
# Please download the dataset from this website
# CIFAR10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html (CIFAR-10 python version	~163 MB)

# 2.
# After unzipping the files, organize the files like this (I removed some unused files):

# |-- cifar10_dataset
# |    |-- data_batch_1
# |    |-- data_batch_2
# |    |-- data_batch_3
# |    |-- data_batch_4
# |    |-- data_batch_5
# |    |-- test_batch

# 3.
# For simplicity, I'm just training the model based on 'data_batch_1' and test the model against 'test_batch'.
# The purpose is to show how to make it work, not to do heavy optimization.
# If you want to get a better performance, please consider to train it based on all data, use data argumentation, use more advanced model architecture, etc.

# 4.
# I'm using the high level API in tensorflow which is tf.keras, I highly recommend tf.keras API, it is very convenient.

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import os
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
np.random.seed(99) # guarantee each splitting is the same

LABELS = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

# image shape is: 32x32x3
W = 32
H = 32
C = 3
BATCH_SIZE = 256
NUM_EPOCHS = 105
DROPOUT_RATE = 0.0 # dropout rate
LEARNING_PHASE = True # learning phase: true or false
N_CLASSES= 10 # 10 classes
LR = 0.0005 # learning rate
L2 = 0.0005


def unpickle(file):
    if not os.path.exists(file):
        raise Exception("The provided file: {} does not exist.".format(file))

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    data = dict['data'.encode()]
    data = data.reshape(-1, W, H, C)
    data = data/255.0 # transform 0-255 to 0-1

    labels = np.array(dict['labels'.encode()])
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    labels = enc.fit_transform(labels).toarray()

    return data, labels


def get_data():
    x = tf.keras.Input(batch_shape=(None, W, H, 3))
    y = tf.keras.Input(batch_shape=(None, 1))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat(NUM_EPOCHS)
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    next = iterator.get_next()
    return next


def conv_bn(x, n_filter):
    x = tf.keras.layers.Conv2D(n_filter, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2))(x)
    x = tf.keras.layers.BatchNormalization(trainable=LEARNING_PHASE)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


def cifar10_simple_network():
    # added dropout; added batch norm for all CONV; used global max pooling instead of flattening

    inputs = tf.keras.Input(batch_shape=(None, W, H, C), dtype='float')
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(inputs)

    x = conv_bn(x, 96)
    x = tf.keras.layers.MaxPool2D((2,2))(x)

    x = conv_bn(x, 96)
    x = tf.keras.layers.MaxPool2D((2,2))(x)

    x = conv_bn(x, 64)
    x = tf.keras.layers.MaxPool2D((2,2))(x)

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    outputs = tf.keras.layers.Dense(N_CLASSES, 'softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    opt = tf.keras.optimizers.SGD(LR, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def save_fig(history, name):
    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(name + '_acc.png')

    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(name + '_loss.png')


def save_confusion_matrix(y_true, y_pred, classes, normalize=True, title=None, model_name=None):
    """
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12,12))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, fraction=0.046, pad=0.04, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(model_name+'_confusion_matrix.png')
    return ax


if __name__ == '__main__':
    print('\nStarting...\n')

    train_file = 'cifar10_dataset/data_batch_1' # only train on the first
    test_file = 'cifar10_dataset/test_batch'
    x_train, y_train = unpickle(train_file)
    x_test, y_test = unpickle(train_file)

    print('train data shape: {}\n'
          'train labels shape: {}\n'
          'test data shape: {}\n'
          'test labels shape: {}\n'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    model = cifar10_simple_network()
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2)

    model_name = 'exp_006_cont'

    save_fig(history, model_name)
    model.save(model_name+'.hdf5')
    print('Model and training loss/accuracy figures are saved successfully.')

    # Predicting against testset
    y_pred_prob = model.predict(x_test, batch_size=BATCH_SIZE)
    y_pred_idx = np.argmax(y_pred_prob, axis=1)
    y_true_idx = np.argmax(y_test, axis=1)
    save_confusion_matrix(y_true_idx, y_pred_idx, classes=list(LABELS.values()), model_name=model_name)
    print('Confusion matrix is saved.')
