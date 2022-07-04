import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Flatten, Dropout, Activation


def create_model(img_width, img_height, img_depth):

    model = Sequential()
    # model layers

    model.add(Conv2D(
        filters=64,
        input_shape=(img_width, img_height, img_depth),
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
    model.add(Dropout(0.25))


    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(
        filters=512,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    ))

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())

    # Fully connected layer 1st layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # Fully connected layer 2nd layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(7, activation='softmax'))

    return model


def split_two(lst, ratio=[0.5, 0.5]):
    assert(np.sum(ratio) == 1.0)  # makes sure the splits make sense
    train_ratio = ratio[0]
    # note this function needs only the "middle" index to split, the remaining is the rest of the split
    indices_for_splittin = [int(len(lst) * train_ratio)]
    train, test = np.split(lst, indices_for_splittin)
    return train, test

def get_data():
    # images are 48x48
    Y = []
    X = []
    first = True
    for line in open('./fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)

    # shuffle and split
    X, Y = shuffle(X, Y)

    x_train, x_test = split_two(X, ratio=[0.9, 0.1])
    y_train, y_test = Y[:len(x_train)], Y[len(x_train):]

    x_train, x_valid = split_two(x_train, ratio=[0.9, 0.1])
    y_train, y_valid = y_train[:len(x_train)], y_train[len(x_train):]

    # if balance_ones:
    #     # balance the 1 class
    #     X0, Y0 = x_train[y_train != 1, :], y_train[y_train != 1]
    #     X1 = x_train[y_train == 1, :]
    #     X1 = np.repeat(X1, 9, axis=0)
    #     x_train = np.vstack([X0, X1])
    #     y_train = np.concatenate((Y0, [1]*len(X1)))

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def get_image_data():
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_data()
    N, D = x_train.shape
    d = int(np.sqrt(D))
    x_train = x_train.reshape(-1, 1, d, d)
    x_valid = x_valid.reshape(-1, 1, d, d)
    x_test = x_test.reshape(-1, 1, d, d)

    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    x_valid = x_valid.reshape(x_valid.shape[0], 48, 48, 1)
    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)

    y_train = to_categorical(y_train, 7)
    y_valid = to_categorical(y_valid, 7)
    y_test = to_categorical(y_test, 7)

    with open('test.npy', 'wb') as f:
        np.save(f, x_test)
        np.save(f, y_test)
    return x_train, y_train, x_valid, y_valid


def plot_hist(history):
    print(pd.DataFrame.keys(history))
    hist = history
    hist['epoch'] = history.epoch

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['categorical_accuracy'], name='accuracy', mode='markers+lines'))
    fig.update_layout(width=1000, height=500, title='Accuracy', xaxis_title='Epoki', yaxis_title='Accuracy', yaxis_type='log')
    fig.show()

    # plt.plot(history['categorical_accuracy'])
    plt.plot(history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    plt.legend(['val'], loc='upper left')
    plt.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='loss', mode='markers+lines'))
    fig.update_layout(width=1000, height=500, title='Loss', xaxis_title='Epoki', yaxis_title='Loss', yaxis_type='log')
    fig.show()

    plt.plot(history['loss'])
    # plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    plt.legend(['val'], loc='upper left')
    plt.show()


def predict(predictions, label_map, x_test):
      X, Y = predictions.shape

      while True:
          for i in range(10):
              plt.imshow(x_test[i].reshape(48, 48), cmap='gray')
              plt.title(label_map[predictions[i].tolist().index(max(predictions[i]))])
              plt.show()
          prompt = input('Quit? Enter Y:\n')
          if prompt.lower().startswith('y'):
              break
