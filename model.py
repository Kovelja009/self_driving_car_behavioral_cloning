import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, InputLayer
from utils import INPUT_SHAPE, batch_generator
import argparse
import os
import tensorflow as tf
import math

np.random.seed(0)

MODEL_SAVE_PATH = 'model/model-{epoch:03d}.h5'


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(height, width, channels):
    """
    Build the model
    """
    # Define the model architecture
    model = Sequential()

    model.add(InputLayer(input_shape=(height, width, channels)))
    model.add( Lambda(lambda x: (x/255) - 0.5 , input_shape=(height, width, channels))) # Normalization layer
    model.add(Conv2D(24, (5,5), strides=(2,2), activation="relu"))
    model.add(Dropout(0.4))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation="relu"))
    model.add(Dropout(0.4))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation="relu"))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), activation="relu"))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), activation="relu"))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))

    return model


def train_model(model, args, x_train, x_valid, y_train, y_valid):
    """
    Train the model
    """

    # compile and train the model using the generator function coded above
    train_generator = batch_generator(args.data_dir, x_train, y_train, args.batch_size, True)
    validation_generator = batch_generator(args.data_dir, x_valid, y_valid, args.batch_size, False)


    # Optimizer
    optimizer = Adam(learning_rate=args.learning_rate)
    # Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    # Model weights are saved at the end of every epoch, if it's the best seen
    # so far.
    checkpoint = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

    # Stop training when a monitored quantity has stopped improving for longer than 10 epochs.
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

    # train model
    model.fit_generator(train_generator, 
        steps_per_epoch=(math.ceil(len(y_train) / args.batch_size)),
        validation_data=validation_generator,
        validation_steps=(math.ceil(len(y_valid) / args.batch_size)),
        epochs=args.nb_epoch, callbacks=[checkpoint, early_stop],verbose=1)

def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',              dest='data_dir',          type=str,   default='data')
    parser.add_argument('-n', help='number of epochs',            dest='nb_epoch',          type=int,   default=30)
    parser.add_argument('-s', help='number of batches per epoch', dest='steps_per_epoch',   type=int,   default=100)
    parser.add_argument('-b', help='batch size',                  dest='batch_size',        type=int,   default=128)
    parser.add_argument('-l', help='learning rate',               dest='learning_rate',     type=float, default=1.0e-3)
    args = parser.parse_args()

    data = load_data(args)
    # INPUT_SHAPE = height, width, channels 
    model = build_model(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])
    train_model(model, args, *data)


if __name__ == '__main__':
    main()

