import common
import random
import os
import numpy as np

import keras.optimizers as opt
import keras.activations as act
import keras.initializers as init

from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

import matplotlib.pyplot as plt

class ConvLSTMModel():

    def __init__(self):
        # Remove warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        '''
        Define the hyperparameters of the model
        '''
        # Batch size
        self.batch_size = 20

        self.filters = 40
        self.kernel_size = (3, 3)

        # Loss function
        self.loss_function = 'binary_crossentropy'

        # Learning_rate = 0.0001...0.001
        self.learning_rate = 0.0001

        # Epochs
        self.num_epochs = 50

        # Activation, weights initializer and optimizer
        self.recurrent_activation = act.relu # https://keras.io/activations/
        self.recurrent_initializer = init.RandomUniform(minval=-0.05, maxval=0.05, seed=None) # https://keras.io/initializers/
        self.model_optimizer = opt.Adam(lr=self.learning_rate) # https://keras.io/optimizers/

        self.define_model()

    def _softmax_custom(self, axis):
        def soft(x):
            return act.softmax(x, axis=axis)
        return soft

    def define_model(self):
        self.model = Sequential()

        self.model.add(ConvLSTM2D(filters=self.filters, kernel_size=self.kernel_size, recurrent_initializer=self.recurrent_initializer, input_shape=(common.frame_window_size, common.number_of_points_per_frame, common.point_coordinates, 1), padding='same', return_sequences=True))
        self.model.add(BatchNormalization())

        self.model.add(ConvLSTM2D(filters=self.filters, kernel_size=self.kernel_size, padding='same', return_sequences=True))
        self.model.add(BatchNormalization())

        self.model.add(ConvLSTM2D(filters=self.filters, kernel_size=self.kernel_size, padding='same', return_sequences=True))
        self.model.add(BatchNormalization())

        self.model.add(ConvLSTM2D(filters=self.filters, kernel_size=self.kernel_size, padding='same', return_sequences=True))
        self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Reshape((-1, common.number_of_points_per_frame, common.point_coordinates * self.filters)))

        self.model.add(Dense(common.number_of_classes, activation=self._softmax_custom(3)))

        self.model.compile(loss=self.loss_function, optimizer=self.model_optimizer)

        self.model.summary()

    def train(self, train_X, train_y, validation_X, validation_y):
        # fit network
        history = self.model.fit(train_X, train_y, epochs=self.num_epochs, batch_size=self.batch_size, validation_data=(validation_X, validation_y), verbose=2, shuffle=False)

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()

    def test(self, test_X, test_y):
        test_X_random_number = random.randint(0, len(test_X) - 1)
        test_X_random = test_X[test_X_random_number]

        #print("Test point index:", (n_train + n_val)*common.frame_window_size + test_X_random_number)

        yhat = self.model.predict(test_X_random[np.newaxis, ::, ::, ::, ::])
        yhat = yhat[0]
        test_y_random = test_y[test_X_random_number]

        predicted_indicies = np.argmax(yhat, axis=2)
        actual_indicies = np.argmax(test_y_random, axis=2)

        time_steps = int(common.frame_window_size / 2)

        for i in range(time_steps):

            current_points_array = test_X_random[i]

            colors_array = ['#3300FF', '#333366', '#0066CC', '#00FFFF', '#660033', '#99FF66', '#336600', '#669900', '#333300', '#999933', '#FFCC33', '#CC6600', '#FF6600', '#CC3333', '#993333', '#996666', '#000000'] * common.number_of_points_per_frame

            display_point_count = common.number_of_points_per_frame

            if i == time_steps - 2:
                for j in range(display_point_count):
                    c = colors_array[j]
                    x_current_predicted = current_points_array[predicted_indicies[i][j]][0]
                    y_current_predicted = current_points_array[predicted_indicies[i][j]][1]
                    plt.scatter(x_current_predicted, y_current_predicted, s=90, color=c)

                    print("Point %s at time %s has predicted index %s and actual index %s" % (j, i, predicted_indicies[i][j], actual_indicies[i][j]))

            for j in range(display_point_count):
                c = colors_array[j]
                x_current = current_points_array[j][0]
                y_current = current_points_array[j][1]
                plt.scatter(x_current, y_current, s=30, color=c)

        plt.show()

        print("test_X shape: %s, yhat shape: %s, test_y shape: %s" % (test_X_random.shape, yhat.shape, test_y_random.shape))

    def evaluation_metric():
