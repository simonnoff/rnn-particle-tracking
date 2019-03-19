import common
import random
import os
from time import time
import numpy as np

import keras.backend as K
import keras.optimizers as opt
import keras.activations as act
import keras.initializers as init
import keras.losses as losses

from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard

import matplotlib.pyplot as plt

class ConvLSTMModel():

    def __init__(self):
        # Remove warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(common.checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
        '''
        Define the hyperparameters of the model
        '''
        # Batch size
        self.batch_size = 20

        self.filters = 40
        self.kernel_size = (3, 3)

        # Loss function alpha
        self.alpha = 0.2
        self.predefined_loss_function = 'binary_crossentropy'

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

    def _custom_loss_function(self, y_true, y_pred):
        print("y_true shape", y_true.shape, "y_pred shape", y_pred.shape)
        loss_1 = (1 - self.alpha) * losses.binary_crossentropy(y_true, y_pred)
        loss_2 = self.alpha * losses.binary_crossentropy(y_true, y_pred)
        return loss_1 + loss_2

        return K.categorical_crossentropy(y_pred, y_true) * final_mask

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

        final_layer = Dense(common.number_of_classes, activation=self._softmax_custom(3))
        self.model.add(final_layer)

        self.model.compile(loss=self.predefined_loss_function, optimizer=self.model_optimizer)

        self.model.summary()

        #print("Layer weights", final_layer.get_weights()[0].shape)

    def train(self, train_X, train_y, validation_X, validation_y):
        # fit network
        history = self.model.fit(train_X, train_y, epochs=self.num_epochs, batch_size=self.batch_size, validation_data=(validation_X, validation_y), verbose=2, shuffle=False, callbacks=[self.tensorboard, self.checkpoint_callback])

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()

    def test(self, test_X, test_y):
        self.model.load_weights(common.checkpoint_path)

        num_samples_test_X = test_X.shape[0]
        track_accuracy_matrix = np.zeros((common.number_of_points_per_frame, num_samples_test_X * common.frame_window_size))

        yhat = self.model.predict(test_X)

        for sample_index, test_X_sample in enumerate(test_X):
            yhat_single = yhat[sample_index]
            test_y_single = test_y[sample_index]

            predicted_indicies = np.argmax(yhat_single, axis=2)
            actual_indicies = np.argmax(test_y_single, axis=2)

            for i in range(common.frame_window_size - 1):

                current_time_step = sample_index * common.frame_window_size + i

                for j in range(common.number_of_points_per_frame):
                    track_accuracy_matrix[j][current_time_step] = 1 if predicted_indicies[i][j] == actual_indicies[i][j] else 0

        print("[ConvLSTM model] Final value:", common.evaluation_metric(track_accuracy_matrix))
            #plt.show()

    def visualize(self, all_X, test_X = None):
        samples_to_go = 1

        self.model.load_weights(common.checkpoint_path)

        test_number_samples = test_X.shape[0]
        yhat = self.model.predict(test_X)
        predicted_points_array = np.zeros((common.number_of_points_per_frame, common.point_coordinates, common.frame_window_size * samples_to_go - 1))
        counter = 0
        for test_sample_index in range(test_number_samples - samples_to_go, test_number_samples):
            yhat_sample = yhat[test_sample_index]
            predicted_indicies = np.argmax(yhat_sample, axis=2)
            for i in range(common.frame_window_size):
                current_time = counter * common.frame_window_size + i
                if i == common.frame_window_size - 1:
                    if test_sample_index == test_number_samples - 1:
                        break
                    next_points_array = test_X[test_sample_index + 1][0]
                else:
                    next_points_array = test_X[test_sample_index][i + 1]

                for j in range(common.number_of_points_per_frame):
                    predicted_points_array[j][0][current_time] = next_points_array[predicted_indicies[i][j]][0]
                    predicted_points_array[j][1][current_time] = next_points_array[predicted_indicies[i][j]][1]

            counter += 1

        all_points_array = np.zeros((common.number_of_points_per_frame, common.point_coordinates, common.frame_window_size * samples_to_go))
        for sample_index, sample in enumerate(all_X[-samples_to_go:]):
            for time_index, time_element in enumerate(sample):
                current_time = sample_index * common.frame_window_size + time_index
                for point_index, point in enumerate(time_element):
                    #print(point, current_time, points_array[point_index][0])
                    all_points_array[point_index][0][current_time] = point[0]
                    all_points_array[point_index][1][current_time] = point[1]

        colors_array = ['#3300FF', '#333366', '#0066CC', '#00FFFF', '#660033', '#99FF66', '#336600', '#669900', '#333300', '#999933', '#FFCC33', '#CC6600', '#FF6600', '#CC3333', '#993333', '#996666', '#000000'] * common.number_of_points_per_frame
        for point_index in range(8):
            points = all_points_array[point_index]
            predicted_points = predicted_points_array[point_index]
            color_real = '#999933'
            color_predicted = '#660033'
            how_many_to_display = int(common.frame_window_size / 2)
            plt.plot(points[0][-how_many_to_display:], points[1][-how_many_to_display:], color=color_real, linestyle='-')
            plt.plot(predicted_points[0][-how_many_to_display:], predicted_points[1][-how_many_to_display:], color=color_predicted, linestyle='dotted')

        plt.show()
