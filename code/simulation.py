import math
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

# 1. Generate one particle positioned randomly on 2D grid
# 2. Generate a sequence of n steps which will move the particle in random directions

grid_height = 100
grid_width = 100

number_of_all_steps = 10000

batch_size = 10
number_of_epochs = 100
tracking_window = 20
number_of_coordinates = 2

number_of_data_steps = number_of_all_steps - tracking_window
number_of_batches = int(number_of_data_steps/batch_size)

num_hidden_units = 24
learning_rate = 0.001

color_code_init_point = '#FF0000'
color_code_next_points = '#0000FF'

points = []

def generate_random_particle():
    x = round(random.uniform(0, grid_width), 2)
    y = round(random.uniform(0, grid_height), 2)
    update_points_dict(0, x, y, 0, 0)

    return x, y

def display_next_point(i, x, y):
    # Plot random particle
    color_code = color_code_init_point if i == 0 else color_code_next_points
    ax.scatter(x, y, color=color_code)
    ax.annotate(i, (x, y))
    plt.pause(0.1)

    # print('X_'+ str(i), ': ', x, 'Y_' + str(i), ': ', y)

def generate_points():
    # Parameters for the random particle generation
    sigma = 0.001
    theta = 0

    x_generated, y_generated = 0, 0 # generate_random_particle()
    # Algorithm is based on https://en.wikipedia.org/wiki/Monte_Carlo_method_for_photon_transport?
    for i in range(number_of_all_steps):
        point = [x_generated, y_generated]
        points.append(point)

        # random.uniform(0, 1) - sample from range distribution
        s = random.uniform(0, 1)

        x_generated = x_generated + s * math.cos(theta)
        y_generated = y_generated + s * math.sin(theta)

        # np.random.randn() - sample from the standard distribution
        t = (2*np.random.randn()*sigma)*math.pi
        theta = theta + t

def display_points():
    #plt.axis([0, grid_width, 0, grid_height])

    for i, point in enumerate(points):
        display_next_point(i, point[0], point[1])

    plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)

print("Start generating points...")

generate_points()
#display_points()

print("End generating points")

# Once you have the data use RNN with TensorFlow to predict the next dots in the sequence.

def generate_input_output_data():
    inputs = []
    outputs = []
    for i in range(number_of_data_steps):
        inputs.append(points[i:i+tracking_window])
        outputs.append(points[i+tracking_window])
    return inputs, outputs

X = tf.placeholder(tf.float32, shape=[batch_size, tracking_window, number_of_coordinates])

Y = tf.placeholder(tf.float32, shape=[batch_size, number_of_coordinates])

weights = tf.Variable(tf.truncated_normal([num_hidden_units, number_of_coordinates]))

biases = tf.Variable(tf.truncated_normal([number_of_coordinates]))

rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=num_hidden_units)

outputs, state = tf.nn.dynamic_rnn(rnn_cell, inputs=X, dtype=tf.float32)

outputs = tf.transpose(outputs, [1, 0, 2])

last_output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

prediction = tf.matmul(last_output, weights) + biases

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=prediction)

total_loss = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=total_loss)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    X_train, Y_train = generate_input_output_data()

    for epoch in range(number_of_epochs):
        iter = 0

        for _ in range(number_of_batches):

            training_x = X_train[iter:iter+batch_size]

            training_y = Y_train[iter:iter+batch_size]

            _, current_total_loss, pred = sess.run([optimizer, total_loss, prediction], feed_dict={X: training_x, Y: training_y})

            print("Epoch: %s/%s; Iteration: %s/%s; Loss: %s;" % (epoch, number_of_epochs, iter, number_of_data_steps, current_total_loss))

            iter += batch_size

            print("__________________")

# Generate
#test_example = [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],
#
#                 [1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],
#
#                 [1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]
#
#prediction_result = sess.run(prediction, {X: test_example})
#
#largest_number_index = prediction_result[0].argsort()[-1:][::-1]
#
#print("Predicted sum: ", largest_number_index, "Actual sum:", 30)
#print("The predicted sequence parity is ", largest_number_index % 2, " and it should be: ", 0)



