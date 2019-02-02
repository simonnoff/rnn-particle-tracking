import math
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

data_path = "/data/generated_points.csv"
generated_points = pd.read_csv(data_path)
generated_points = generated_points.drop(columns=generated_points.columns[0], axis = 1)
points = generated_points.values

train_split = 0.9
number_of_train_points = int(len(points) * train_split)

print(number_of_train_points)
train_points = points[0:number_of_train_points]
test_points = points[number_of_train_points:]

batch_size = 10
number_of_epochs = 10
tracking_window = 20
number_of_coordinates = 2

number_of_data_steps = number_of_train_points - tracking_window
number_of_batches = int(number_of_data_steps/batch_size)

num_hidden_units = 512
learning_rate = 0.001

# Once you have the data use RNN with TensorFlow to predict the next dots in the sequence.

def generate_input_output_data(curr_points):
    inputs = []
    outputs = []
    for i in range(number_of_data_steps):
        inputs.append(curr_points[i:i+tracking_window])
        outputs.append(curr_points[i+tracking_window])
    return inputs, outputs

X = tf.placeholder(tf.float32, shape=[batch_size, tracking_window, number_of_coordinates], name="X")

Y = tf.placeholder(tf.float32, shape=[batch_size, number_of_coordinates], name="Y")

weights = tf.Variable(tf.random_uniform([num_hidden_units, number_of_coordinates], minval=0, maxval=1, dtype=tf.float32), name="weights")
tf.summary.histogram("weights", weights)

biases = tf.Variable(tf.random_uniform([number_of_coordinates], minval=0, maxval=1, dtype=tf.float32), name="biases")
tf.summary.histogram("biases", biases)

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=num_hidden_units)

outputs, state = tf.nn.dynamic_rnn(rnn_cell, inputs=X, dtype=tf.float32)

outputs = tf.transpose(outputs, [1, 0, 2])

last_output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

prediction = tf.matmul(last_output, weights) + biases

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=Y, predictions=prediction))
tf.summary.scalar("loss", loss)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

with tf.name_scope("prediction"):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)

merged_summaries = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_writer = tf.summary.FileWriter("./tensorboard_log/7/train", sess.graph)
print("TensorFlow train graph is added to TensorBoard")

def train():
    X_train, Y_train = generate_input_output_data(train_points)

    for epoch in range(number_of_epochs):
        iter = 0

        for _ in range(number_of_batches):

            training_x = X_train[iter:iter+batch_size]
            training_y = Y_train[iter:iter+batch_size]

            summary, _, total_loss, curr_prediction, curr_last_output, curr_weights = sess.run([merged_summaries, optimizer, loss, prediction, last_output, weights], feed_dict={X: training_x, Y: training_y})
            summary_step = iter*epoch
            train_writer.add_summary(summary, iter*epoch)
            print("Epoch: %s/%s; Iteration: %s/%s; Summary step: %s; Loss: %s;" % (epoch, number_of_epochs, iter, number_of_data_steps, summary_step, total_loss))
            #print("\nPrediction: %s \nActual: %s; Last output: %s" % (curr_prediction, training_y, curr_last_output))

            iter += batch_size

            print("__________________")
train()


'''
Plan:

1. Generate 100000 points using the algorithm 
2. Save these points into csv
3. Train the network on these points
4. Train with 3 models:
    - LSTM (different variations);
    - Seq2Seq (different variations);
    - Seq2Seq with Attention;
'''

