import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from plugin_feature_extractor import PluginFeatureExtractor
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from utility_functions import *
from tqdm import trange
from collections import namedtuple
from math import sqrt

lstm_num_hidden = 128
lstm_number_layers = 3
x = tf.placeholder(tf.float32, [None, 1024])
y = tf.placeholder(tf.float32, [None, 155])
keep_probability = tf.placeholder(tf.float32)

def conv2d(x, weights):
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')

x_spectrogram = tf.reshape(x, [-1, 32, 32, 1])

conv1_weights = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
conv1_bias = tf.Variable(tf.constant(0.1, shape=[32]))
conv1_hidden = tf.nn.relu(conv2d(x_spectrogram, conv1_weights) + conv1_bias)
conv1_pooling = tf.nn.max_pool(conv1_hidden, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
conv2_bias = tf.Variable(tf.constant(0.1, shape=[64]))
conv2_hidden = tf.nn.relu(conv2d(conv1_pooling, conv2_weights) + conv2_bias)
conv2_pooling = tf.nn.max_pool(conv2_hidden, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
conv2_output = tf.reshape(conv2_pooling, [-1, 64, 64])

# changes to [8, BatchSize, 8, 64]
tr_x = tf.transpose(conv2_output, [1, 0, 2])
re_x = tf.reshape(tr_x, [-1, 64])
sp_x = tf.split(0, 64, re_x)

lstm_cell = rnn_cell.LSTMCell(lstm_num_hidden, forget_bias=1.0, state_is_tuple=True)
lstm_cell = rnn_cell.MultiRNNCell([lstm_cell] * lstm_number_layers, state_is_tuple=True)
init_state = lstm_cell.zero_state(100, tf.float32)
lstm_output, _ = rnn.rnn(cell=lstm_cell, inputs=sp_x, dtype=tf.float32, initial_state=init_state)
lstm_weights = tf.Variable(tf.truncated_normal([lstm_num_hidden, 155], stddev=0.1))
lstm_bias = tf.Variable(tf.truncated_normal([155], stddev=0.1))
out = tf.add(tf.matmul(lstm_output[-1], lstm_weights), lstm_bias)

fully_connected1_weights = tf.Variable(tf.truncated_normal([155, 1024], stddev=0.1))
fully_connected1_biases = tf.Variable(tf.truncated_normal([1024], stddev=0.1))
fully_connected1 = tf.nn.relu(tf.matmul(out, fully_connected1_weights) + fully_connected1_biases)
fully_connected1_dropout = tf.nn.dropout(fully_connected1, keep_probability)

fully_connected2_weights = tf.Variable(tf.truncated_normal([1024, 155], stddev=0.1))
fully_connected2_biases = tf.Variable(tf.truncated_normal([155], stddev=0.1))
prediction = tf.matmul(fully_connected1_dropout, fully_connected2_weights) + fully_connected2_biases

def error(labels, prediction):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.sub(labels, prediction))))

rmse = error(y, prediction)
optimise = tf.train.AdamOptimizer(1e-4).minimize(rmse)

# Get training and testing batch.
train_batch_x = np.load("train_batch_x.npy")
train_batch_y = np.load("train_batch_y.npy")
test_batch_x = np.load("test_batch_x.npy")
test_batch_y = np.load("test_batch_y.npy")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):

    for batch in trange(99, desc="Training"):
        start = batch * 100
        end = batch * 100 + 100
        sess.run(optimise, { x: train_batch_x[start:end],
                             y: train_batch_y[start:end],
                             keep_probability: 0.5 })

    rmse_error = sess.run(rmse, { x: test_batch_x[0:100],
                                  y: test_batch_y[0:100],
                                  keep_probability: 1.0 })
    print "Root Mean Squared Error:" + str(rmse_error)
