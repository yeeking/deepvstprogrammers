import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from plugin_feature_extractor import PluginFeatureExtractor
import tensorflow as tf
import numpy as np
from utility_functions import *
from tqdm import trange
from collections import namedtuple
from math import sqrt

x = tf.placeholder(tf.float32, [None, 1024])
y = tf.placeholder(tf.float32, [None, 155])
keep_probability = tf.placeholder(tf.float32)

def conv2d(x, weights):
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')

x_tensor = tf.reshape(x, [-1, 32, 32, 1])

def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky rectifier.
    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

h_1 = lrelu(tf.contrib.layers.batch_norm(conv2d(x_tensor, 32, name='conv1'),
                       is_training, scope='bn1'), name='lrelu1')
h_2 = lrelu(tf.contrib.layers.batch_norm(conv2d(h_1, 64, name='conv2'),
                       is_training, scope='bn2'), name='lrelu2')
h_3 = lrelu(tf.contrib.layers.batch_norm(conv2d(h_2, 64, name='conv3'),
                       is_training, scope='bn3'), name='lrelu3')
h_3_flat = tf.reshape(h_3, [-1, 64 * 4 * 4])
prediction = linear(h_3_flat, 10)

def error(labels, prediction):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.sub(labels, prediction))))

rmse = error(y, prediction)
optimise = tf.train.AdamOptimizer(1e-4).minimize(rmse)

# Load VST.
extractor = PluginFeatureExtractor(midi_note=24, note_length_secs=0.4,
                                   desired_features=[i for i in range(8, 21)],
                                   render_length_secs=0.7,
                                   pickle_path="utils/normalisers",
                                   warning_mode="ignore", normalise_audio=False)
path = "/home/tollie/Development/vsts/dexed/Builds/Linux/build/Dexed.so"
extractor.load_plugin(path)

# Get training and testing batch.
train_batch_x = np.load("train_x.npy")
train_batch_y = np.load("train_y.npy")
test_batch_x = np.load("test_x.npy")
test_batch_y = np.load("test_y.npy")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):

    for batch in trange(99, desc="Training"):
        start = batch * 100
        end = batch * 100 + 100
        sess.run(optimise, { x: train_batch_x[start:end],
                             y: train_batch_y[start:end],
                             keep_probability: 0.5 })

    rmse_error = sess.run(rmse, { x: test_batch_x,
                                  y: test_batch_y,
                                  keep_probability: 1.0 })
    print "Root Mean Squared Error:" + str(rmse_error)
