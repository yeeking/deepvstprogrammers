import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../lazyloadingutils'))
import tensorflow as tf
from tensorflow.contrib import rnn
from lazyloading import define_scope

class LSTM:

    def __init__(self, **kwargs):
        self.features = kwargs.get('features', None)
        self.labels = kwargs.get('labels', None)
        self.number_hidden = kwargs.get('hidden_size', 100)
        self.number_layers = kwargs.get('number_layers', 3)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.batch_size = kwargs.get('batch_size', None)
        self.prediction
        self.optimise
        self.error

    @define_scope
    def prediction(self):
        def init_weights(shape, name):
            return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)
        number_outputs = int(self.labels.get_shape()[1])
        number_inputs = int(self.features.get_shape()[2])
        number_timesteps = int(self.features.get_shape()[1])
        weights = {
            'out': init_weights([self.number_hidden, number_outputs],
                                "weights_out")
        }
        biases = {
            'out': init_weights([number_outputs], "biases_out")
        }

        tr_x = tf.transpose(self.features, [1, 0, 2])
        re_x = tf.reshape(tr_x, [-1, number_inputs])
        sp_x = tf.split(re_x, number_timesteps, 0)
                          
        lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.number_hidden, forget_bias=1.0, state_is_tuple=True) for _ in range(self.number_layers)],
                                     state_is_tuple=True)
        init_state = lstm_cell.zero_state(self.batch_size, tf.float32)
        outputs, states = rnn.static_rnn(cell=lstm_cell, inputs=sp_x,
                                         dtype=tf.float32,
                                         initial_state=init_state)

        # Linear activation using rnn inner loop last output
        return tf.add(tf.matmul(outputs[-1], weights['out']), biases['out'])

    @define_scope
    def optimise(self):
        optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimiser.minimize(self.error)

    @define_scope
    def error(self):
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.labels,
                                                            self.prediction))))
