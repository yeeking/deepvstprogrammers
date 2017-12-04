import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../lazyloadingutils'))
import tensorflow as tf
from lazyloading import define_scope

class MLP:

    def __init__(self, **kwargs):
        self.features = kwargs.get('features', None)
        self.labels = kwargs.get('labels', None)
        self.parameters = kwargs.get('parameters', [2, 2])
        self.amount_layers = len(self.parameters)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.prob_keep_input = kwargs.get('prob_keep_input', None)
        self.prob_keep_hidden = kwargs.get('prob_keep_hidden', None)
        self.prediction
        self.optimise
        self.error

    @define_scope
    def prediction(self):
        def init_weights(shape, name):
            return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)
        number_inputs = int(self.features.get_shape()[1]) * int(self.features.get_shape()[2])
        x = tf.reshape(self.features, [-1, number_inputs])
        number_outputs = int(self.labels.get_shape()[1])

        weights = []
        biases = []
        weights += [init_weights([number_inputs, self.parameters[0]], "weights_hidden_0")]
        biases += [init_weights([self.parameters[0]], "biases_hidden_0")]
        for i, layer in enumerate(self.parameters):
            weights_name = "weights_hidden_" + str(i + 1)
            biases_name = "biases_hidden_" + str(i + 1)
            if i == (self.amount_layers - 1):
                weights += [init_weights([self.parameters[(self.amount_layers - 1)], number_outputs], weights_name)]
                biases += [init_weights([number_outputs], biases_name)]
            else:
                weights += [init_weights([self.parameters[i], self.parameters[i + 1]], weights_name)]
                biases += [init_weights([self.parameters[i + 1]], biases_name)]

        for i in range(len(weights)):
            if i < (len(weights) - 1):
                with tf.name_scope("Layer_" + str(i)):
                    prob = self.prob_keep_input if i == 0 else self.prob_keep_input
                    x = tf.nn.dropout(x, prob)
                    x = tf.add(tf.matmul(x, weights[i]), biases[i])
                    x = tf.nn.relu(x)
            else:
                with tf.name_scope("Output"):
                    x = tf.nn.dropout(x, self.prob_keep_hidden)
                    x = tf.add(tf.matmul(x, weights[i]), biases[i])
            tf.summary.histogram("weights_" + str(i) + "_summary", weights[i])
        return x

    @define_scope
    def optimise(self):
        optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimiser.minimize(self.error)

    @define_scope
    def error(self):
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.labels, self.prediction))))
