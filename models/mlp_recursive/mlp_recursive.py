import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../lazyloadingutils'))
import tensorflow as tf
from lazyloading import define_scope

# Other: 0 - 22
# OP1:  23 - 44
# OP2:  45 - 66
# OP3:  67 - 88
# OP4:  89 - 110
# OP5: 111 - 132
# OP6: 133 - 154

# What if the global params were predicted first, and then fed the histogram
# of feature data to help choose the operator? Write a new class. Also what if
# one network was just trained solely on global params, then that network's
# input was passed in with the features to a MLP or something, perhaps using
# multiple optimisers like the link below, where the loss for the global params
# is pred_global - actual_global and the loss for the final prediction is the
# predicted_patch - actual_patch.

# http://www.kdnuggets.com/2016/07/multi-task-learning-tensorflow-part-1.html/2
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
class RecursiveMLP:

    def __init__(self, **kwargs):
        self.features = kwargs.get('features', None)
        self.labels = kwargs.get('labels', None)
        self.input_size = kwargs.get('input_size', None)
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

        number_outputs_op = 22
        number_outputs_other = 23

        operators_outputs = []

        weights = []
        biases = []

        for op in range(7):
            if op < 6:
                amount_outputs = number_outputs_op
                name_prefix = "op_"
            else:
                amount_outputs = number_outputs_other
                name_prefix = "_other"
            weights_layer = []
            biases_layer = []
            weights_layer += [init_weights([number_inputs, self.parameters[0]], name_prefix + str(op) + "_weights_hidden_0")]
            biases_layer += [init_weights([self.parameters[0]], name_prefix + str(op) + "_biases_hidden_0")]
            for i, layer in enumerate(self.parameters):
                weights_name = name_prefix + str(op) + "_weights_hidden_" + str(i + 1)
                biases_name = name_prefix + str(op) + "_biases_hidden_" + str(i + 1)
                if i == (self.amount_layers - 1):
                    weights_layer += [init_weights([self.parameters[(self.amount_layers - 1)], amount_outputs], weights_name)]
                    biases_layer += [init_weights([amount_outputs], biases_name)]
                else:
                    weights_layer += [init_weights([self.parameters[i], self.parameters[i + 1]], weights_name)]
                    biases_layer += [init_weights([self.parameters[i + 1]], biases_name)]
            weights += [weights_layer]
            biases += [biases_layer]
            operators_outputs += [x]

            for i in range(len(weights[0])):

                if i < (len(weights) - 1):
                    with tf.name_scope(name_prefix + str(op) + "_Layer_" + str(i)):
                        prob = self.prob_keep_input if i == 0 else self.prob_keep_input

                        operators_outputs[op] = tf.nn.dropout(operators_outputs[op], prob)
                        operators_outputs[op] = tf.add(tf.matmul(operators_outputs[op], weights[op][i]), biases[op][i])
                        operators_outputs[op] = tf.nn.relu(operators_outputs[op])
                else:
                    with tf.name_scope(name_prefix + str(op) + "_Output"):
                        operators_outputs[op] = tf.nn.dropout(operators_outputs[op], self.prob_keep_hidden)
                        operators_outputs[op] = tf.add(tf.matmul(operators_outputs[op], weights[op][i]), biases[op][i])
                tf.summary.histogram(name_prefix + str(op) + "_weights_" + str(i) + "_summary", weights[op][i])
        all_operators = tf.concat([o for o in operators_outputs[0:6]], 1)
        global_params = operators_outputs[6]
        predicted_patch = tf.concat([global_params, all_operators], 1)

        final_weights = []
        final_biases = []
        amount_outputs = 155
        final_weights += [init_weights([amount_outputs, self.parameters[0]], "final_weights_hidden_0")]
        final_biases += [init_weights([self.parameters[0]], "final_biases_hidden_0")]
        for i, layer in enumerate(self.parameters):
            final_weights_name = "final_weights_hidden_" + str(i + 1)
            final_biases_name = "final_biases_hidden_" + str(i + 1)
            if i == (self.amount_layers - 1):
                final_weights += [init_weights([self.parameters[(self.amount_layers - 1)], amount_outputs], final_weights_name)]
                final_biases += [init_weights([amount_outputs], final_biases_name)]
            else:
                final_weights += [init_weights([self.parameters[i], self.parameters[i + 1]], final_weights_name)]
                final_biases += [init_weights([self.parameters[i + 1]], final_biases_name)]

        for i in range(len(final_weights)):
            if i < (len(final_weights) - 1):
                with tf.name_scope("final_Layer_" + str(i)):
                    prob = self.prob_keep_input if i == 0 else self.prob_keep_input
                    predicted_patch = tf.nn.dropout(predicted_patch, prob)
                    predicted_patch = tf.add(tf.matmul(predicted_patch, final_weights[i]), final_biases[i])
                    predicted_patch = tf.nn.relu(predicted_patch)
            else:
                with tf.name_scope("final_Output"):
                    predicted_patch = tf.nn.dropout(predicted_patch, self.prob_keep_hidden)
                    predicted_patch = tf.add(tf.matmul(predicted_patch, final_weights[i]), final_biases[i])
            tf.summary.histogram("final_weights_" + str(i) + "_summary", final_weights[i])
        return predicted_patch

    @define_scope
    def optimise(self):
        optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimiser.minimize(self.error)

    @define_scope
    def error(self):
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.labels, self.prediction))))
