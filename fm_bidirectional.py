import os
import sys
import tflearn
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from plugin_feature_extractor import PluginFeatureExtractor

training = False
learning_rate = 0.001
training_iters = 20
batch_size = 32
number_hidden = 256
np.random.seed(8)
checkpoint = 'model3.tfl.ckpt-361000'
data_folder = "data/fm/"

overriden_parameters = np.load(data_folder + "overriden_parameters.npy").tolist()
desired_features = np.load(data_folder + "desired_features.npy").tolist()
print "Features amount: " + str(len(desired_features))

extractor = PluginFeatureExtractor(midi_note=24, note_length_secs=0.4,
                                   desired_features=desired_features,
                                   overriden_parameters=overriden_parameters,
                                   render_length_secs=0.7,
                                   pickle_path="utils/normalisers",
                                   warning_mode="ignore",
                                   normalise_audio=False)
path = "/home/tollie/Downloads/synths/FMSynth/Builds/LinuxMakefile/build/FMSynthesiser.so"
# path = "/home/tollie/Development/vsts/synths/granulator/Builds/build-granulator-Desktop-Debug/build/debug/granulator.so"
extractor.load_plugin(path)

if extractor.need_to_fit_normalisers():
    extractor.fit_normalisers(10000)

(features, parameters) = extractor.get_random_normalised_example()

if training:
    def unison_shuffled_copies(a, b):
        """ Shuffle NumPy arrays in unison. """
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    a_x = np.load(data_folder + "0_all_x.npy")
    a_y = np.load(data_folder + "0_all_y.npy")
    b_x = np.load(data_folder + "1_all_x.npy")
    b_y = np.load(data_folder + "1_all_y.npy")
    c_x = np.load(data_folder + "2_all_x.npy")
    c_y = np.load(data_folder + "2_all_y.npy")
    d_x = np.load(data_folder + "3_all_x.npy")
    d_y = np.load(data_folder + "3_all_y.npy")
    all_x = np.concatenate((a_x, b_x, c_x, d_x))
    all_y = np.concatenate((a_y, b_y, c_y, d_y))
    train_x, train_y = unison_shuffled_copies(all_x, all_y)

valid_x = np.load(data_folder + "random_examples/valid_x.npy")
valid_y = np.load(data_folder + "random_examples/valid_y.npy")
test_x = np.load(data_folder + "random_examples/test_x.npy")
test_y = np.load(data_folder + "random_examples/test_y.npy")

features_cols = test_x[0].shape[0]
features_rows = test_x[0].shape[1]
parameter_size = test_y[0].shape[0]

# Net1
# net = tflearn.input_data([None, features_cols, features_rows])
# net = tflearn.lstm(net, number_hidden, dropout=0.8)
# net = tflearn.fully_connected(net, parameter_size, activation='relu')
# net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
#                          loss='mean_square')

from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.core import dropout
# Net2

# from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
# from tflearn.layers.core import dropout
# net = tflearn.input_data([None, features_cols, features_rows])
# net = bidirectional_rnn(net, BasicLSTMCell(number_hidden),
#                         BasicLSTMCell(number_hidden))
# net = dropout(net, 0.5)
# net = tflearn.fully_connected(net, parameter_size, activation='relu')
# net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
#                          loss='mean_square')
# model = tflearn.DNN(net, tensorboard_verbose=0,
#                     checkpoint_path=(data_folder + 'model2.tfl.ckpt'))

# Net3
net = tflearn.input_data([None, features_cols, features_rows])
net = bidirectional_rnn(net, BasicLSTMCell(number_hidden),
                        BasicLSTMCell(number_hidden))
net = dropout(net, 0.8)
net = tflearn.fully_connected(net, 64, activation='elu',
                              regularizer='L2', weight_decay=0.001)
net = tflearn.highway(net, 64, activation='elu',
                      regularizer='L2', weight_decay=0.001,
                      transform_dropout=0.8)
net = tflearn.highway(net, 64, activation='elu',
                      regularizer='L2', weight_decay=0.001,
                      transform_dropout=0.8)
net = tflearn.highway(net, 64, activation='elu',
                      regularizer='L2', weight_decay=0.001,
                      transform_dropout=0.8)
net = tflearn.highway(net, 64, activation='elu',
                      regularizer='L2', weight_decay=0.001,
                      transform_dropout=0.8)
net = tflearn.highway(net, 64, activation='elu',
                      regularizer='L2', weight_decay=0.001,
                      transform_dropout=0.8)
net = tflearn.highway(net, 64, activation='elu',
                      regularizer='L2', weight_decay=0.001,
                      transform_dropout=0.8)
net = tflearn.fully_connected(net, parameter_size, activation='elu')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
                         loss='mean_square')
model = tflearn.DNN(net, tensorboard_verbose=0,
                    checkpoint_path=(data_folder + 'model3.tfl.ckpt'))

if training:
    model.fit(train_x, train_y, validation_set=((valid_x, valid_y)),
              show_metric=True, batch_size=batch_size, n_epoch=training_iters,
              snapshot_epoch=True, snapshot_step=1000, run_id='granulator_lstm')
else:
    model.load(data_folder + checkpoint)

feature_distances = []
parameter_distances = []
predicted_patches = []
actual_patches = []


def get_abs_error(pred, actual):
    """Get the absolute error between the two arrays."""
    return np.add.reduce(np.abs(pred - actual).flatten())


for i in tqdm(range(len(test_x)), desc="Testing the model"):
    features_example = test_x[i].reshape((-1, features_cols, features_rows))
    predicted_patch = np.array(model.predict(features_example)).reshape((parameter_size))
    predicted_full_patch = extractor.partial_patch_to_patch(predicted_patch)
    predicted_patch_with_indices = extractor.add_patch_indices(predicted_full_patch)
    predicted_features = extractor.get_features_from_patch(predicted_patch_with_indices)
    actual_features = test_x[i]
    feature_distance = get_abs_error(predicted_features, actual_features)
    parameter_distance = get_abs_error(predicted_patch, test_y[i])
    feature_distances += [feature_distance]
    parameter_distances += [parameter_distance]
    predicted_patches += [predicted_patch]
    actual_patches += [test_y[i]]

np.save(data_folder + "feature_distances.npy", np.array(feature_distances))
np.save(data_folder + "parameter_distances.npy", np.array(parameter_distances))
np.save(data_folder + "predicted_patches.npy", np.array(predicted_patches))
np.save(data_folder + "actual_patches.npy", np.array(actual_patches))
print "Done :)"
