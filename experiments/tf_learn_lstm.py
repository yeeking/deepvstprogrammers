import sys
import os
import cPickle as pickle
import tflearn
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utility_functions import get_batches, get_stats, display_stats
from plugin_feature_extractor import PluginFeatureExtractor

algorithm_number = 18
# Works:  1-15
# Bleh:  16-19
# Works: 20-32
alg = (1.0 / 32.0) * float(algorithm_number - 1) + 0.001
a = 4

overriden_parameters = [(0, 1.0), (1, 0.0), (2, 1.0), (3, 0.0), (a, alg)]

other_params = [((i + 5), 0.5) for i in range(17)]

operator_one = [((i + 23), 0.0) for i in range(22)]
operator_two = [((i + 45), 0.0) for i in range(22)]
operator_thr = [((i + 67), 0.0) for i in range(22)]
operator_fou = [((i + 89), 0.0) for i in range(22)]
operator_fiv = [((i + 111), 0.0) for i in range(22)]
operator_six = [((i + 133), 0.0) for i in range(22)]

# overriden_parameters.extend(operator_one)
overriden_parameters.extend(operator_two)
overriden_parameters.extend(operator_thr)
overriden_parameters.extend(operator_fou)
overriden_parameters.extend(operator_fiv)
overriden_parameters.extend(operator_six)

overriden_parameters.extend(other_params)

extractor = PluginFeatureExtractor(midi_note=24, note_length_secs=0.4,
                                   desired_features=[i for i in range(8, 21)],
                                   overriden_parameters=overriden_parameters,
                                   render_length_secs=0.7,
                                   pickle_path="utils/normalisers",
                                   warning_mode="ignore", normalise_audio=False)
print np.array(extractor.overriden_parameters)
path = "/home/tollie/Development/vsts/dexed/Builds/Linux/build/Dexed.so"
extractor.load_plugin(path)

folder_path = "../data/overriden/one_operator/"

# Get training and testing batch.
train_x = np.load(folder_path + "train_x.npy")
train_y = np.load(folder_path + "train_y.npy")
test_x = np.load(folder_path + "test_x.npy")
test_y = np.load(folder_path + "test_y.npy")

features_cols = train_x[0].shape[0]
features_rows = train_x[0].shape[1]
parameter_size = train_y[0].shape[0]

net = tflearn.input_data([None, features_cols, features_rows])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, parameter_size, activation='relu')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='mean_square')
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(train_x, train_y, validation_set=(test_x, test_y), show_metric=True,
          batch_size=32, n_epoch=1000)

predictions = []
for i in range(1000):
    predicted_patch = model.predict(test_x[i])
    full_patch = extractor.partial_patch_to_patch(predicted_patch)
    predictions.append(np.array(full_patch))

predictions = np.array(predictions).reshape((-1, 155))
print predictions.shape

actual_features = []
for i in range(1000):
    patch = extractor.add_patch_indices(test_y[i])
    features = extractor.get_features_from_patch(patch)
    actual_features += [features]
actual_features = np.array(actual_features)

stats = get_stats(extractor, predictions, actual_features, test_y[0:1000])
display_stats(stats)
