import sys
import os
import cPickle as pickle

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

import numpy as np
import scipy.io.wavfile
from plugin_feature_extractor import PluginFeatureExtractor
from tqdm import trange


algorithm_number = 18
# Works:  1-15
# Bleh:  16-19
# Works: 20-32
alg = (1.0 / 32.0) * float(algorithm_number - 1) + 0.001
overriden_parameters = [(0, 1.0), (1, 0.0), (2, 1.0), (3, 0.0), (4, alg)]

other_params = [((i + 5), 0.5) for i in range(18)]

# operator_one = [((i + 23), 0.0) for i in range(22)]
# operator_two = [((i + 45), 0.0) for i in range(22)]
# operator_thr = [((i + 67), 0.0) for i in range(22)]
# operator_fou = [((i + 89), 0.0) for i in range(22)]
# operator_fiv = [((i + 111), 0.0) for i in range(22)]
# operator_six = [((i + 133), 0.0) for i in range(22)]

# overriden_parameters.extend(operator_one)
# overriden_parameters.extend(operator_two)
# overriden_parameters.extend(operator_thr)
# overriden_parameters.extend(operator_fou)
# overriden_parameters.extend(operator_fiv)
# overriden_parameters.extend(operator_six)
overriden_parameters.extend(other_params)

# overriden_parameters = np.load("data/fm/overriden_parameters.npy").tolist()
# desired_features = np.load("data/fm/desired_features.npy").tolist()

desired_features = [0, 1, 6]
desired_features.extend([i for i in range(8, 21)])
extractor = PluginFeatureExtractor(midi_note=24, note_length_secs=0.4,
                                   desired_features=desired_features,
                                   overriden_parameters=overriden_parameters,
                                   render_length_secs=0.7,
                                   pickle_path="utils/normalisers",
                                   warning_mode="ignore", normalise_audio=False)

# print np.array(extractor.overriden_parameters)

path = "/home/tollie/Development/vsts/dexed/Builds/Linux/build/Dexed.so"
# path = "/home/tollie/Development/vsts/synths/granulator/Builds/build-granulator-Desktop-Debug/build/debug/granulator.so"
# path = "/home/tollie/Downloads/synths/FMSynth/Builds/LinuxMakefile/build/FMSynthesiser.so"

extractor.load_plugin(path)

if extractor.need_to_fit_normalisers():

    print "No normalisers found, fitting new ones."
    extractor.fit_normalisers(10000)


# Get training and testing batch.
test_size = 10000
train_size = 100000

operator_folder = "data/ieee/"


def get_batches(train_batch_size, test_batch_size, extractor):

    (f, p) = extractor.get_random_normalised_example()
    f_shape = np.array(f).shape
    train_batch_x = np.zeros((train_batch_size, f_shape[0], f_shape[1]),
                             dtype=np.float32)
    train_batch_y = np.zeros((train_batch_size, p.shape[0]), dtype=np.float32)
    for i in trange(train_batch_size, desc="Generating Train Batch"):
        (features, parameters) = extractor.get_random_normalised_example()
        train_batch_x[i] = features
        train_batch_y[i] = parameters
        audio = extractor.float_to_int_audio(extractor.get_audio_frames())
        location = operator_folder + 'train' + str(i) + '.wav'
        scipy.io.wavfile.write(location, 44100, audio)

    test_batch_x = np.zeros((test_batch_size, f_shape[0], f_shape[1]),
                            dtype=np.float32)
    test_batch_y = np.zeros((test_batch_size, p.shape[0]), dtype=np.float32)
    for i in trange(test_batch_size, desc="Generating Test Batch"):
        (features, parameters) = extractor.get_random_normalised_example()
        test_batch_x[i] = features
        test_batch_y[i] = parameters
        audio = extractor.float_to_int_audio(extractor.get_audio_frames())
        location = operator_folder + 'test' + str(i) + '.wav'
        scipy.io.wavfile.write(location, 44100, audio)

    return train_batch_x, train_batch_y, test_batch_x, test_batch_y


train_x, train_y, test_x, test_y = get_batches(train_size, test_size,
                                               extractor)
# train_x, train_y, test_x, test_y = get_spectrogram_batches(train_size,
#                                                            test_size,
#                                                            extractor)
np.save(operator_folder + "/overriden_parameters.npy", overriden_parameters)
np.save(operator_folder + "/train_x.npy", train_x)
np.save(operator_folder + "/test_x.npy", test_x)
np.save(operator_folder + "/train_y.npy", train_y)
np.save(operator_folder + "/test_y.npy", test_y)

print "Finished."
