import sys
import os
import cPickle as pickle
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), 'models/hill_climber'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from hill_climber import HillClimber
from plugin_feature_extractor import PluginFeatureExtractor
import numpy as np
from utility_functions import get_batches, get_stats, display_stats, plot_error, write_wavs
from tqdm import trange

with warnings.catch_warnings():
    # Load VST.
    operator_folder = "six_operator"
    data_folder = "data/overriden/" + operator_folder + "/"
    overriden_parameters = np.load(data_folder + "overriden_parameters.npy").tolist()
    extractor = PluginFeatureExtractor(midi_note=24, note_length_secs=0.4,
                                       desired_features=[i for i in range(8, 21)],
                                       overriden_parameters=overriden_parameters,
                                       render_length_secs=0.7,
                                       pickle_path="utils/normalisers",
                                       warning_mode="ignore",
                                       normalise_audio=False)
    path = "/home/tollie/Development/vsts/dexed/Builds/Linux/build/Dexed.so"
    extractor.load_plugin(path)

    if extractor.need_to_fit_normalisers():
        extractor.fit_normalisers(2000000)

    # Get training and testing batch.
    nn_train_pass = 20
    test_size = 30
    train_size = 32
    iterations = 15
    train_x = np.load(data_folder + "train_x.npy")
    train_y = np.load(data_folder + "train_y.npy")
    test_x = np.load(data_folder + "test_x.npy")[0:test_size]
    test_y = np.load(data_folder + "test_y.npy")[0:test_size]

    # Load models.
    features_cols = train_x[0].shape[0]
    features_rows = train_x[0].shape[1]
    parameter_size = train_y[0].shape[0]

    warnings.simplefilter("ignore")

    hill_climber = HillClimber(extractor=extractor, target_features=test_x,
                               feature_size=(features_cols * features_rows),
                               parameter_size=parameter_size,
                               averaging_amount=4)

    model_errors = {
        'hill_climber': [],
    }

    hill_prediction = hill_climber.prediction()
    hill_climber_stats = get_stats(extractor, hill_prediction, test_x, test_y)
    model_errors['hill_climber'] += [hill_climber_stats[0]]

    for iteration in range(iterations):

        print "\n*** Iteration: " + str(iteration) + " ***"

        print "\nHill Climber: "
        hill_climber.optimise()
    #   hill_prediction = hill_climber.prediction()
    #    hill_climber_stats = get_stats(extractor,
    #                                   hill_prediction,
    #                                   test_x,
    #                                   test_y)
    #    model_errors['hill_climber'] += [hill_climber_stats[0]]

#        print "Hill: " + str(hill_climber_stats[0])
#
#        print "Start iteration " + str(iteration) + " pickling."
#        pickle.dump(hill_climber_stats, open("stats/" + operator_folder + #"/hill_climber.p", "wb"))
#        pickle.dump(model_errors, open("stats/" + operator_folder + #"/all_hills_error.p", "wb"))
#        print "Finished iteration " + str(iteration) + " pickling."
