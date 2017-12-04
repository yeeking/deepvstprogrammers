import sys
import os
import cPickle as pickle
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), 'models/mlp'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models/mlp_recursive'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models/rnn'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models/ga'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models/hill_climber'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from mlp import MLP
from mlp_recursive import RecursiveMLP
from rnn import LSTM
from ga import GeneticAlgorithm
from hill_climber import HillClimber
from plugin_feature_extractor import PluginFeatureExtractor
import tensorflow as tf
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
    features = tf.placeholder(tf.float32, [None, features_cols, features_rows])
    patches = tf.placeholder(tf.float32, [None, parameter_size])
    prob_keep_input = tf.placeholder(tf.float32)
    prob_keep_hidden = tf.placeholder(tf.float32)
    batch_size = tf.placeholder(tf.int32)

    warnings.simplefilter("ignore")

    lstm = LSTM(features=features, labels=patches, batch_size=batch_size)

    ga = GeneticAlgorithm(extractor=extractor, population_size=200,
                          percent_elitism_elites=5, percent_elitist_parents=5,
                          dna_length=(parameter_size), target_features=test_x,
                          feature_size=(features_cols * features_rows),
                          mutation_rate=0.01, mutation_size=0.1)

    hill_climber = HillClimber(extractor=extractor, target_features=test_x,
                               feature_size=(features_cols * features_rows),
                               parameter_size=parameter_size,
                               averaging_amount=4)

    mlp = MLP(features=features, labels=patches, parameters=[50, 40, 30],
              prob_keep_input=prob_keep_input,
              prob_keep_hidden=prob_keep_hidden)

    if parameter_size == 155:
        hier_mlp = RecursiveMLP(features=features, labels=patches,
                                parameters=[50, 40, 30],
                                prob_keep_input=prob_keep_input,
                                prob_keep_hidden=prob_keep_hidden)

    print "Initialising TensorFlow variables and building tensor graph..."
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    model_errors = {
        'ga': [],
        'lstm': [],
        'hill_climber': [],
        'mlp': [],
        'hier_mlp': []
    }

    mlp_prediction = sess.run(mlp.prediction, {features: test_x,
                                               patches: test_y,
                                               prob_keep_input: 1.0,
                                               prob_keep_hidden: 1.0})
    mlp_stats = get_stats(extractor, mlp_prediction, test_x, test_y)
    model_errors['mlp'] += [mlp_stats[0]]
    if parameter_size == 155:
        hier_mlp_prediction = sess.run(hier_mlp.prediction, {features: test_x,
                                                             patches: test_y,
                                                             prob_keep_input: 1.0,
                                                             prob_keep_hidden: 1.0})
        hier_mlp_stats = get_stats(extractor, hier_mlp_prediction, test_x, test_y)
        model_errors['hier_mlp'] += [hier_mlp_stats[0]]
    lstm_prediction = sess.run(lstm.prediction, {features: test_x,
                                                 patches: test_y,
                                                 batch_size: test_size})
    lstm_stats = get_stats(extractor, lstm_prediction, test_x, test_y)
    model_errors['lstm'] += [lstm_stats[0]]
    hill_prediction = hill_climber.prediction()
    hill_climber_stats = get_stats(extractor,
                                   hill_prediction,
                                   test_x,
                                   test_y)
    model_errors['hill_climber'] += [hill_climber_stats[0]]
    ga_prediction = ga.prediction()
    ga_stats = get_stats(extractor, ga_prediction, test_x, test_y)
    model_errors['ga'] += [ga_stats[0]]

    for iteration in range(iterations):

        print "\n*** Iteration: " + str(iteration) + " ***"

        print "\nMLP Network: "
        done = False
        step = 0
        while ((step + 1) * train_size) < len(train_y):
            start = step * train_size
            step += 1
            end = step * train_size
            sess.run(mlp.optimise, {features: train_x[start:end],
                                    patches: train_y[start:end],
                                    prob_keep_input: 0.75,
                                    prob_keep_hidden: 0.75})

        mlp_prediction = sess.run(mlp.prediction, {features: test_x,
                                                   patches: test_y,
                                                   prob_keep_input: 1.0,
                                                   prob_keep_hidden: 1.0})
        mlp_stats = get_stats(extractor, mlp_prediction, test_x, test_y)
        model_errors['mlp'] += [mlp_stats[0]]

        if parameter_size == 155:
            print "\nHierarchical MLP Network: "
            done = False
            step = 0
            while ((step + 1) * train_size) < len(train_y):
                start = step * train_size
                step += 1
                end = step * train_size
                sess.run(hier_mlp.optimise, {features: train_x[start:end],
                                             patches: train_y[start:end],
                                             prob_keep_input: 0.75,
                                             prob_keep_hidden: 0.75})

            hier_mlp_prediction = sess.run(hier_mlp.prediction, {features: test_x,
                                                                 patches: test_y,
                                                                 prob_keep_input: 1.0,
                                                                 prob_keep_hidden: 1.0})
            hier_mlp_stats = get_stats(extractor, hier_mlp_prediction, test_x, test_y)
            model_errors['hier_mlp'] += [hier_mlp_stats[0]]

        print "\nLSTM Network: "
        done = False
        step = 0
        while ((step + 1) * train_size) < len(train_y):
            start = step * train_size
            step += 1
            end = step * train_size
            sess.run(lstm.optimise, {features: train_x[start:end],
                                     patches: train_y[start:end],
                                     batch_size: train_size})

        lstm_prediction = sess.run(lstm.prediction, {features: test_x,
                                                     patches: test_y,
                                                     batch_size: test_size})
        lstm_stats = get_stats(extractor, lstm_prediction, test_x, test_y)
        model_errors['lstm'] += [lstm_stats[0]]

        print "\nHill Climber: "
        hill_climber.optimise()
        hill_prediction = hill_climber.prediction()
        hill_climber_stats = get_stats(extractor,
                                       hill_prediction,
                                       test_x,
                                       test_y)
        model_errors['hill_climber'] += [hill_climber_stats[0]]

        print "\nGenetic Algorithm: "
        ga.optimise()
        ga_prediction = ga.prediction()
        ga_stats = get_stats(extractor, ga_prediction, test_x, test_y)
        model_errors['ga'] += [ga_stats[0]]

        print "Hill: " + str(hill_climber_stats[0])
        print "Gene: " + str(ga_stats[0])
        print "Lstm: " + str(lstm_stats[0])
        if parameter_size == 155:
            print "Hier: " + str(hier_mlp_stats[0])
        print " Mlp: " + str(mlp_stats[0])

        print "Start iteration " + str(iteration) + " pickling."
        pickle.dump(hill_climber_stats, open("stats/" + operator_folder + "/hill_climber.p", "wb"))
        pickle.dump(ga_stats, open("stats/" + operator_folder + "/ga.p", "wb"))
        pickle.dump(lstm_stats, open("stats/" + operator_folder + "/lstm.p", "wb"))
        pickle.dump(mlp_stats, open("stats/" + operator_folder + "/mlp.p", "wb"))
        if parameter_size == 155:
            pickle.dump(hier_mlp_stats, open("stats/" + operator_folder + "/hier_mlp.p", "wb"))
        pickle.dump(model_errors, open("stats/" + operator_folder + "/all_models_error.p", "wb"))
        print "Finished iteration " + str(iteration) + " pickling."
