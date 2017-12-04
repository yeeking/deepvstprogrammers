import numpy as np
from tqdm import trange
import time


class HillClimber:

    def __init__(self, **kwargs):
        """Based on the Wikipedia Hill Climbing article's psuedo code."""
        self.extractor = kwargs.get('extractor', None)
        self.targets = kwargs.get('target_features', None)
        self.feature_size = kwargs.get('feature_size', None)
        self.parameter_size = kwargs.get('parameter_size', None)
        self.averaging_amt = kwargs.get('averaging_amount', 1)
        self.target_index = 0
        self.current_point = [[0.5 for _ in range(self.parameter_size)]
                              for _ in range(len(self.targets))]
        self.step_size = [[0.1 for _ in range(self.parameter_size)]
                          for _ in range(len(self.targets))]
        self.acceleration = 1.2
        self.candidate = [-self.acceleration, -1.0 / self.acceleration,
                          0.0, 1.0 / self.acceleration, self.acceleration]

    def get_fitness(self, individual):
        """Get Euclidean distance between target and individual's features.
        Greater is better.
        """
        patch = self.extractor.partial_patch_to_patch(individual)
        patch_w_ind = self.extractor.add_patch_indices(patch)
        features = self.extractor.get_features_from_patch(patch_w_ind)
        target = self.targets[self.target_index]
        dist = np.add.reduce(np.abs(features - target).flatten())
        fitness = float(max(0, (self.feature_size - dist)))
        return fitness

    def optimise(self):
        """Breed a new population."""
        for t in range(len(self.targets)):
            start_time = time.time()
            self.target_index = t
            for i in range(len(self.current_point[t])):
                best_score = self.get_fitness(self.current_point[t])
                best = 2
                for j in range(len(self.candidate)):
                    increment = self.step_size[t][i] * self.candidate[j]
                    self.current_point[t][i] += increment
                    temp = 0
                    for _ in range(self.averaging_amt):
                        temp += self.get_fitness(self.current_point[t])
                    temp /= self.averaging_amt
                    self.current_point[t][i] -= increment
                    if (temp > best_score):
                        best_score = temp
                        best = j
                if best == 2:
                    self.step_size[t][i] /= self.acceleration
                else:
                    increment = self.step_size[t][i] * self.candidate[best]
                    self.current_point[t][i] += increment
                    self.step_size[t][i] *= self.candidate[best]
            print("--- %s seconds ---" % (time.time() - start_time))

    def prediction(self):
        """Returns best indiviual for every respective target."""
        return self.current_point
