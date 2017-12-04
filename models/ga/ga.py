import numpy as np
from tqdm import trange
import resource
import random


class GeneticAlgorithm:

    def __init__(self, **kwargs):
        """Initialise the population, mutation and crossover settings."""
        self.memory = list()
        self.end = 0
        self.tab = 0
        self.extractor = kwargs.get('extractor', None)
        self.population_size = kwargs.get('population_size', 1000)
        self.dna_length = kwargs.get('dna_length', None)
        self.crossover_index1 = int(np.floor(self.dna_length / 2.0))
        self.crossover_index2 = self.crossover_index1 + (self.dna_length % 2)
        self.targets = kwargs.get('target_features', None)
        self.feature_size = kwargs.get('feature_size', None)
        self.index_elitism_elites = int(np.floor((kwargs.get('percent_elitism_elites', 5) * 0.01) * self.population_size))
        self.index_elitist_parents = self.index_elitism_elites + int(np.floor(kwargs.get('percent_elitism_parents', 5) * 0.01) * self.population_size)
        self.population = []
        for target in range(len(self.targets)):
            current_population = []
            for _ in range(self.population_size):
                current_population.append(np.random.rand(self.dna_length))
            self.population.append(current_population)
        self.mutation_rate = kwargs.get('mutation_rate', 0.01)
        self.mutation_size = kwargs.get('mutation_size', 0.1)
        self.target_index = 0
        self.do_sum = False
        self.total_fitness_sum = 0
        self.sort_and_sum_population()

    def start_log_resources(self, function_name):
        # self.memory.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # self.tab += 1
        return

    def end_log_resources(self, function_name):
        # self.end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # last_element = len(self.memory) - 1
        # self.diff = self.end - self.memory[last_element]
        # self.memory.pop(last_element)
        # # do_print = False
        # # if do_print:
        # #     print "    " * self.tab + function_name
        # #     print "    " * self.tab + "Diff: " + str(self.diff) + "kb"
        # self.tab -= 1
        # return self.diff
        return

    def get_fitness(self, individual):
        """Get Euclidean distance between target and individual's features.
        Greater is better.
        """
        patch = self.extractor.partial_patch_to_patch(individual)
        patch_w_ind = self.extractor.add_patch_indices(patch)
        features = self.extractor.get_features_from_patch(patch_w_ind)
        target = self.targets[self.target_index]
        dist = np.add.reduce(np.abs(features - target).flatten())
        fitness = float(max(0, (self.feature_size - dist))) / self.feature_size
        if self.do_sum:
            self.total_fitness_sum += fitness
        return fitness

    def get_roulette_list(self):
        """Return list of population size that has all the indices to pick from."""
        roulette_list = []
        averaging_amount = 4
        for i in range(self.population_size):
            fitness = 0
            for _ in range(averaging_amount):
                dna = self.population[self.target_index][i]
                fitness += (self.get_fitness(dna) / averaging_amount)
            individials_amount = np.round(fitness / float(self.total_fitness_sum) * self.population_size)
            roulette_list.extend([i] * int(individials_amount))
        return roulette_list

    def crossover(self, dna1, dna2):
        """Takes half the dna from both arguments and returns the crossed over product."""
        c = np.concatenate((dna1[0:self.crossover_index1], dna2[-self.crossover_index2:]), axis=0)
        return c

    def mutation_sign(self):
        """When mutating, should we add or subtract the mutation size from the given parameter?"""
        return -1 if np.random.randint(0, 1, size=1) == 0 else 1

    def mutate(self, dna):
        """Has the probability of the mutuation_rate to mutate a given parameter in the dna representing the patch."""
        for parameter in dna:
            if np.random.uniform(0.0, 1.0) < self.mutation_rate:
                parameter += self.mutation_size * self.mutation_sign()
        return dna

    def sort_and_sum_population(self):
        """Sorts population by greatest fitness first and sums total fitness whilst doing it."""
        self.total_fitness_sum = 0
        self.do_sum = True
        self.population[self.target_index] = sorted(self.population[self.target_index], key=self.get_fitness, reverse=True)
        self.do_sum = False

    def prediction(self):
        """Returns best indiviual from each population for every respective target."""
        return [self.population[i][0] for i in range(len(self.population))]

    def optimise(self):
        """Breed a new population."""
        for i in range(len(self.targets)):
            self.target_index = i
            roulette_list = self.get_roulette_list()
            new_population = []
            for p in range(self.population_size):
                if p < self.index_elitism_elites:
                    new_population.append(self.population[self.target_index][(p % 2)])
                    continue
                elif p < self.index_elitist_parents:
                    index0 = p % 2
                    index1 = (p + 1) % 2
                    dna0 = self.population[self.target_index][index0]
                    dna1 = self.population[self.target_index][index1]
                else:
                    random0 = random.randint(0, len(roulette_list)-1)
                    random1 = random.randint(0, len(roulette_list)-1)
                    index0 = roulette_list[random0]
                    index1 = roulette_list[random1]
                    dna0 = self.population[self.target_index][index0]
                    dna1 = self.population[self.target_index][index1]
                new_dna = self.mutate(self.crossover(dna0, dna1))
                new_population.append(new_dna)
            assert len(new_population) == len(self.population[self.target_index])
            self.population[self.target_index] = new_population
            self.sort_and_sum_population()
