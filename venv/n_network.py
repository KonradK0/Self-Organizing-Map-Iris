from neuron import Neuron
from random import random
import numpy as np


class Network:
    def __init__(self, neurons, input_set, normalizer):
        self.neurons = neurons
        self.normalizer = normalizer
        self.input_set = input_set
        self.learning_ratio = random.uniform(0.1, 0.7)

    def learning_algorithm(self, input_vector, learning_ratio):
        max_neuron = self.neurons[0]
        max_product = float('-inf')
        for neu in self.neurons:
            if neu.used is True:
                neu.normalize(self.normalizer)
            dot_product = 0
            for n_weight, in_weight in zip(neu.weights, input_vector):
                dot_product += n_weight * in_weight
            if dot_product > max_product:
                max_neuron = neu
                max_product = dot_product
        max_neuron.set_used(True)
        max_neuron.update_weights(input_vector, learning_ratio)

    def run_epoch(self):
        for input_vector in self.input_set:
            self.learning_algorithm(input_vector, self.learning_ratio)
        for i in range(self.neurons.size):
            if self.neurons[i].used is False:
                self.neurons = np.delete(self.neurons, i)
            else:
                self.neurons[i].used = False

    def run_network(self):
        threshold = 0
        max_dot_product = 0
        while max_dot_product > threshold:
            self.run_epoch()
