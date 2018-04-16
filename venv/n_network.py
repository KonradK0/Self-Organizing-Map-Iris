from random import random
import numpy as np
from neuron import Neuron


class Network:
    def __init__(self, neuron_num, input_set, normalizer):
        self.normalizer = normalizer
        self.neurons = self.init_neurons(neuron_num)
        self.input_set = input_set
        self.learning_ratio = random.uniform(0.1, 0.7)

    def init_neurons(self, neuron_num):
        neus = []
        for i in range(neuron_num):
            neu = Neuron()
            neu.normalize(self.normalizer)
            neus.append(neu)
        return np.array(neus)

    def learning_algorithm(self, input_vector):
        max_neuron = max(self.neurons, key=lambda neuron: np.dot(neuron.weights, input_vector))
        max_neuron.set_used(True)
        max_neuron.update_weights(input_vector, self.learning_ratio)
        max_neuron.normalize(self.normalizer)

    def run_epoch(self):
        for input_vector in self.input_set:
            self.learning_algorithm(input_vector)
        for i in range(self.neurons.size):
            if self.neurons[i].used is False:
                self.neurons = np.delete(self.neurons, i)
            else:
                self.neurons[i].set_used(False)

    def run_network(self):
        threshold = 0
        max_dot_product = 0
        while max_dot_product > threshold:
            self.run_epoch()

    def test_network(self, testing_set):
        pass