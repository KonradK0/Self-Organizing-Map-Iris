import numpy as np


class Neuron:
    def __init__(self, weights):
        self.weights = weights
        self.used = False

    def normalize(self, normalizer):
        normalizer.normalize(self.weights)

    def set_used(self, was_used):
        self.used = was_used

    def update_weights(self, input_vector, learning_ratio):
        for in_weight, i in zip(input_vector, range(self.weights.size)):
            self.weights[i] = self.weights[i] + learning_ratio * (in_weight - self.weights[i])
