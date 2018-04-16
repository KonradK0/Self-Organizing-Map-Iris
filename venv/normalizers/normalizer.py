from abc import ABCMeta, abstractmethod


class Normalizer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def normalize(self, vector):
        pass
