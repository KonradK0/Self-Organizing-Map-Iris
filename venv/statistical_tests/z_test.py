from math import sqrt
from scipy.stats import norm


def run_z_test(mean, std, mean_to_test, num_of_samples, p_value):
    z = (mean_to_test - mean) / (std / sqrt(num_of_samples))
    lower_rejection_threshold = norm.ppf(- (1 - p_value / 2))
    upper_rejection_threshold = norm.ppf(1 - p_value / 2)
    return lower_rejection_threshold < z < upper_rejection_threshold
