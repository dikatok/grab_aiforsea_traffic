import numpy as np


def calc_rmse(actual, pred):
    return np.sqrt(np.mean(np.square(actual - pred)))
