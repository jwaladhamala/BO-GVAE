import sys
import json
import logging

import numpy as np
import torch
from skimage.filters import threshold_otsu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calc_dc(x, y):
    """Dice coefficient between x and y; threshold based on ostu method
    
    Args: 
        x: a matrix of input tissue properties
        y: a matrix of reconstructed tissue properties
    
    Output:
        mean dice coefficient
    """
    thresh_gt = 0.16  # tissue property threshold
    dc = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        idx_gt = np.where(x[i, :] >= thresh_gt)[0]
        thresh_c = threshold_otsu(y[i, :])
        idx_c = np.where(y[i, :] >= thresh_c)[0]
        dc[i] = 2 * len(np.intersect1d(idx_gt, idx_c)) / (len(idx_gt) + len(idx_c))
    return np.mean(dc)


def calc_msse(x, y):
    """Calculates the mean of the sum of squared error 
    between matrices X and Y
    """
    rmse = np.zeros(x.shape[0])
    sse = ((x - y) ** 2).sum(axis=1)
    mae = (np.absolute(x - y)).mean(axis=1)
    return np.mean(sse), np.mean(mae)


def calc_dc_fixedthres(x, y):
    """Dice coefficient with fixed threshold for both healthy and scar regions
    """
    thresh_gt = 0.16
    dch = np.zeros(x.shape[0])
    dcs = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        idx_gth = np.where(x[i, :] <= 0.16)[0]
        idx_gts = np.where(x[i, :] > 0.4)[0]
        idx_ch = np.where(y[i, :] <= 0.16)[0]
        idx_cs = np.where(y[i, :] > 0.4)[0]
        dch[i] = 2 * len(np.intersect1d(idx_gth, idx_ch)) / (len(idx_gth) + len(idx_ch))
        dcs[i] = 2 * len(np.intersect1d(idx_gts, idx_cs)) / (len(idx_gts) + len(idx_cs))
    return np.mean(dch), np.mean(dcs)


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

def inline_print(s):
    sys.stdout.write(s + '\r')
    sys.stdout.flush()
    

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
