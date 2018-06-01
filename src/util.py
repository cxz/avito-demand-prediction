import os
import logging
import pickle

from timeit import default_timer as timer
from functools import wraps
from pickle import HIGHEST_PROTOCOL

import pandas as pd
import numpy as np
import scipy


def setup_logs(root, save_file):
    ## initialize logger
    logger = logging.getLogger(root)
    logger.setLevel(logging.INFO)

    ## create the logging file handler
    fh = logging.FileHandler(save_file)

    ## create the logging console handler
    ch = logging.StreamHandler()

    ## format
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    ## add handlers to logger object
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def save_npz(fpath, d):
    if isinstance(d, pd.DataFrame):
        with open("{}.pkl".format(fpath), "wb") as f:
            pickle.dump(d, f, protocol=HIGHEST_PROTOCOL)
    elif scipy.sparse.issparse(d):
        scipy.sparse.save_npz(fpath, d)
    else:
        np.save(fpath, d)


def load_npz(fpath):
    if "dataframe" in fpath:
        with open("{}.pkl".format(fpath), "rb") as f:
            return pickle.load(f)
    elif "sparse" in fpath:
        return scipy.sparse.load_npz(fpath)
    else:
        return np.load(fpath)


def cache(fpath):
    logger = logging.getLogger()

    def wrap(f):
        def wrapped(*args, **kwargs):
            if os.path.exists(fpath):
                logger.info(f"loading from {fpath}")
                return load_npz(fpath)
            else:
                result = f(*args, **kwargs)
                logger.info(f"writing to {fpath}")
                save_npz(fpath, result)
                return result

        return wrapped

    return wrap


def timeit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("")
        start = timer()
        result = f(*args, **kwargs)
        end = timer()
        logger.info(f"{f.__name__} - elapsed time: {end-start:.4f} seconds")
        return result

    return wrapper
