import os
import logging
from timeit import default_timer as timer
from functools import wraps

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
    if scipy.sparse.issparse(d):
        scipy.sparse.save_npz(fpath, d)
    else:
        np.savez(fpath, d)


def load_npz(fpath):
    if "sparse" in fpath:
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
