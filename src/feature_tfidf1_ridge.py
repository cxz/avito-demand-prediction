import pickle
import time

import pandas as pd
import numpy as np

import scipy
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt

SEED = 42
NFOLDS = 5

start = time.time()


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool=True):
        if seed_bool is True:
            params["random_state"] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


def get_oof(clf, kf, x_train, y, x_test):
    oof_train = np.zeros((x_train.shape[0],))
    oof_test = np.zeros((x_test.shape[0],))
    oof_test_skf = np.empty((NFOLDS, x_test.shape[0]))

    for i, (train_index, test_index) in enumerate(kf):
        print("\nFold {}".format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def run():
    df = pd.read_csv("../input/train.csv", usecols=["deal_probability"])

    ntrain = len(df)
    y = df.deal_probability.values

    ready_df = scipy.sparse.load_npz("../cache/20180601_traintestX_tfidf1.sparse.npz")

    ridge_params = {
        "alpha": 20.0,
        "fit_intercept": True,
        "normalize": False,
        "copy_X": True,
        "max_iter": 100,  # 'max_iter': None, 'tol': 0.001,
        "tol": 0.0025,
        "solver": "auto",
        "random_state": SEED,
    }

    kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)
    ridge = SklearnWrapper(clf=Ridge, seed=SEED, params=ridge_params)
    ridge_oof_train, ridge_oof_test = get_oof(
        ridge, kf, ready_df[:ntrain], y, ready_df[ntrain:]
    )

    rms = sqrt(mean_squared_error(y, ridge_oof_train))
    print("Ridge OOF RMSE: {}".format(rms))
    print("Elapsed: ", ((time.time() - start) / 60))

    out = np.concatenate([ridge_oof_train, ridge_oof_test])
    return out


if __name__ == "__main__":
    run()
