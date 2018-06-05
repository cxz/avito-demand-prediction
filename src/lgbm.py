import os
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import lightgbm as lgb

from util import setup_logs
import data

DATA = "../data/run5"

logger = setup_logs("", os.path.join(DATA, "run.log"))


def load_train():
    X, y = data.load()

    with open("../cache/20180601_traintestX.columns.pkl", "rb") as f:
        column_names = pickle.load(f)

    train_rows = y.shape[0]
    X = X[:train_rows]

    categorical = [
        "user_id",
        "region",
        "city",
        "parent_category_name",
        "category_name",
        "user_type",
        "image_top_1",
        "param_1",
        "param_2",
        "param_3",
    ]  # + ["weekday"]

    return X, y, column_names, categorical


def load_test():
    X, y = data.load()
    return X[y.shape[0] :, :]


def train(fold_no, X, y, column_names, categorical):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.10, random_state=23
    )

    lgb_train = lgb.Dataset(
        X_train,
        y_train,
        feature_name=column_names,
        categorical_feature=[x for x in column_names if x in categorical],
        free_raw_data=False,
    )

    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    params = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 270,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.75,
        "learning_rate": 0.016,
        "verbose": 0,
    }
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1500,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=50,
        verbose_eval=200,
    )

    best_iter = gbm.best_iteration
    y_pred = gbm.predict(X_valid, num_iteration=best_iter)
    score = np.sqrt(mean_squared_error(y_valid, y_pred))
    logger.info(f"fold {fold_no} rmse: {score}")

    gbm.save_model(os.path.join(DATA, "fold{}_model.txt".format(fold_no)))

    with open(os.path.join(DATA, "fold{}_model.pkl".format(fold_no)), "wb") as f:
        pickle.dump(gbm, f)

    return gbm


def main():
    seed = 42
    nfolds = 10

    logger.info("loading data...")
    X, y, column_names, categorical = load_train()
    logger.info("loaded.")

    # store folds
    fold = np.zeros_like(y)

    folds = KFold(nfolds, shuffle=True, random_state=seed)

    for fold_no, (train_index, test_index) in enumerate(folds.split(y)):
        logger.info(f"train fold {fold_no}")
        fold[test_index] = fold_no

        gbm = train(fold_no, X[train_index], y[train_index], column_names, categorical)

    # save folds
    np.save(os.path.join(DATA, "fold.npy"), fold)


def infer():
    nfolds = 10

    X = load_test()

    preds = np.zeros((X.shape[0],), dtype=np.float32)
    for fold_no in range(nfolds):
        logger.info(f"loading fold {fold_no}")

        model_file = os.path.join(DATA, f"fold{fold_no}_model.pkl")
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        logger.info(f"predicting...")
        preds += model.predict(X)

    preds /= nfolds
    subm = pd.read_csv("../input/sample_submission.csv")
    subm["deal_probability"] = np.clip(preds, .0, 1.)
    subm.to_csv(os.path.join(DATA, "subm.csv"), index=False)


if __name__ == "__main__":
    main()
    infer()
