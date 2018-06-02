import warnings
import pickle

import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

from util import setup_logs
import data

logger = setup_logs("", "../tmp/tmp.log")


def validate():
    logger.info("loading data")
    X, y = data.load()

    with open("../cache/20180601_traintestX.columns.pkl", "rb") as f:
        column_names = pickle.load(f)

    train_rows = y.shape[0]
    X = X[:train_rows]
    print(X.shape)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=23
    )

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
    ] + ["weekday"]

    params = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 300,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.5,
        "learning_rate": 0.015,
        "verbose": 0,
    }

    print("categorical: ", categorical)

    lgb_train = lgb.Dataset(
        X_train,
        y_train,
        feature_name=column_names,
        categorical_feature=[x for x in column_names if x in categorical],
        free_raw_data=False,
    )

    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=3000,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=100,
        verbose_eval=50,
    )

    print("rmse:", np.sqrt(mean_squared_error(y_valid, gbm.predict(X_valid))))


if __name__ == "__main__":
    validate()
