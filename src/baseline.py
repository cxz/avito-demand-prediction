import warnings
import pickle

import pandas as pd
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

    X_test = X[train_rows:]
    X = X[:train_rows]
    print(X.shape)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.10, random_state=23
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
    ] + [
        # "weekday",
        # "top_1_name_resnet50",
        # "top_1_name_xception",
        # "top_1_name_inceptionresnetv2",
        # "top_1_name_vgg16",
    ]

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
        num_boost_round=1500,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    print("rmse:", np.sqrt(mean_squared_error(y_valid, gbm.predict(X_valid))))
    gbm.save_model("baseline_model.txt")

    # temporary
    subm = pd.read_csv("../input/sample_submission.csv")
    subm.deal_probability = np.clip(gbm.predict(X_test), .0, 1.)
    subm.to_csv("baseline.csv", index=False)


if __name__ == "__main__":
    validate()
