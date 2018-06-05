""" Preprocessing, caching and dataset loading.
"""

import logging
import pickle
import pandas as pd
import numpy as np

from scipy.sparse import hstack
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from util import cache, timeit

import feature_base
import feature_text_stats
import feature_tfidf1
import feature_tfidf1_ridge
import feature_tfidf2
import feature_mean_price
import feature_target_encoded
import feature_user_stats
import feature_text2  # temporary


@timeit
def load():
    X = load_traintestX()
    y = load_trainy()
    return X, y


@cache("../cache/20180601_traintestX.sparse")
def load_traintestX():
    X = load_traintestX_base()
    column_names = list(X.columns)
    cnt = 0
    for df_fn in [
        feature_text2.run,
        load_traintestX_tfidf1_ridge,
        load_traintestX_tfidf1,
        # load_traintestX_text_stats,
        # load_traintestX_mean_price,
        # load_traintestX_user_stats,
        # load_traintestX_target_encoded,
        # load_traintestX_tfidf2,
    ]:
        df2 = df_fn()
        if isinstance(df2, pd.DataFrame):
            column_names.extend(list(df2.columns))
        else:
            column_names.extend(
                ["{}_{}".format(cnt, idx) for idx in range(df2.shape[1])]
            )
        cnt += 1
        X = hstack([X, df2]).tocsr()

    with open("../cache/20180601_traintestX.columns.pkl", "wb") as f:
        pickle.dump(column_names, f)

    return X


@cache("../cache/20180601_traintestX_base.dataframe")
def load_traintestX_base():
    return feature_base.run()


@cache("../cache/20180601_traintestX_text_stats.dataframe")
@timeit
def load_traintestX_text_stats():
    return feature_text_stats.run()


@cache("../cache/20180601_traintestX_tfidf1.sparse")
@timeit
def load_traintestX_tfidf1():
    return feature_tfidf1.run()


@cache("../cache/20180601_traintestX_tfidf2.dataframe")
@timeit
def load_traintestX_tfidf2():
    return feature_tfidf2.run()


@cache("../cache/20180601_traintestX_tfidf1_ridge")
@timeit
def load_traintestX_tfidf1_ridge():
    return feature_tfidf1_ridge.run()


@cache("../cache/20180601_traintestX_mean_price.dataframe")
def load_traintestX_mean_price():
    return feature_mean_price.run()


@cache("../cache/20180601_traintestX_target_encoded.dataframe")
def load_traintestX_target_encoded():
    return feature_target_encoded.run()


@cache("../cache/20180601_traintestX_feature_user_stats.dataframe")
def load_traintestX_user_stats():
    return feature_user_stats.run()


@cache("../cache/20180601_trainy")
def load_trainy():
    df = pd.read_csv("../input/train.csv", usecols=["deal_probability"])
    return df.deal_probability.values


if __name__ == "__main__":
    logger = logging.getLogger()

    # load_traintestX_tfidf2()
    # load_traintestX_tfidf1()
    # load_traintestX_tfidf1_ridge()
    # X, y = load()
    # print(X.shape, y.shape)
    # x1 = load_traintestX_mean_price()
    # print(x1.dtypes)

    load_traintestX_base()
    load_traintestX_tfidf1()
    feature_text2.run()
