""" Preprocessing, caching and dataset loading.
"""

import logging

import pandas as pd
import numpy as np

from scipy.sparse import hstack
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from util import cache, timeit

import feature_base
import feature_text_stats

logger = logging.getLogger()


@timeit
def load():
    X = load_traintestX()
    y = load_trainy()
    return X, y


@cache("../cache/20180601_traintestX")
def load_traintestX():
    X = load_traintestX_base()
    X = pd.concat([X, load_traintestX_text_stats()], axis=1)

    if False:
        text = ["title", "description"]
        vectorizers = []
        for c in text:
            print("fitting %s" % c)
            v = TfidfVectorizer(
                max_features=100000, token_pattern="\w+", ngram_range=(1, 2)
            )
            v.fit(df[c].fillna(""))
            vectorizers.append(v)
        print(".")

        print("title")
        title = vectorizers[0].transform(df.loc[:, "title"].fillna("").values)

        print("desc")
        desc = vectorizers[1].transform(df.loc[:, "description"].fillna("").values)

        print(".")

        X = hstack([df[categorical], df[["price"]], title, desc]).tocsr()

    return X


@cache("../cache/20180601_traintestX_base.dataframe")
def load_traintestX_base():
    return feature_base.run()


@cache("../cache/20180601_traintestX_text_stats.dataframe")
@timeit
def load_traintestX_text_stats():
    return feature_text_stats.run()


@cache("../cache/20180601_trainy")
def load_trainy():
    df = pd.read_csv("../input/train.csv", usecols=["deal_probability"])
    return df.deal_probability.values


if __name__ == "__main__":
    load()
