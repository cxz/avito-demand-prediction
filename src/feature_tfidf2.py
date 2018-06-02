"""
"""

import pandas as pd
import numpy as np
import time
import logging

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
import scipy
import pickle
import re

from nltk.corpus import stopwords

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

import text

logger = logging.getLogger()


def get_col(col_name):
    return lambda x: x[col_name]


def run():
    df = pd.read_csv("../input/train.csv", usecols=["description", "title"])
    df_test = pd.read_csv("../input/test.csv", usecols=["description", "title"])
    df = pd.concat([df, df_test], axis=0)

    cleanup = text.SimpleCleanup()

    df["title"] = df["title"].fillna("").apply(lambda x: cleanup.process2(x))
    df["description"] = (
        df["description"].fillna("").apply(lambda x: cleanup.process2(x))
    )

    text_columns = ["title", "description"]
    out = {}
    for c in text_columns:

        logger.info("processing %s" % c)

        df[c].fillna("", inplace=True)

        logger.info("fitting tfidf")
        tfidf = TfidfVectorizer(
            max_features=75000, token_pattern="\w+", ngram_range=(1, 2)
        )
        tfidf.fit(df[c])

        logger.info("fitting cvec")
        cvec = CountVectorizer(
            min_df=5,
            ngram_range=(1, 3),
            max_features=500,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w+",
        )
        cvec.fit(df[c])

        logger.info("tfidf")
        tmp = tfidf.transform(df[c])
        out["{}_tfidf_sum".format(c)] = tmp.sum(axis=1)
        out["{}_tfidf_mean".format(c)] = tmp.mean(axis=1)
        out["{}_tfidf_len".format(c)] = (tmp != 0).sum(axis=1)

        logger.info("svd")
        svd = TruncatedSVD(n_components=100)
        out["{}_svd".format(c)] = svd.fit_transform(tmp)
        out["{}_len".format(c)] = [len(x) for x in df[c]]
        out["{}_uniq".format(c)] = [len(set(x)) for x in df[c]]

        logger.info("cvec")
        tmp = cvec.transform(df[c])
        out["{}_cvec_sum".format(c)] = tmp.sum(axis=1)
        out["{}_cvec_mean".format(c)] = tmp.mean(axis=1)
        out["{}_cvec_len".format(c)] = (tmp != 0).sum(axis=1)

    for k in out.keys():
        logger.info(k)
        extra = pd.DataFrame(out[k])
        extra.columns = ["{}_{}".format(k, idx) for idx in range(extra.shape[1])]
        # df = pd.concat([df, extra], axis=1)
        for c in extra.columns:
            df[c] = extra[c]

    df.drop(text_columns, axis=1, inplace=True)
    return df


if __name__ == "__main__":
    import util

    logger = util.setup_logs("", "../tmp/tmp.log")
    run()
