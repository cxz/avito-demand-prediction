import logging

import pandas as pd
import numpy as np

from scipy.sparse import hstack
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


from util import cache, timeit

logger = logging.getLogger()


@timeit
def load():
    X = load_traintestX()
    y = load_trainy()
    return X, y


@cache("../cache/20180601_traintestSample.sparse.npz")
def load_traintestXSample():
    from scipy.sparse import csr_matrix

    return csr_matrix((3, 4), dtype=np.int8)


@cache("../cache/20180601_traintestX.sparse.npz")
def load_traintestX():
    df = pd.read_csv("../input/train.csv")
    df.drop(["deal_probability"], axis=1, inplace=True)

    df_test = pd.read_csv("../input/test.csv")
    df = pd.concat([df, df_test], axis=0)

    categorical = [
        "item_id",
        "user_id",
        "region",
        "city",
        "parent_category_name",
        "category_name",
        "item_seq_number",
        "user_type",
    ]
    text = ["title", "description"]
    target = "deal_probability"
    for c in categorical:
        print(c)
        le = preprocessing.LabelEncoder()
        df[c] = le.fit_transform(df[c])

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


@cache("../cache/20180601_trainy.npz")
def load_trainy():
    df = pd.read_csv("../input/train.csv", usecols=["deal_probability"])
    return df.deal_probability.values


if __name__ == "__main__":
    load()
