import pandas as pd
import numpy as np
import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
import scipy
import pickle
import re

from nltk.corpus import stopwords

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize


class Cleanup:
    def __init__(self):
        self.stop_words = set(stopwords.words("russian"))

    def process(self, text):
        tokens = wordpunct_tokenize(text)
        tokens = [w.lower() for w in tokens]
        words = [
            word for word in tokens if word.isalpha() and word not in self.stop_words
        ]
        return " ".join(words)

    def process2(self, text):
        textProc = text.lower()
        textProc = re.sub("[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]", "", textProc)
        textProc = " ".join(textProc.split())
        return textProc


def get_col(col_name):
    return lambda x: x[col_name]


def run():
    df = pd.read_csv("../input/train.csv", usecols=["description", "title"])
    df_test = pd.read_csv("../input/test.csv", usecols=["description", "title"])
    df = pd.concat([df, df_test], axis=0)

    cleanup = Cleanup()

    df["title"] = df["title"].fillna("").apply(lambda x: cleanup.process2(x))
    df["description"] = (
        df["description"].fillna("").apply(lambda x: cleanup.process2(x))
    )

    tfidf_para = {
        "stop_words": set(stopwords.words("russian")),
        "analyzer": "word",
        "token_pattern": r"\w{1,}",
        "sublinear_tf": True,
        "dtype": np.float32,
        "norm": "l2",
        # "min_df": .05,
        # "max_df": .9,
        "smooth_idf": False,
    }

    vectorizer = FeatureUnion(
        [
            (
                "description",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=17000,
                    **tfidf_para,
                    preprocessor=get_col("description")
                ),
            ),
            (
                "title",
                CountVectorizer(
                    ngram_range=(1, 2),
                    stop_words=set(stopwords.words("russian")),
                    preprocessor=get_col("title"),
                ),
            ),
        ]
    )

    vectorizer.fit(df.to_dict("records"))
    out_df = vectorizer.transform(df.to_dict("records"))
    vocab = vectorizer.get_feature_names()

    with open("../cache/feature_tfidf_names.pkl", "wb") as f:
        pickle.dump(vocab, f)

    return out_df


if __name__ == "__main__":
    run()
