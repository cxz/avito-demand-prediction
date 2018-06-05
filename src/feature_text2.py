import pickle
import pandas as pd
import numpy as np

import scipy
from scipy.sparse import hstack

import vectorizer
import string
import re


def cleanName(text):
    textProc = text.lower()
    # textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
    # regex = re.compile(u'[^[:alpha:]]')
    # textProc = regex.sub(" ", textProc)
    textProc = re.sub("[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]", "", textProc)
    textProc = " ".join(textProc.split())
    return textProc


def run():
    df = pd.read_csv("../input/train.csv", usecols=["description", "title"])
    df_test = pd.read_csv("../input/test.csv", usecols=["description", "title"])
    df = pd.concat([df, df_test], axis=0)

    textfeats = ["description", "title"]
    df["desc_punc"] = df["description"].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation])
    )

    df["title"] = df["title"].fillna("").apply(lambda x: cleanName(x))
    df["description"] = df["description"].fillna("").apply(lambda x: cleanName(x))

    for cols in textfeats:
        df[cols] = df[cols].astype(str)
        df[cols] = df[cols].astype(str).fillna("missing")  # FILL NA
        df[cols] = df[
            cols
        ].str.lower()  # Lowercase all text, so that capitalized words dont get treated differently
        df[cols + "_num_words"] = df[cols].apply(
            lambda comment: len(comment.split())
        )  # Count number of Words
        df[cols + "_num_unique_words"] = df[cols].apply(
            lambda comment: len(set(w for w in comment.split()))
        )
        df[cols + "_words_vs_unique"] = (
            df[cols + "_num_unique_words"] / df[cols + "_num_words"] * 100
        )  # Count Unique Words

    # out = []
    # for c in ['description', 'title']:
    #     print('fitting ', c)
    #     v = vectorizer.Vectorizer(max_features=30000, token_pattern='\\w+', ngram_range=(1, 2))
    #     v.fit(df[c].fillna(''))
    #     out.append(v.transform(df.loc[:, c].fillna('').values))
    #
    # xout = hstack(out).tocsr()
    # scipy.sparse.save_npz('../cache/feature_tfidf2.npz', xout)
    # with open('../cache/feature_tfidf_names.pkl', 'wb') as f:
    #    pickle.dump(vocab, f)
    df.drop(["title", "description"], axis=1, inplace=True)
    return df


if __name__ == "__main__":
    run()
