import re
import string
import logging

import pandas as pd
import numpy as np

import text
import super_pool

logger = logging.getLogger()
cleanup = text.SimpleCleanup()
emoji = text.Emoji()


def process_cleanup(text):
    return cleanup.process(text)


def process_emoji_count(text):
    return emoji.count(text)


def run(df=None):
    if df is None:
        df = pd.read_csv("../input/train.csv", usecols=["description", "title"])
        df_test = pd.read_csv("../input/test.csv", usecols=["description", "title"])
        df = pd.concat([df, df_test], axis=0)

    pool = super_pool.SuperPool()
    text_columns = ["description", "title"]

    for c in text_columns:
        logger.info(f"processing {c}")
        df[c].fillna("", inplace=True)

        df[f"{c}_emojis"] = pool.map(
            process_emoji_count, df[c].values, chunksize=1000, description=f"{c} emoji"
        )

        values = pool.map(
            process_cleanup, df[c].values, chunksize=1000, description=f"{c} cleanup"
        )

        count_words = [len(w) for w in values]
        unique_words = [len(set(x.split(" "))) for x in values]

        df[f"{c}_count_words"] = count_words
        df[f"{c}_unique_words"] = unique_words
        df[f"{c}_unique_words_percent"] = [
            float(u) / c if c != 0 else 0 for u, c in zip(unique_words, count_words)
        ]

    pool.exit()
    df.drop(text_columns, axis=1, inplace=True)
    return df


if __name__ == "__main__":
    df = pd.read_csv(
        "../input/train.csv", usecols=["description", "title"], nrows=10000
    )
    run(df)

    print(df.columns)
    print(df.dtypes)
    print(df.head())
