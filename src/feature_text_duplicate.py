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


def hash_(x):
    return hash(x)


def run(df=None):
    if df is None:
        df = pd.read_csv(
            "../input/train.csv", usecols=["description", "title", "deal_probability"]
        )
        df_test = pd.read_csv("../input/test.csv", usecols=["description", "title"])
        df = pd.concat([df, df_test], axis=0)

    pool = super_pool.SuperPool()
    text_columns = ["description", "title"]

    for c in text_columns:
        logger.info(f"processing {c}")
        df[c].fillna("", inplace=True)

        hash_column = f"{c}_hash"

        df[hash_column] = pool.map(
            hash_, df[c].values, chunksize=10000, description=f"{c} hash"
        )

        empty_hash = hash("")

        gp = (
            df[(~df.deal_probability.isnull()) & (df[hash_column] != empty_hash)]
            .groupby([hash_column])
            .agg({"deal_probability": ["count", "min", "max", "median", "mean"]})
            .rename(columns={"deal_probability": "duplicate_%s" % c})
            .reset_index()
        )

        gp.columns = [
            "_".join([x for x in col if len(x) > 0]).strip()
            for col in gp.columns.values
        ]

        df = df.merge(gp[gp[f"duplicate_{c}_count"] > 10], on=hash_column, how="left")
        print(gp[gp[f"duplicate_{c}_count"] > 10].head())

        df.drop([hash_column], axis=1, inplace=True)

    pool.exit()
    df.drop(text_columns + ["deal_probability"], axis=1, inplace=True)
    return df


if __name__ == "__main__":
    df = pd.read_csv(
        "../input/train.csv",
        usecols=["description", "title", "deal_probability"],
        nrows=100000,
    )
    df = run(df)
