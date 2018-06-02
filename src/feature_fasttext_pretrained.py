"""
Pre-trained model obtained from: 
https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.ru.zip

https://gist.github.com/brandonrobertz/49424db4164edb0d8ab34f16a3b742d5
"""

import pandas as pd
import numpy as np

import text
import super_pool
from tqdm import tqdm

cleanup = text.SimpleCleanup()
pool = super_pool.SuperPool()


def prepare():
    """ Cleanup and save one text per line to feed fasttext.
    """
    df = pd.read_csv("../input/train.csv", usecols=["description", "title"])
    df_test = pd.read_csv("../input/test.csv", usecols=["description", "title"])
    df = pd.concat([df, df_test], axis=0)

    df["text"] = df["description"].astype(str) + df["title"].astype(str)

    text = pool.map(
        lambda x: cleanup.process(x),
        df.text.values,
        chunksize=10000,
        description="cleanup",
    )

    out = pd.DataFrame(text, columns=["text"])
    out.to_csv("../cache/title-description.txt", index=False)


def to_numpy():
    """
    """
    df = pd.read_csv("../data/title-description.bow.text")
    out = np.zeros((len(df), 300), dtype=np.float32)
    for i in tqdm(range(len(df))):
        txt = df.iloc[i][0].split(" ")[-301:-1]
        out[i, :] = [float(x) for x in txt]
    np.save("../data/title-description.bow.npy", out)


def run():
    # prepare()
    to_numpy()


if __name__ == "__main__":
    run()
