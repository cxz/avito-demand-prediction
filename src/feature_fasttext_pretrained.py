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


def load_pretrain():
    # 1888424 300
    with open("../data/wiki.ru.vec", "r") as f:
        data = f.readlines()

    samples, dim = data[0].split()
    E = np.zeros(shape=(int(samples), int(dim)), dtype="float32")
    word_index = {}

    idx = 0
    for line in tqdm(data[1:], total=E.shape[0]):
        word, vec = line.split(" ", 1)
        word_index[word] = idx
        E[idx, :] = [float(i) for i in vec.split()]
        idx += 1
    return word_index, E


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
    wi, E = load_pretrain()
    dim = E.shape[1]

    txt_df = pd.read_csv("../cache/title-description.txt")
    seq_len = 100
    text_vec = np.zeros((len(txt_df), seq_len), dtype=np.uint32)
    for idx, t in tqdm(enumerate(txt_df.text.values), total=len(txt_df)):
        words = t.split(" ")
        vec = [wi[w] if w in wi else 0 for w in words[:seq_len]]
        padding = max(0, seq_len - len(vec))  # left pad
        text_vec[idx, padding:] = vec

    np.save("../cache/title-description-seq100.npy", text_vec)


def run():
    prepare()
    # to_numpy()


if __name__ == "__main__":
    run()
