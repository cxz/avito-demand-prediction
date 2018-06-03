import numpy as np

# loading fasttext weights
# https://gist.github.com/brandonrobertz/49424db4164edb0d8ab34f16a3b742d5


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


def load_cached():
    # wi, E = load_pretrain()
    # dim = E.shape[1]

    # seq_len = 100
    # text_vec = np.zeros((len(txt_df), seq_len), dtype=np.uint32)
    # for idx, t in tqdm(enumerate(txt_df.text.values), total=len(txt_df)):
    #     words = t.split(' ')
    #     vec = [wi[w] if w in wi else 0 for w in words[:seq_len]]
    #     padding = max(0, seq_len - len(vec))  # left pad
    #     text_vec[idx, padding:] = vec
    # np.save('../cache/fasttext.weights.npy', E)
    E = np.load("../cache/fasttext.weights.npy")
    return E
