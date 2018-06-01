from pymystem3 import Mystem
import unicodedata
import sys
import time
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer:
    def __init__(
        self,
        max_df=0.8,
        min_df=0.01,
        max_features=100000,
        token_pattern="\w+",
        ngram_range=(1, 2),
    ):
        self.excluded = list(pd.read_csv("stopwords.txt", header=None)[0].values)
        self.lemmatizer = Mystem(grammar_info=False, disambiguation=False)
        self.tbl = dict.fromkeys(
            i
            for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith("P")
        )
        self.emojis = (
            "✨➢❗🚩🚚🚘🙌😼😻😭😜😗😖😔😏😎😍😌😊😉😄😃😀🕶🔴🔱🔥🔝🔗📷📲📦📣📏📌💳💰💯💭💧💦💥💣💞💑💐💏💎💍💋💆💄💃👰👫👤👠👟👙👗👔👏👍👌👆🐈🏼🏻🏢🏡🏠🏝🏂🎤🎊🎉🎈🎄🎂🎁🎀🍹🍬🍓🍏🌹🌸🌴🌡🌟🌞🌝🌛🌐🌏🌍🌊🌅🇹🇭⏰⌚⭐￼⬇➜➕❤️❀✔️✅⚠☺️️★●▬₽№™↓√≫⋙⇇"
        )
        self.others = "ﻩ❺❹❸❷❶⑧⑨②③①ᗒᗣᗕღლஜ"
        self.vectorizer = TfidfVectorizer(
            preprocessor=self._preprocess,
            tokenizer=self._tokenizer,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
        )

    def _remove_punctuation(self, text):
        text = text.translate({ord(c): None for c in self.emojis + self.others})
        return text.translate(self.tbl)

    def _preprocess(self, s):
        words = [
            w for w in self._remove_punctuation(s).split(" ") if w not in self.excluded
        ]
        return " ".join(words)

    def _tokenizer(self, s):
        self.cnt += 1
        if time.time() - self.last > 300:
            print("{} / {}".format(self.cnt, self.total))
            self.last = time.time()
        lemmas = self.lemmatizer.lemmatize(s)
        return [w for w in lemmas if w not in self.excluded]

    def fit(self, raw_documents):
        self.cnt = 0
        self.start = time.time()
        self.last = time.time()
        self.total = len(raw_documents)
        self.vectorizer.fit(raw_documents)

    def transform(self, raw_documents):
        self.cnt = 0
        self.start = time.time()
        self.last = time.time()
        self.total = len(raw_documents)
        return self.vectorizer.transform(raw_documents)
