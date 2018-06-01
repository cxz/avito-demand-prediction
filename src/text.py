import unicodedata
import sys
import time
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.snowball import SnowballStemmer


class Emoji:
    def __init__(self):
        self.excluded = list(stopwords.words("russian"))
        self.tbl = dict.fromkeys(
            i
            for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith("P")
        )
        self.emojis = (
            "✨➢❗🚩🚚🚘🙌😼😻😭😜😗😖😔😏😎😍😌😊😉😄😃😀🕶🔴🔱🔥🔝🔗📷📲📦📣📏📌💳💰💯💭💧💦💥💣💞💑💐💏💎💍💋💆💄💃👰👫👤👠👟👙👗👔👏👍👌👆🐈🏼🏻🏢🏡🏠🏝🏂🎤🎊🎉🎈🎄🎂🎁🎀🍹🍬🍓🍏🌹🌸🌴🌡🌟🌞🌝🌛🌐🌏🌍🌊🌅🇹🇭⏰⌚⭐￼⬇➜➕❤️❀✔️✅⚠☺️️★●▬₽№™↓√≫⋙⇇"
        )
        self.others = "ﻩ❺❹❸❷❶⑧⑨②③①ᗒᗣᗕღლஜ"

    def _remove_punctuation(self, text):
        text = text.translate({ord(c): None for c in self.emojis + self.others})
        return text.translate(self.tbl)

    def _preprocess(self, s):
        words = [
            w for w in self._remove_punctuation(s).split(" ") if w not in self.excluded
        ]
        return " ".join(words)

    def count(self, text):
        emojis = set(self.emojis) | set(self.others)
        return len([c for c in str(text) if c in emojis])


class Cleanup:
    def __init__(self):
        self.x = None
        self.stemmer = SnowballStemmer("russian")
        self.stop_words = set(stopwords.words("russian"))

    def process(self, text):
        tokens = wordpunct_tokenize(text)
        tokens = [w.lower() for w in tokens]
        words = [
            self.stemmer.stem(word)
            for word in tokens
            if word.isalpha() and word not in self.stop_words
        ]
        return " ".join(words)


if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv", nrows=1000, usecols=["description", "title"])
    df.description.fillna("", inplace=True)

    # c = Cleanup()
    # for d in df.description.values:
    #    print('1)', d)
    #    print('2)', c.process(d))
    #    print('=')

    e = Emoji()
    for d in df.description.values:
        x = e.count(d)
        if x > 0:
            print(x)
