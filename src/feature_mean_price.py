import pandas as pd
import numpy as np


def run():
    usecols = ["item_id", "region", "city", "image_top_1", "price"]

    df = pd.read_csv("../input/train.csv", usecols=usecols)

    df_test = pd.read_csv("../input/test.csv", usecols=usecols)

    df = pd.concat([df, df_test], axis=0)

    city_mean = (
        df.groupby(["city"])
        .price.mean()
        .reset_index()
        .rename(columns={"price": "city_mean"})
    )

    region_mean = (
        df.groupby(["region"])
        .price.mean()
        .reset_index()
        .rename(columns={"price": "region_mean"})
    )

    image_top_1_mean = (
        df.groupby(["image_top_1"])
        .price.mean()
        .reset_index()
        .rename(columns={"price": "image_top_1_mean"})
    )

    o = df.merge(image_top_1_mean, on=["image_top_1"], how="left")
    o = o.merge(city_mean, on=["city"], how="left")
    o = o.merge(region_mean, on=["region"], how="left")

    for c in ["city_mean", "region_mean", "image_top_1_mean"]:
        m1, m2 = np.min(o[c]), np.max(o[c])
        o[c] = (o[c] - m1) / (m2 - m1)
        # o[c] = np.log1p(o[c])

    print(o.head())

    o.drop(usecols, axis=1, inplace=True)
    return o


if __name__ == "__main__":
    run()
