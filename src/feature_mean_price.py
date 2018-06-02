import pandas as pd
import numpy as np


def run():
    df = pd.read_csv(
        "../input/train.csv", usecols=["item_id", "region", "city", "price"]
    )
    df_test = pd.read_csv(
        "../input/test.csv", usecols=["item_id", "region", "city", "price"]
    )
    df = pd.concat([df, df_test], axis=0)

    df["price"] = np.log1p(df.price)
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

    print(len(df))
    print(city_mean.head())
    print(region_mean.head())

    o = df.merge(city_mean, on=["city"], how="left")
    o = o.merge(region_mean, on=["region"], how="left")

    o.drop(["item_id", "region", "city", "price"], axis=1, inplace=True)
    return o


if __name__ == "__main__":
    run()
