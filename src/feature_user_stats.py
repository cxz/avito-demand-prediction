"""
from https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm
"""

import gc
import pandas as pd
import numpy as np


def run():
    used_cols = ["item_id", "user_id"]

    train = pd.read_csv("../input/train.csv", usecols=used_cols)
    train_active = pd.read_csv("../input/train_active.csv", usecols=used_cols)

    test = pd.read_csv("../input/test.csv", usecols=used_cols)
    test_active = pd.read_csv("../input/test_active.csv", usecols=used_cols)

    train_periods = pd.read_csv(
        "../input/periods_train.csv", parse_dates=["date_from", "date_to"]
    )
    test_periods = pd.read_csv(
        "../input/periods_test.csv", parse_dates=["date_from", "date_to"]
    )

    print("loaded csvs")

    all_samples = pd.concat([train, train_active, test, test_active]).reset_index(
        drop=True
    )
    all_samples.drop_duplicates(["item_id"], inplace=True)

    del train_active
    del test_active
    gc.collect()

    all_periods = pd.concat([train_periods, test_periods])

    del train_periods
    del test_periods
    gc.collect()

    print("samples & periods df ok")

    all_periods["days_up"] = (
        all_periods["date_to"].dt.dayofyear - all_periods["date_from"].dt.dayofyear
    )

    gp = all_periods.groupby(["item_id"])[["days_up"]]

    gp_df = pd.DataFrame()
    gp_df["days_up_sum"] = gp.sum()["days_up"]
    gp_df["times_put_up"] = gp.count()["days_up"]
    gp_df.reset_index(inplace=True)
    gp_df.rename(index=str, columns={"index": "item_id"})

    all_periods.drop_duplicates(["item_id"], inplace=True)
    all_periods = all_periods.merge(gp_df, on="item_id", how="left")

    all_periods = all_periods.merge(all_samples, on="item_id", how="left")

    gp = (
        all_periods.groupby(["user_id"])[["days_up_sum", "times_put_up"]]
        .mean()
        .reset_index()
        .rename(
            index=str,
            columns={
                "days_up_sum": "avg_days_up_user",
                "times_put_up": "avg_times_up_user",
            },
        )
    )

    n_user_items = (
        all_samples.groupby(["user_id"])[["item_id"]]
        .count()
        .reset_index()
        .rename(index=str, columns={"item_id": "n_user_items"})
    )
    gp = gp.merge(n_user_items, on="user_id", how="outer")

    print("features generated")

    # gp columns: user_id, avg_days_up_user, avg_times_up_user, n_user_items
    df = pd.read_csv("../input/train.csv", usecols=["user_id"])
    df_test = pd.read_csv("../input/test.csv", usecols=["user_id"])
    df = pd.concat([df, df_test], axis=0)

    df = df.merge(gp, on="user_id", how="left")
    df.drop(["user_id"], axis=1, inplace=True)

    print("done.")

    return df


if __name__ == "__main__":
    run()
