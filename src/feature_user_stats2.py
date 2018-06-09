import gc
import pandas as pd
import numpy as np


def run():
    used_cols = ["item_id", "user_id"]

    train = pd.read_csv("../input/train.csv", usecols=used_cols)
    train_active = pd.read_csv("../input/train_active.csv", usecols=used_cols)

    test = pd.read_csv("../input/test.csv", usecols=used_cols)
    test_active = pd.read_csv("../input/test_active.csv", usecols=used_cols)

    date_cols = ["date_from", "date_to", "activation_date"]

    train_periods = pd.read_csv("../input/periods_train.csv", parse_dates=date_cols)
    test_periods = pd.read_csv("../input/periods_test.csv", parse_dates=date_cols)

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

    all_periods["days_up"] = (
        all_periods["date_to"].dt.dayofyear - all_periods["date_from"].dt.dayofyear
    )

    all_periods["start_lag"] = (
        all_periods["date_from"].dt.dayofyear
        - all_periods["activation_date"].dt.dayofyear
    )

    gp = all_periods.groupby(["item_id"])

    gp_df = pd.DataFrame()
    gp_df["days_up_sum"] = gp.sum()["days_up"]
    gp_df["days_up_min"] = gp.min()["days_up"]
    gp_df["days_up_max"] = gp.max()["days_up"]
    gp_df["days_up_count"] = gp.count()["days_up"]

    gp_df["start_lag_sum"] = gp.sum()["start_lag"]
    gp_df["start_lag_min"] = gp.min()["start_lag"]
    gp_df["start_lag_max"] = gp.max()["start_lag"]

    gp_df.reset_index(inplace=True)
    gp_df.rename(index=str, columns={"index": "item_id"})

    all_periods.drop_duplicates(["item_id"], inplace=True)
    all_periods = all_periods.merge(gp_df, on="item_id", how="left")
    all_periods = all_periods.merge(all_samples, on="item_id", how="left")

    gp = all_periods.groupby(["user_id"]).agg(
        {
            "days_up_sum": ["mean", "min", "max"],
            "days_up_max": ["mean", "min", "max"],
            "days_up_count": ["mean", "min", "max"],
            "start_lag_sum": ["mean", "min", "max"],
            "start_lag_min": ["mean", "min", "max"],
            "start_lag_max": ["mean", "min", "max"],
        }
    )

    gp.columns = ["_".join(col).strip() for col in gp.columns.values]
    n_user_items = (
        all_samples.groupby(["user_id"])[["item_id"]]
        .count()
        .reset_index()
        .rename(index=str, columns={"item_id": "n_user_items"})
    )
    gp = gp.merge(n_user_items, on="user_id", how="outer")

    # gp columns: user_id, avg_days_up_user, avg_times_up_user, n_user_items
    df = pd.read_csv("../input/train.csv", usecols=["user_id"])
    df_test = pd.read_csv("../input/test.csv", usecols=["user_id"])
    df = pd.concat([df, df_test], axis=0)

    df = df.merge(gp, on="user_id", how="left")
    df.drop(["user_id"], axis=1, inplace=True)

    return df


if __name__ == "__main__":
    run()
