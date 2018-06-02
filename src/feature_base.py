import pandas as pd
import numpy as np

from sklearn import preprocessing


def run():
    categorical = [
        "user_id",
        "region",
        "city",
        "parent_category_name",
        "category_name",
        "user_type",
        "image_top_1",
        "param_1",
        "param_2",
        "param_3",
    ]

    usecols = ["activation_date", "price", "image"] + categorical

    df = pd.read_csv("../input/train.csv", usecols=usecols)
    df_test = pd.read_csv("../input/test.csv", usecols=usecols)
    df = pd.concat([df, df_test], axis=0)

    df["price"] = np.log(df["price"] + 0.001)
    df["price"].fillna(df.price.mean(), inplace=True)
    df["image_top_1"].fillna(-1, inplace=True)
    df["has_image"] = (~df["image"].isnull()).astype(np.uint8)
    df["weekday"] = pd.to_datetime(df.activation_date).dt.weekday
    categorical.append("weekday")

    lbl = preprocessing.LabelEncoder()
    for col in categorical:
        df[col].fillna("NA")
        df[col] = lbl.fit_transform(df[col].astype(str))

    df.drop(["activation_date", "image"], axis=1, inplace=True)

    print(df.columns)
    print(df.dtypes)
    return df


if __name__ == "__main__":
    run()
