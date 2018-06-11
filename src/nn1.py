import tensorflow as tf
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras


from keras.layers import *
from keras.callbacks import *
from keras.models import Model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

import pandas as pd
import numpy as np
from tqdm import tqdm


from util import setup_logs
import data
import fasttext

logger = setup_logs("", "../tmp/tmp.log")


def build_model(E, numerical, sequence):
    # numerical inputs:
    # price, item_seq_number, ...
    numerical = Input(shape=(numerical,), name="numerical")
    n1 = Reshape((1, -1))(numerical)

    # i0 = Input(shape=(1,), name='user_id')
    # e0 = Embedding(1009906, 14)(i0)
    i1 = Input(shape=(1,), name="region")
    e1 = Embedding(27, 4)(i1)
    i2 = Input(shape=(1,), name="city")
    e2 = Embedding(1751, 8)(i2)
    i3 = Input(shape=(1,), name="parent_category_name")
    e3 = Embedding(8, 3)(i3)
    i4 = Input(shape=(1,), name="category_name")
    e4 = Embedding(46, 4)(i4)
    # i5 = Input(shape=(1, ), name='item_seq_number')
    # e5 = Embedding(33945, 11)(i5)
    i6 = Input(shape=(1,), name="user_type")
    e6 = Embedding(2, 2)(i6)
    i7 = Input(shape=(1,), name="image_top_1")
    e7 = Embedding(3063, 9)(i7)
    i8 = Input(shape=(1,), name="param_1")
    e8 = Embedding(371, 6)(i8)
    i9 = Input(shape=(1,), name="param_2")
    e9 = Embedding(277, 6)(i9)
    i10 = Input(shape=(1,), name="param_3")
    e10 = Embedding(1275, 8)(i10)
    i11 = Input(shape=(1,), name="weekday")
    e11 = Embedding(6, 2)(i11)
    # i13 = Input(shape=(1,), name="has_image")
    # e13 = Embedding(1, 1)(i13)

    i14 = Input(shape=(1,), name="top_1_name_resnet50")
    e14 = Embedding(1, 1)(i14)

    i15 = Input(shape=(1,), name="top_1_name_vgg16")
    e15 = Embedding(1, 1)(i15)

    # sequence inputs
    sequence = Input(shape=(sequence,), name="sequence")
    embedding = Embedding(E.shape[0], E.shape[1], weights=[E], trainable=False)(
        sequence
    )
    x = SpatialDropout1D(0.1)(embedding)
    # x = Bidirectional(GRU(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = Conv1D(32, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(
        x
    )
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    inputs = concatenate(
        [
            # e0,
            e1,
            e2,
            e3,
            e4,
            # e5,
            e6,
            e7,
            e8,
            e9,
            e10,
            e11,
            # e13,
            # e14,
            # e15,
            n1,
            Reshape((-1, 32))(avg_pool),
            Reshape((-1, 32))(max_pool),
        ]
    )

    x = Dense(256, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Flatten()(x)
    predictions = Dense(1, activation="linear")(x)

    keras_input = [  # i0,
        i1,
        i2,
        i3,
        i4,
        i6,
        i7,
        i8,
        i9,
        i10,
        i11,
        # i13,
        # i14,
        # i15,
        numerical,
        sequence,
    ]
    model = Model(inputs=keras_input, outputs=predictions)

    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam, loss="mse", metrics=[])
    return model


def build_input(X, text, X_tfidf2, X_target_encoded, X_user_stats, X_image_cnn):
    X_dict = dict([(c, X[c].values) for c in X.columns])

    # print(X_image_cnn.head())
    # for c in X_image_cnn.columns:
    #    if c.startswith("top_1_name"):
    #        print('adding ', c)
    #        X_dict[c] = X_image_cnn[c]

    X_dict["numerical"] = np.hstack(
        [
            X[["price", "item_seq_number"]].values,
            X_tfidf2.values,
            # X_target_encoded.values,
            # X_user_stats.values
        ]
    )
    X_dict["sequence"] = text
    return X_dict


def run(model, X_train, y_train, X_val, y_val):
    ckpt = ModelCheckpoint(
        "../tmp/weights.{epoch:02d}-{val_loss:.4f}.hdf5", verbose=1, save_best_only=True
    )

    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5)
    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr, ckpt],
    )

    y_val_pred = model.predict(X_val, verbose=1, batch_size=1024)
    print(np.sqrt(mse(y_val, y_val_pred)))


def main(model=None):
    X = data.load_traintestX_base()
    y = np.load("../cache/20180601_trainy.npy")
    ntrain = y.shape[0]

    X = X.iloc[: y.shape[0]]
    logger.info(f"{X.shape} columns:{' '.join(X.columns)}")

    text_vec = np.load("../cache/title-description-seq100.npy")
    text_vec = text_vec[: y.shape[0]]
    logger.info(f"text shape: {text_vec.shape}")

    # X_ridge1 = data.load_traintestX_tfidf1_ridge()
    # X_ridge1 = X_ridge1[:ntrain]

    X_tfidf2 = data.load_traintestX_tfidf2()  # df

    # X_mean_price = data.load_traintestX_mean_price()
    X_target_encoded = data.load_traintestX_target_encoded()

    X_user_stats = data.load_traintestX_user_stats2()

    X_image_cnn = data.load_traintestX_image_cnn()

    train_idx, val_idx = train_test_split(range(ntrain), test_size=0.2)

    X_train_dict = build_input(
        X.iloc[train_idx],
        text_vec[train_idx],
        X_tfidf2.iloc[train_idx],
        # X_mean_price.iloc[train_idx]
        X_target_encoded.iloc[train_idx],
        X_user_stats.iloc[train_idx],
        X_image_cnn.iloc[train_idx],
    )
    y_train = y[train_idx]

    X_val_dict = build_input(
        X.iloc[val_idx],
        text_vec[val_idx],
        X_tfidf2.iloc[val_idx],
        # X_mean_price.iloc[val_idx]
        X_target_encoded.iloc[val_idx],
        X_user_stats.iloc[val_idx],
        X_image_cnn.iloc[val_idx],
    )

    y_val = y[val_idx]

    logger.info("train/val loaded.")

    logger.info("loading fasttext weights..")
    E = fasttext.load_cached()

    if model is None:
        model = build_model(
            E, X_train_dict["numerical"].shape[1], X_train_dict["sequence"].shape[1]
        )

        run(model, X_train_dict, y_train, X_val_dict, y_val)
    else:
        y_val_pred = model.predict(X_val_dict, verbose=1, batch_size=1024)
        print(np.sqrt(mse(y_val, y_val_pred)))


if __name__ == "__main__":
    # model = load_model("../tmp/weights.10-0.0514.hdf5")
    model = None
    main(model)
