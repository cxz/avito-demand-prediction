import glob
import os
import csv

os.environ["KERAS_BACKEND"] = "tensorflow"

from functools import partial
from PIL import Image
import cv2
from datetime import datetime
import pandas as pd

from keras.preprocessing import image
from timeit import default_timer as timer

# import faulthandler
# faulthandler.enable()

import cProfile

import code, traceback, signal


def debug(sig, frame):
    """Interrupt running process, and provide a python prompt for
    interactive debugging."""
    d = {"_frame": frame}  # Allow access to frame object.
    d.update(frame.f_globals)  # Unless shadowed by globalq
    d.update(frame.f_locals)

    i = code.InteractiveConsole(d)
    message = "Signal received : entering python shell.\nTraceback:\n"
    message += "".join(traceback.format_stack(frame))
    i.interact(message)


def listen():
    signal.signal(signal.SIGUSR1, debug)  # Register handler


def resnet():
    from keras.applications.resnet50 import ResNet50
    from keras.applications.resnet50 import preprocess_input, decode_predictions

    model = ResNet50(weights="imagenet")
    target_size = (224, 224)
    preprocess_fn = preprocess_input
    decode_fn = partial(decode_predictions, top=1)
    return "resnet50_imagenet", model, target_size, preprocess_fn, decode_fn


def xception():
    from keras.applications.xception import Xception
    from keras.applications.xception import preprocess_input, decode_predictions

    model = Xception(weights="imagenet")
    target_size = (299, 299)
    preprocess_fn = preprocess_input
    decode_fn = partial(decode_predictions, top=1)
    return "xception_imagenet", model, target_size, preprocess_fn, decode_fn


def vgg16():
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg16 import preprocess_input, decode_predictions

    model = VGG16(weights="imagenet")
    target_size = (224, 224)
    preprocess_fn = preprocess_input
    decode_fn = partial(decode_predictions, top=1)
    return "vgg16_imagenet", model, target_size, preprocess_fn, decode_fn


def inceptionresnetv2():
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.applications.inception_resnet_v2 import (
        preprocess_input,
        decode_predictions,
    )

    model = InceptionResNetV2(weights="imagenet")
    target_size = (299, 299)
    preprocess_fn = preprocess_input
    decode_fn = partial(decode_predictions, top=1)
    return "inceptionresnetv2_imagenet", model, target_size, preprocess_fn, decode_fn


batch_size = 128
limit = None  # Limit number of images processed (useful for debug)
bar_iterval = 10  # in seconds

import numpy as np
import zipfile
from tqdm import tqdm
from pathlib import PurePath
import gc


try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

_PIL_INTERPOLATION_METHODS = {
    "nearest": pil_image.NEAREST,
    "bilinear": pil_image.BILINEAR,
    "bicubic": pil_image.BICUBIC,
}


def resize(img, target_size=None, interpolation="nearest"):
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    "Invalid interpolation method {} specified. Supported "
                    "methods are {}".format(
                        interpolation, ", ".join(_PIL_INTERPOLATION_METHODS.keys())
                    )
                )
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


def predict_batch(model, preprocess_fn, decode_fn, X_batch):
    # X_batch = np.array([image.img_to_array(x) for x in X_batch])
    X_batch = preprocess_fn(X_batch)
    features_batch = decode_fn(model.predict_on_batch(X_batch))
    return features_batch


def process(ds_name, model_name, model, target_size, preprocess_fn, decode_fn):

    fnames = list(glob.glob(f"../data/{ds_name}/**/*jpg"))
    items_ids = [os.path.basename(fname).split(".")[0] for fname in fnames]
    bar = None

    X_batch = np.zeros(
        (batch_size, target_size[0], target_size[1], 3), dtype=np.float32
    )
    batch_item_ids = []
    batch_idx = 0

    # result = []

    profiler = cProfile.Profile()
    profile_status = False

    p_1 = 0
    p_2 = 1

    out_fname = "{}_{}.csv".format(model_name, ds_name)

    if os.path.exists(out_fname):
        df = pd.read_csv(out_fname)
        existing = set(df.item_id.values)
    else:
        existing = set()

    with open(out_fname, "a+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["item_id", "top_1", "top_1_name", "top_1_score"])
        for idx, fname in enumerate(fnames):
            if idx % 10000 == 0:
                print(str(datetime.now()), idx, p_1 / p_2)
                p_1 = 0
                p_2 = 1

            # if idx > 300000:
            #    if idx % 50000 == 0:
            #        if profile_status:
            #            profile_status = False
            #            profiler.disable()
            #            profiler.dump_stats("profiler_{}.txt".format(idx))
            #        else:
            #            profile_status = True
            #            profiler.enable()

            item_id = os.path.basename(fname).split(".")[0]
            if item_id in existing:
                continue
            try:
                # im = Image.open(fname)

                x1 = timer()
                im = cv2.imread(fname)
                if im.shape is None or len(im.shape) != 3:
                    im = np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
                else:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                x2 = timer()
                p_1 += x2 - x1
                p_2 += 1
            except:
                print(fname)
                # im = Image.new('RGB', target_size)
                im = np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)

            # if bar is None:
            #    bar = tqdm(total=len(fnames), mininterval=bar_iterval, unit_scale=True)

            X_batch[batch_idx, : im.shape[0], : im.shape[1], :] = im
            batch_item_ids.append(item_id)
            batch_idx += 1

            if batch_idx == batch_size:
                for item_id, row in zip(
                    batch_item_ids,
                    predict_batch(model, preprocess_fn, decode_fn, X_batch),
                ):
                    writer.writerow([item_id] + list(row[0]))
                batch_item_ids = []
                batch_idx = 0
                # bar.update(X_batch.shape[0])

        # Predict last batch
        if batch_idx > 0:
            for item_id, row in zip(
                batch_item_ids, predict_batch(model, preprocess_fn, decode_fn, X_batch)
            ):
                writer.writerow([item_id] + list(row[0]))


def run_test():
    fname = "../input/{}_jpg.zip".format("train")
    print(fname)
    zip = zipfile.ZipFile(fname)
    items = zip.infolist()
    for idx, zinfo in tqdm(enumerate(items), total=len(items)):
        if zinfo.filename.endswith(".jpg"):
            zpath = PurePath(zinfo.filename)
            item_id = zpath.stem
            with zip.open(zinfo) as file:
                f = file.read()


def extract_worker(idx, zip_fname, prefix, fnames, target_size):
    zip = zipfile.ZipFile(zip_fname)
    items = zip.infolist()
    for idx, zinfo in tqdm(enumerate(items), total=len(items), position=idx):
        if zinfo.filename.endswith(".jpg"):
            zpath = PurePath(zinfo.filename)
            item_id = zpath.stem

            out_dir = os.path.join("../data", prefix, item_id[:3])
            os.makedirs(out_dir, exist_ok=True)
            out_fname = os.path.join(out_dir, "{}.jpg".format(item_id))

            if os.path.exists(out_fname):
                continue

            with zip.open(zinfo) as file:
                try:
                    img = Image.open(file)
                except:
                    img = Image.new("RGB", target_size)
                img = resize(img, target_size)

            img.save(out_fname)
        else:
            print(zinfo.filename)


def extract(ds_name, target_size, workers):
    """ Extract images from zip and resize.    
    """
    zip_fname = "../input/{}_jpg.zip".format(ds_name)
    extract_worker(0, zip_fname, ds_name, None, target_size)


def load(ds_name, target_size):
    fnames = list(sorted(glob.glob(f"../data/{ds_name}/**/*jpg")))
    db = np.zeros((len(fnames), target_size[0], target_size[1], 3), dtype=np.float32)
    for idx, fname in enumerate(fnames):
        item_id = os.path.basename(fname).split(".")[0]
        try:
            im = Image.open(fname)
        except:
            im = Image.new("RGB", target_size)
        db[idx] = im
    np.save(f"{ds_name}.npy", db)


def load_csv(model_name, kind):
    df = pd.read_csv(f"../src/{model_name}_imagenet_{kind}.csv")
    # cleanup multiple headers
    df = df[df.item_id != "item_id"]
    df.drop("top_1", axis=1, inplace=True)
    column_names = [
        c if c == "item_id" else "%s_%s" % (c, model_name) for c in df.columns
    ]
    df.columns = column_names
    return df


def merge_csvs(kind):
    df = load_csv("resnet50", kind)
    for model_name in [
        "inceptionresnetv2",
        # 'resnet50',
        "xception",
        "vgg16",
    ]:
        df_ = load_csv(model_name, kind)
        df = df.merge(df_, on="item_id")
    return df


def prepare():
    df_train = merge_csvs("train")
    df_train.to_csv("cnn_train.csv", index=False)

    df_test = merge_csvs("test")
    df_test.to_csv("cnn_test.csv", index=False)


def run():
    df = pd.read_csv("../input/train.csv", usecols=["item_id", "image"])
    df_test = pd.read_csv("../input/test.csv", usecols=["item_id", "image"])
    df = pd.concat([df, df_test], axis=0)
    rows = len(df)

    # merge cnn predictions with train/test df
    cnn = pd.read_csv("../cache/cnn_train.csv")
    cnn_test = pd.read_csv("../cache/cnn_test.csv")
    cnn = pd.concat([cnn, cnn_test], axis=0)
    cnn.rename(columns={"item_id": "image"}, inplace=True)

    df = df.merge(cnn, on="image", how="left")
    assert rows == len(df)

    # top_1_name_resnet50, top_1_score_resnet50,
    # top_1_name_inceptionresnetv2, top_1_score_inceptionresnetv2,
    # top_1_name_xception,top_1_score_xception,top_1_name_vgg16,top_1_score_vgg16
    from sklearn import preprocessing

    lbl = preprocessing.LabelEncoder()

    column_names = [x for x in df.columns if x.startswith("top_1_name_")]

    values = set()
    for c in column_names:
        df[c].fillna("", inplace=True)
        values = values.union(df[c].values)

    lbl.fit(list(values))

    for c in column_names:
        df[c] = lbl.transform(df[c].astype(str))

    df.drop(["image", "item_id"], axis=1, inplace=True)
    print(df.dtypes)
    print(df.head())

    return df[["top_1_name_resnet50", "top_1_name_vgg16"]]


if __name__ == "__main__":
    # run_test()
    # workers = 4
    # extract('train', (224, 224), workers)
    # extract('test', (224, 224), workers)

    # process('train', *resnet())
    # process('test', *resnet())

    # process('train', *inceptionresnetv2())
    # process('test', *inceptionresnetv2())

    # process('train', *xception())
    # process('test', *xception())

    # process('train', *vgg16())
    # process('test', *vgg16())

    # merge_csvs()

    print("done.")
