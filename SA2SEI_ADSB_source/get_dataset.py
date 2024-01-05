import numpy as np
from sklearn.model_selection import train_test_split
import random

def get_num_class_pretraindata():
    x = np.load(f"/data1/liuhw/lc/Dataset_ADSB_without_icao/X_train_90Class.npy")
    y = np.load(f"/data1/liuhw/lc/Dataset_ADSB_without_icao/Y_train_90Class.npy")
    x = x.transpose(0, 2, 1)
    train_index_shot = []
    for i in range(90):
        index_classi = [index for index, value in enumerate(y) if value == i]
        train_index_shot += index_classi[0:100]
    return x[train_index_shot], y[train_index_shot]

def get_num_class_finetunedata(k):
    x = np.load(f"/data1/liuhw/lc/Dataset_ADSB_without_icao/X_train_10Class.npy")
    y = np.load(f"/data1/liuhw/lc/Dataset_ADSB_without_icao/Y_train_10Class.npy")
    x = x.transpose(0, 2, 1)
    x_test = np.load(f"/data1/liuhw/lc/Dataset_ADSB_without_icao/X_test_10Class.npy")
    y_test = np.load(f"/data1/liuhw/lc/Dataset_ADSB_without_icao/Y_test_10Class.npy")
    x_test = x_test.transpose(0, 2, 1)
    finetune_index_shot = []
    for i in range(10):
        index_classi = [index for index, value in enumerate(y) if value == i]
        finetune_index_shot += random.sample(index_classi, k)
    return x[finetune_index_shot], x_test, y[finetune_index_shot], y_test

def PreTrainDataset_prepared():
    X_train_ul, Y_train_ul = get_num_class_pretraindata()
    Y_train_ul = Y_train_ul.astype(np.uint8)

    min_value = X_train_ul.min()
    max_value = X_train_ul.max()

    X_train_ul = (X_train_ul - min_value) / (max_value - min_value)

    return X_train_ul, Y_train_ul

def FineTuneDataset_prepared(k):
    X_train, X_test, Y_train, Y_test = get_num_class_finetunedata(k)
    Y_train = Y_train.astype(np.uint8)
    Y_test = Y_test.astype(np.uint8)

    max_value = X_train.max()
    min_value = X_train.min()

    X_train = (X_train - min_value) / (max_value - min_value)
    X_test = (X_test - min_value) / (max_value - min_value)

    return X_train, X_test, Y_train, Y_test

