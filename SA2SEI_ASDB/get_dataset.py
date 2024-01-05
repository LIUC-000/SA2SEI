import numpy as np
import math
import json
import h5py
import random
import yaml

def PreTrainDataset_prepared():
    x = np.load(f"/data/liuc/Dataset/ADS-B_4800_without_icao/X_train_90Class.npy")
    y = np.load(f"/data/liuc/Dataset/ADS-B_4800_without_icao/Y_train_90Class.npy")
    x = x.transpose(0, 2, 1)
    train_index_shot = []
    for i in range(90):
        index_classi = [index for index, value in enumerate(y) if value == i]
        train_index_shot += index_classi[0:100]

    X_train = x[train_index_shot]
    Y_train = y[train_index_shot].astype(np.uint8)
    max_value = X_train.max()
    min_value = X_train.min()

    X_train = (X_train - min_value) / (max_value - min_value)
    return X_train, Y_train

def FineTuneDataset_prepared():
    config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
    params = config['finetune']
    k = params['k_shot']
    x = np.load(f"/data/liuc/Dataset/ADS-B_4800_without_icao/X_train_10Class.npy")
    y = np.load(f"/data/liuc/Dataset/ADS-B_4800_without_icao/Y_train_10Class.npy")
    x = x.transpose(0, 2, 1)
    X_test = np.load(f"/data/liuc/Dataset/ADS-B_4800_without_icao/X_test_10Class.npy")
    Y_test = np.load(f"/data/liuc/Dataset/ADS-B_4800_without_icao/Y_test_10Class.npy")
    X_test = X_test.transpose(0, 2, 1)
    finetune_index_shot = []
    for i in range(10):
        index_classi = [index for index, value in enumerate(y) if value == i]
        finetune_index_shot += random.sample(index_classi, k)
    X_train = x[finetune_index_shot]
    Y_train = y[finetune_index_shot]

    max_value = X_train.max()
    min_value = X_train.min()

    X_train = (X_train - min_value) / (max_value - min_value)
    X_test = (X_test - min_value) / (max_value - min_value)
    return X_train, X_test, Y_train, Y_test

if __name__ == "__main__":
    x = np.load('/data/liuc/Dataset/ADS-B_4800_without_icao/X_train_100Class.npy')
    y = np.load('/data/liuc/Dataset/ADS-B_4800_without_icao/Y_train_100Class.npy')
    print(x.shape)
    print(y.shape)
    count = 0
    for i in range(0, 100):
        index_classi_len = len([index for index, value in enumerate(y) if value == i])
        if (index_classi_len) >= 250:
            print(f'class{i},size{index_classi_len}')
            count += 1
    print(count)