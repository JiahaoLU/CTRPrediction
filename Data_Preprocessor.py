import numpy as np
import math
import os
import pandas as pd


def open_csv(path, iterator=False):
    try:
        return pd.read_csv(path, sep=',', engine='python', iterator=iterator)
    except FileNotFoundError:
        print("file not found")


def cut_smaller_data(origin_path, new_path, size=2000, skip_size=40000):
    print("read file")
    data_file = open_csv(origin_path, iterator=True)

    print("cut by chunck")
    chunck_size = 100
    loops = math.floor(size/chunck_size)
    chunks = []
    for loop in range(loops * 2):
        try:
            print("chunck ", loop)
            if loop % 2 == 0:  # avoid load data of same timestamp
                chunk = data_file.get_chunk(chunck_size)
                chunks.append(chunk)
            else:
                chunk = data_file.get_chunk(skip_size)
        except StopIteration:
            print("Iteration is stopped.")

    print('start concat')
    data_frame = pd.concat(chunks, ignore_index=True)

    if os.path.exists(new_path):
        os.remove(new_path)
    data_frame.to_csv(new_path, index=False)
    print("new smaller file created")


def show_column_names(df):
    print("column #: {0}\ncolumns: {1}".format(df.shape[1], df.columns.tolist()))
    return df.shape[1], df.columns.tolist()


def get_feature_set(feature_array):
    return np.unique(feature_array)


def one_hot_encoding(feature_instance, feature_set):
        dimension = len(feature_set)
        one_hot_vector = np.zeros((len(feature_instance),dimension))
        for i in range(len(feature_instance)):
            if feature_instance[i] not in feature_set:
                raise Exception("instance is not in the set")

            index = np.argwhere(feature_set == feature_instance[i])
            one_hot_vector[i, index] = 1

        return one_hot_vector


if __name__ == "__main__":
    print("Start data cleaning")
    f = "./Data/smaller_train.csv"
    # cut_smaller_data("./Data/train.csv", f)

    train_df = open_csv(f)
    s1 = get_feature_set(np.array(train_df['C14']))
    one_hot = one_hot_encoding(train_df['C14'], s1)
    for i in one_hot:
        print(int(np.argwhere(i == 1)))
    # show_column_names(train_df)
