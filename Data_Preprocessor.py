import numpy as np
import math
import os
import pandas as pd


def open_csv(path, iterator=False):
    try:
        return pd.read_csv(path, sep=',', engine='python', iterator=iterator)
    except FileNotFoundError:
        print("file not found")


def cut_smaller_data(origin_path, new_path, size=2000):
    print("read file")
    data_file = open_csv(origin_path, iterator=True)

    print("cut by chunck")
    chunckSize = 1000
    loops = math.floor(size/chunckSize)
    chunks = []
    for loop in range(loops):
        try:
            print("chunck ", loop)
            chunk = data_file.get_chunk(chunckSize)
            chunks.append(chunk)
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


def one_hot_encoding(feature_instance, context):
    feature_set = list(set(context.tolist()))

    dimension = len(feature_set)




if __name__ == "__main__":
    print("Start data cleaning")
    f = "./Data/smaller_train.csv"
    cut_smaller_data("./Data/train.csv", f)
    train_df = open_csv(f)
    print(show_column_names(train_df))
    # show_column_names(train_df)
