import numpy as np
import pandas as pd


def open_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print("file not found")


def cut_smaller_data(origin_path, new_path, size=1000):
    data_frame = open_csv(origin_path)

    if data_frame.shape[0] >= size:
        small_data = data_frame.head(size)
        small_data.to_csv(new_path, index=True)
        print("new smaller file created")
    else:
        print("origin data size smaller than {0}".format(size))


def show_column_names(df):
    print("column #: {0}\ncolumns: {1}".format(df.shape[1], df.columns.tolist))
    return df.shape[1], df.columns.tolist

if __name__ == "__main__":
    print("Start data cleaning")
    # cut_smaller_data("./Data/train.csv", "./Data/smaller_train.csv")
