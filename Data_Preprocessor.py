import numpy as np
import math
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class DataPreprocessor(Dataset):

    def __init__(self, path):
        self.path = path
        self.data = self.read_data_by_chunk()
        self.labels = np.asarray(self.data.iloc[:, 1])
        self.one_hot_data = self.get_one_hot_dataset()

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, index):
        return self.one_hot_data[index], self.labels[index]

    def open_csv(self, iterator=False):
        try:
            return pd.read_csv(self.path, sep=',', engine='python', iterator=iterator)
        except FileNotFoundError:
            print("file not found")

    def cut_smaller_data_for_exam(self, new_path, size=2000, skip_size=40000):
        print("producing smaller data")
        print("read file")
        data_file = self.open_csv(iterator=True)

        print("cut by chunk")
        chunk_size = 100
        loops = math.floor(size/chunk_size)
        chunks = []
        for loop in range(loops * 2):
            try:
                print("chunk ", loop)
                if loop % 2 == 0:  # avoid load data of same timestamp
                    chunk = data_file.get_chunk(chunk_size)
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

    def read_data_by_chunk(self, chunk_size=2000):
        print("reading large data")
        print("read file")
        data_file = self.open_csv(iterator=True)

        print("cut by chunk")
        loop = True
        chunks = []
        index = 0
        while loop:
            try:
                print("chunk: ", index)
                chunk = data_file.get_chunk(chunk_size)
                chunks.append(chunk)
                index += 1

            except StopIteration:
                loop = False
                print("Iteration is stopped.")

        print('start concact')
        data = pd.concat(chunks, ignore_index=True)

        print("data imported")
        return data

    def show_column_names(self):
        print("column #: {0}\ncolumns: {1}".format(self.data.shape[1], self.data.columns.tolist()))
        return self.data.shape[1], self.data.columns.tolist()

    def get_feature_set(self, feature_array):
        return np.array(np.unique(feature_array))

    def one_hot_encoding(self, feature_instances, feature_set):
            dimension = len(feature_set)
            one_hot_mat = np.zeros((len(feature_instances), dimension))
            for i in range(len(feature_instances)):
                if feature_instances[i] not in feature_set:
                    raise Exception("instance is not in the set")

                index = np.argwhere(feature_set == feature_instances[i])
                one_hot_mat[i, index] = 1

            return one_hot_mat

    def get_field_dims(self):
        dims = []
        for i in range(2, self.data.shape[1]):
            dims.append(len(self.get_feature_set(self.data.iloc[:, i])))
        return np.array(dims)

    def get_one_hot_dataset(self):
        one_hot_array = self.one_hot_encoding(np.asarray(self.data.iloc[:, 2]),
                                              self.get_feature_set(np.asarray(self.data.iloc[:, 2])))
        for i in range(3, self.data.shape[1]):
            next_array = self.one_hot_encoding(np.asarray(self.data.iloc[:, i]),
                                               self.get_feature_set(np.asarray(self.data.iloc[:, i])))
            one_hot_array = np.hstack((one_hot_array, next_array))
        return one_hot_array


if __name__ == "__main__":
    print("Start data cleaning")
    f = "./Data/smaller_train.csv"

    processor = DataPreprocessor(f)
    loader = DataLoader(processor, batch_size=10, shuffle=True)
    length = 0
    for data, label in loader:
        length += len(label)
