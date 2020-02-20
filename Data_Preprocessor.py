import numpy as np
import math
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class DataPreprocessor(Dataset):

    def __init__(self, path=None):
        if path is not None:
            self.path = path
            self.data = self.read_data_by_chunk()
            self.labels = np.asarray(self.data.iloc[:, 1])
            self.one_hot_data = self.get_one_hot_dataset()
            print('Data set initiated from {0}.'.format(self.path))
        else:
            self.path = './Data/train.csv'
            print('Data provider is ready.')

    def __len__(self):
        return np.size(self.one_hot_data, 0)

    def __getitem__(self, index):
        return self.one_hot_data[index], self.labels[index]

    def open_csv(self, iterator=False):
        try:
            return pd.read_csv(self.path, sep=',', engine='python', iterator=iterator)
        except FileNotFoundError:
            print("file not found")

    def cut_smaller_data_for_exam(self, new_path, size=2000, skip_size=40000):
        print("Producing smaller data")
        print("Read file")
        data_file = self.open_csv(iterator=True)

        print("cut by chunk")
        chunk_size = 100
        loops = math.floor(size/chunk_size)
        chunks = []
        for loop in range(loops * 2):
            try:
                print("chunk: ", loop)
                if loop % 2 == 0:  # avoid load data of same timestamp
                    chunk = data_file.get_chunk(chunk_size)
                    chunks.append(chunk)
                else:
                    chunk = data_file.get_chunk(skip_size)
                    del chunk
            except StopIteration:
                print("Iteration is stopped.")

        print('Start concatenation')
        data_frame = pd.concat(chunks, ignore_index=True)

        if os.path.exists(new_path):
            os.remove(new_path)
        data_frame.to_csv(new_path, index=False)
        print("New smaller file created")

    def read_data_by_chunk(self, chunk_size=2000):
        print("Reading large data")
        data_file = self.open_csv(iterator=True)

        print("Cut by chunk")
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

        print('Start concatenation')
        whole_data = pd.concat(chunks, ignore_index=True)

        print("Data imported")
        return whole_data

    def get_feature_set(self, feature_array):
        return np.array(np.unique(feature_array))

    def one_hot_encoding(self, feature_instances, feature_set):
            one_hot_vector = np.zeros((len(feature_instances), 1), dtype=int)
            for i in range(len(feature_instances)):
                if feature_instances[i] not in feature_set:
                    raise Exception("instance is not in the set")

                index = np.argwhere(feature_set == feature_instances[i])
                one_hot_vector[i, 0] = int(index)

            return one_hot_vector

    def get_field_dims(self):
        dims = []
        for i in range(2, self.data.shape[1]):
            dims.append(len(self.get_feature_set(self.one_hot_data[:, i-2])))
        return np.array(dims)

    def get_one_hot_dataset(self):
        features = np.asarray(self.data.iloc[:, 2])
        one_hot_array = self.one_hot_encoding(features, self.get_feature_set(features))
        for i in range(3, self.data.shape[1]):
            features = np.asarray(self.data.iloc[:, i])
            next_array = self.one_hot_encoding(features, self.get_feature_set(features))
            one_hot_array = np.hstack((one_hot_array, next_array))
        return one_hot_array


if __name__ == "__main__":
    print('Start generating smaller data')
    provider = DataPreprocessor()
    provider.cut_smaller_data_for_exam('./Data/train20k.csv', size=20000, skip_size=40000)
    # print("Start data cleaning")
    # f = "./Data/train2k.csv"
    #
    # processor = DataPreprocessor(f)
    # print(processor.data.head())
    # loader = DataLoader(processor, batch_size=10, shuffle=True)
    # print(len(processor))
    # print(processor.get_field_dims())
    # for data, label in loader:
    #     print(data.size(), label)
