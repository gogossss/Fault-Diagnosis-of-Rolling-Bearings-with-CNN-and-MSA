import torch
from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同
import random

from torch.utils import data


def prepro(d_path, length=0, number=0, normal=True, rate=[0, 0, 0], enc=False, enc_step=28, random_seed=42):
    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)
    np.random.seed(random_seed)
    random.seed(random_seed)

    def capture(original_path):
        files = {}
        print(filenames)
        for i in filenames:
            # 文件路径
            file_path = os.path.join(d_path, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if 'B_A_1.mat' in filenames:
                    files[i] = np.array(file[key]).ravel()
                else:
                    if 'DE' in key:
                        files[i] = file[key].ravel()
        return files

    def slice_enc(data, slice_rate=rate[1] + rate[2]):
        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]

            all_lenght = len(slice_data)
            # end_index = int(all_lenght * (1 - slice_rate))
            samp_train = int(number * (1 - slice_rate))  # 1000(1-0.3)
            Train_sample = []
            Test_Sample = []

            for j in range(samp_train):
                sample = slice_data[j * 150: j * 150 + length]
                Train_sample.append(sample)

            # 抓取测试数据
            for h in range(number - samp_train):
                sample = slice_data[samp_train * 150 + length + h * 150: samp_train * 150 + length + h * 150 + length]
                Test_Sample.append(sample)
            Train_Samples[i] = Train_sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples

    # 仅抽样完成，打标签
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        for i in filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        return X, Y

    def scalar_stand(Train_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        data_all = np.vstack((Train_X, Test_X))
        scalar = preprocessing.StandardScaler().fit(data_all)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    def valid_test_slice(Test_X, Test_Y):

        test_size = rate[2] / (rate[1] + rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)

        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]

            return X_valid, Y_valid, X_test, Y_test

    # 从所有.mat文件中读取出数据的字典
    data = capture(original_path=d_path)
    # 将数据切分为训练集、测试集
    train, test = slice_enc(data)
    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test)

    # 训练数据/测试数据 是否标准化.
    # if normal:
    #     Train_X, Test_X = scalar_stand(Train_X, Test_X)
    Train_X = np.asarray(Train_X)
    Test_X = np.asarray(Test_X)
    # 将测试集切分为验证集和测试集.
    Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, Test_Y)

    train_data = {}
    for i in range(len(Train_X)):
        if Train_Y[i] in train_data.keys():
            train_data[Train_Y[i]] += [{"tokens": Train_X[i]}]
        else:
            train_data[Train_Y[i]] = [{"tokens": Train_X[i]}]

    test_data = {}
    for i in range(len(Test_X)):
        if Test_Y[i] in test_data.keys():
            test_data[Test_Y[i]] += [{"tokens": Test_X[i]}]
        else:
            test_data[Test_Y[i]] = [{"tokens": Test_X[i]}]

    val_data = {}
    for i in range(len(Valid_X)):
        if Valid_Y[i] in val_data.keys():
            val_data[Valid_Y[i]] += [{"tokens": Valid_X[i]}]
        else:
            val_data[Valid_Y[i]] = [{"tokens": Valid_X[i]}]

    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y, train_data, test_data, val_data


class FewRelDataset(data.Dataset):
    def __init__(self, name, encoder, N, K, Q, na_rate):
        self.data = name
        self.classes \
            = list(self.data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, mask = self.encoder.tokenize(item['tokens'])
        return word, mask

    def __additem__(self, d, word, mask):
        d['word'].append(word)
        d['mask'].append(mask)

    def __getitem__(self, index):
        # np.random.seed(1)
        # random.seed(1)
        
        target_classes = random.sample(self.classes, self.N)
        support_set = {'word': [], 'mask': []}
        query_set = {'word': [], 'mask': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,
                                 self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                list(range(len(self.data[i]))),
                self.K + self.Q, False)
            count = 0
            for j in indices:
                word, mask = self.__getraw__(self.data[i][j])
                word = torch.tensor(word).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, mask)
                else:
                    self.__additem__(query_set, word, mask)
                count += 1

            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                list(range(len(self.data[cur_class]))),
                1, False)[0]
            word, mask = self.__getraw__(self.data[cur_class][index])
            word = torch.tensor(word).long()
            mask = torch.tensor(mask).long()
            self.__additem__(query_set, word, mask)
        query_label += [self.N] * Q_na

        return support_set, query_set, query_label

    def __len__(self):
        return 1000000000


def collate_fn(data):
    batch_support = {'word': [], 'mask': []}
    batch_query = {'word': [], 'mask': []}
    batch_label = []
    support_sets, query_sets, query_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label


def get_loader(name, encoder, N, K, Q, batch_size,
               num_workers=0, collate_fn=collate_fn, na_rate=0, ):
    dataset = FewRelDataset(name, encoder, N, K, Q, na_rate)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)

