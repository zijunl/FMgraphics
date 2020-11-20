## the pipeline.py build the data pipeline for 32*32 pixel space images(original). 
# The data pipeline send the data to the training client by batches


import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import numpy as np
import os.path as osp
from tqdm import tqdm
from PIL import Image

from utils import Config

import pickle


class celeba32_dataset:
    def __init__(self):
        self.image_dir = Config['path_output']
        self.transforms = self.get_data_transforms()
        # self.X_train, self.X_test, self.y_train, self.y_test, self.classes = self.create_dataset()



    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms

    def create_dataset(self):
        # map id to category
    
        # create X, y pairs
        files = os.listdir(self.image_dir)
        pickle_in = open("/home/ubuntu/python/labels.pickle","rb")
        y = pickle.load(pickle_in)
        X = []
        for x in files:
            X.append(x)
        y = LabelEncoder().fit_transform(y)

        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test, max(y) + 1

# training data class
class celeba32_train(Dataset):
    def __init__(self, X_train, y_train, transform):
        self.X_train = X_train
        self.y_train = y_train
        self.transform = transform
        self.image_dir = Config['path_output']

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, item):
        file_path = osp.join(self.image_dir, self.X_train[item])
        return self.transform(Image.open(file_path)),self.y_train[item]



# test data class
class celeba32_test(Dataset):
    def __init__(self, X_test, y_test, transform):
        self.X_test = X_test
        self.y_test = y_test
        self.transform = transform
        self.image_dir = Config['path_output']


    def __len__(self):
        return len(self.X_test)


    def __getitem__(self, item):
        file_path = osp.join(self.image_dir, self.X_test[item])
        return self.transform(Image.open(file_path)), self.y_test[item]




def get_dataloader(debug, batch_size, num_workers):
    dataset = celeba32_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, classes = dataset.create_dataset()
    num = 0
    for temp in y_train:
        if temp == 1:
            num = num + 1
    print(num)

    num = 0
    for temp in y_test:
        if temp == 1:
            num = num + 1
    print(num)

    if debug==True:
        train_set = celeba32_train(X_train[:100], y_train[:100], transform=transforms['train'])
        test_set = celeba32_test(X_test[:100], y_test[:100], transform=transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}
    else:
        train_set = celeba32_train(X_train, y_train, transforms['train'])
        test_set = celeba32_test(X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}
        print(dataset_size)

    datasets = {'train': train_set, 'test': test_set}
    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x=='train' else False,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
                                 for x in ['train', 'test']}
    return dataloaders, classes, dataset_size


