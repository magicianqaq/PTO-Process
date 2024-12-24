from torch.utils.data import Dataset
from torch.utils.data import Subset, DataLoader
import pickle
import torch
import numpy as np
from untils import *

# define dataset and dataloader
class Dataset_adds(Dataset):
    def __init__(self, features, labels, adds, device):
        self.features = features.to(device)
        self.labels = labels.to(device)

        if adds != 0:
            data_dir = adds
            with open(data_dir + "//adds", 'rb') as file:
                self.adds = pickle.load(file)
        else:
            self.adds = np.zeros(labels.shape[0])
        self.adds = torch.from_numpy(self.adds).float().to(device)

        print("features shape: ", self.features.shape)
        print("labels shape: ", self.labels.shape)
        print("add shape: ", self.adds.shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.adds[idx]
        
class DataSplit():
    def __init__(self, set_params, params, opt_params, device):
        self.pre_type = set_params['pre_file']
        self.opt_type = set_params['opt_file']

        self.train_rate = params['train_rate_all']

        self.adds = opt_params['adds']

        self.device = device
        
    
    def __call__(self):
        device = self.device
        data_dir = r".\data\\" + self.opt_type
        with open(data_dir + "/labels", 'rb') as file:
            labels = pickle.load(file)
        with open(data_dir + "/features", 'rb') as file:
            features = pickle.load(file)
        
        if self.pre_type == 'MLP':
            features = features.reshape(features.shape[0], -1)
            features = torch.from_numpy(features).float().to(device)
            
            labels = labels.reshape(labels.shape[0], -1)
            labels = torch.from_numpy(labels).float().to(device)
            
        elif self.pre_type == 'CombRenset18' or 'LSTM':
            features= torch.from_numpy(features).float().to(device)
            labels = torch.from_numpy(labels).float().to(device)
            
        else:
            print("No predictive model is available")
        
        # data_spilt
        dataset_all = Dataset_adds(features, labels, self.adds, device)
        train_dataset, test_dataset = dataset_split(dataset_all, self.train_rate)
        return train_dataset, test_dataset


