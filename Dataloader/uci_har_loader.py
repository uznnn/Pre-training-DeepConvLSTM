# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:10:22 2022

@author: weizh
"""
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class data_set(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        

    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, index):
        sample_x = self.data_x[index]
        sample_y = self.data_y[index]
        return sample_x, sample_y
    
class uci_har_loader():
    def __init__ (self,batch_size=128): # default value batch_size = 128
        self.batch_size = batch_size    
        self.PATH =  os.getcwd()+"\Dataset\\UCI HAR Dataset"  #os.path.abspath('..')
        
        self.LABEL_NAMES = ["Walking", "Walking upstairs", "Walking downstairs", "Sitting", "Standing", "Laying"]

    def load_y_data(self,y_path):
        y = np.loadtxt(y_path, dtype=np.int32).reshape(-1,1)
        # change labels range from 1-6 t 0-5, this enables a sparse_categorical_crossentropy loss function
        return y - 1
    
    def load_X_data(self,X_path):
        X_signal_paths = [X_path + file for file in os.listdir(X_path)]
        X_signals = [np.loadtxt(path, dtype=np.float32) for path in X_signal_paths]
        return np.transpose(np.array(X_signals), (1, 2, 0))
    
    
    def dataloader(self):
        # load X data
        X_train = self.load_X_data(os.path.join(self.PATH + r'\train\Inertial Signals/'))
        _x_test_ = self.load_X_data(os.path.join(self.PATH + r'\test\Inertial Signals/'))
        X_valid = _x_test_[:1500]
        X_test = _x_test_[1500:]
        
        # load y label
        y_train = self.load_y_data(os.path.join(self.PATH + r'\train\y_train.txt'))
        _y_test_ = self.load_y_data(os.path.join(self.PATH + r'\test\y_test.txt'))
        y_valid = _y_test_[:1500]
        y_test = _y_test_[1500:]
        
        print("useful information:")
        print(f"shapes (n_samples, n_steps, n_signals) of X_train: {X_train.shape} and X_valid: {X_valid.shape} and X_test: {X_test.shape}")

        
        train_data = data_set(X_train,y_train[:,0])
        valid_data = data_set(X_valid,y_valid[:,0])
        test_data = data_set(X_test,y_test[:,0])
        train_data_loader = DataLoader(train_data, 
                                       batch_size   =  self.batch_size,
                                       shuffle      =  True,
                                       num_workers  =  0,
                                       drop_last    =  False)
        valid_data_loader = DataLoader(valid_data, 
                                       batch_size   =  self.batch_size,
                                       shuffle      =  False,
                                       num_workers  =  0,
                                       drop_last    =  False)
        test_data_loader = DataLoader(test_data, 
                                       batch_size   =  self.batch_size,
                                       shuffle      =  False,
                                       num_workers  =  0,
                                       drop_last    =  False)
        return {'train_data_loader':train_data_loader,
                'valid_data_loader':valid_data_loader,
                'test_data_loader':test_data_loader}

if __name__ =='__main__':
    uci_har_loader().dataloader()