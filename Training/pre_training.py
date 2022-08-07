# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:59:11 2022

@author: weizh
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from Dataloader.uci_har_loader import uci_har_loader
from math import cos, pi
import os 
from torch import optim
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class create_mask():
    
    def __init__(self,L,feat,lm,masking_ratio):  #直接调用里面的complt_mask就可以
        self.L = L #L: length of mask and sequence to be masked
        self.lm = lm #average length of masking subsequences (streaks of 0s) 
        self.masking_ratio = masking_ratio #proportion of L to be masked
        self.feat = feat # #生成mask特征数
    def geom_noise_mask_single(self):
        """
        Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
        proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
        
        """
        keep_mask = np.ones(self.L, dtype=bool)
        p_m = 1 / self.lm  # probability of each masking sequence stopping. parameter of geometric distribution.
        p_u = p_m * self.masking_ratio / (1 - self.masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
        p = [p_m, p_u]

        # Start in state 0 with masking_ratio probability
        state = int(np.random.rand() > self.masking_ratio)  # state 0 means masking, 1 means not masking
        for i in range(self.L):
            keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
            if np.random.rand() < p[state]:
                state = 1 - state
        return keep_mask
    def complt_mask(self):
        fmask =self.geom_noise_mask_single().reshape(-1, 1)
        for i in range(self.feat-1):
            _mask_ = self.geom_noise_mask_single()
            _mask_ = _mask_.reshape(-1, 1)
            fmask= np.hstack((fmask,_mask_))
        return fmask.astype(np.float32)  #  boolean numpy array intended to mask ('drop') with 0s a sequence of length L


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')     # 这里会存储迄今最优模型的参数
        save_path = os.getcwd() +'\Log_model\\Reg_model_alt'
        torch.save(model, save_path)                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss
        
        
def visual(org_data,reg_data,loss):   # org_data,reg_data: Tensor(batch_size,sequence,features);  
                                      # loss: 损失函数的item值，用来在图中标注
    import matplotlib.pyplot as plt
    size = org_data.shape
    org = org_data.cpu().numpy()[1,::]
    reg = reg_data.cpu().numpy()[1,::]
    fig = plt.figure(figsize=(50,20))
    
    for i in range(size[2]):
        try: pic = fig.add_subplot(3,3,i+1)  #这里可以修改子图尺寸,最多显示设置子图数量上限的特征数
        except: break
            
        index = [i for i, e in enumerate(reg[:,i]) if e != 0]  #下四行是数据填充
        value = [e for i, e in enumerate(reg[:,i]) if e != 0]
        pic.scatter(index,value, color = 'hotpink')
        pic.plot(range(size[1]),org[:,i], color = 'saddlebrown',marker="o", markersize=4)
        
        pic.axhline(y=0, c="gray", ls="--", lw=1)  # 加一些标注，辅助看图
        fig.legend(["Orginal","Zero_linear","Regression"],ncol=3)
        pic.set_xlabel("Time Steps")
        pic.set_ylabel("Value")
        _name_ = str(round(loss,3)) #loss只保留三位有效数字

        plt.title("Feature_{}_Loss_{}".format(i+1,_name_))  
    dir_path = os.getcwd()
    _path_ = dir_path + '/Log_model/Record.png'
    fig.savefig(_path_,dpi=100)

    plt.show()
    
    
def validation_reg(test_data_loader, criterion,model,mask_par):  #validation过程
    
    model.eval()
    test_loss = []

    with torch.no_grad():
            for i, (batch_x_reg,batch_y) in enumerate(test_data_loader):
                mask = create_mask(mask_par[0],mask_par[1],mask_par[2],mask_par[3]).complt_mask()
                mask = torch.from_numpy(mask)
                batch_x = (batch_x_reg*mask).double().to(device)
                batch_y = (batch_x_reg*(1-mask)).double().to(device)

                outputs = model(batch_x)[1]*(1-mask).to(device)
                #print(time.time()-start)
                loss = criterion(outputs, batch_y)*6.66
                test_loss.append(loss.item())   
                
                if i ==3:  #选其中的第三组数据进行展示，可以随机修改
                    org_data,reg_data = batch_x_reg,outputs
            output_loss = sum(test_loss)/len(test_loss)
            #可视化
            visual(org_data,reg_data,output_loss)
                
    return output_loss
def validation_class(data_loader, criterion,model,mask_par):
    
    model.eval()
    total_loss = []
    preds = []
    trues = []
    
    with torch.no_grad():
        for i, (batch_x,batch_y) in enumerate(data_loader):
            mask = create_mask(mask_par[0],mask_par[1],mask_par[2],mask_par[3]).complt_mask()
            mask = torch.from_numpy(mask)
            batch_x = batch_x.double().to(device)
            batch_y = batch_y.long().to(device)

            outputs = model(batch_x)[0]

            pred = outputs.detach()#.cpu()
            true = batch_y.detach()#.cpu()

            loss = criterion(pred, true) 
            total_loss.append(loss.cpu())

            preds.extend(list(np.argmax(outputs.detach().cpu().numpy(),axis=1)))
            trues.extend(list(batch_y.detach().cpu().numpy()))   

    total_loss = np.average(total_loss)
    acc = accuracy_score(preds,trues)

    f_w = f1_score(trues, preds, average='weighted')
    f_macro = f1_score(trues, preds, average='macro')
    f_micro = f1_score(trues, preds, average='micro')
    model.train()

    return total_loss,  acc, f_w,  f_macro, f_micro#, f_1


def adjust_learning_rate(optimizer, current_epoch,max_epoch,lr_min=0,lr_max=0.1,warmup=True):
    warmup_epoch = 100 if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

def pretraining(model,dataset='uci',pretrain_epochs =100):
    
    ###dataloader,这是需要根据dataloader添加的，这里只需要训练和valid集
    if dataset=='uci':
        dataloader = uci_har_loader().dataloader()
    train_data_loader,valid_data_loader = dataloader['train_data_loader'],dataloader['valid_data_loader']
    #------------------------------------------------------------------------------------------------------
    
    train_steps = len(train_data_loader)
    # def a mask  (length,mask_length,ratio,features)
    test_loss_record=[]
    loss_base_list =[]
    # Loss function
    criterion_mse =  nn.MSELoss(reduction="mean").to(device)
    criterion_cross =  nn.CrossEntropyLoss(reduction="mean").to(device)
    # Use early stop
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # mask parameter automatic calculated 
    mask_len = 3
    mask_ratio = 0.15
    for i, (batch_x_reg,batch_y) in enumerate(train_data_loader):
        L,F = batch_x_reg.shape[1],batch_x_reg.shape[2]
        mask_par = [L,F,mask_len,mask_ratio]
        break
    
    #显示Baseline
    for i, (batch_x_reg,batch_y) in enumerate(train_data_loader):
        mask = create_mask(L,F,mask_len,mask_ratio).complt_mask()
        mask = torch.from_numpy(mask)
        loss_base = criterion_mse(torch.zeros(batch_x_reg.shape), batch_x_reg*mask)*6.66
        loss_base_list.append(loss_base.item())
    loss_base_line = np.mean(loss_base_list)
    
    #正式预训练
    lr_max=0.001
    lr_min=0.0001
    max_epoch=pretrain_epochs
    model_optim_reg =optim.Adam([
                                {'params': model.conv_blocks.parameters()},
                                {'params': model.lstm_layers.parameters()},
                                {'params': model.dropout.parameters()},
                                {'params': model.fc.parameters(), 'lr': 0},
                                {'params': model.reg.parameters()}
                                ], lr=0.001)
    model_optim_class =optim.Adam([
                                  {'params': model.conv_blocks.parameters()},
                                  {'params': model.lstm_layers.parameters()},
                                  {'params': model.dropout.parameters()},
                                  {'params': model.fc.parameters()},
                                  {'params': model.reg.parameters(), 'lr': 0}
                                  ], lr=0.001)
    
    for epoch in range(pretrain_epochs):
        train_loss_reg = []
        train_loss_class = []
        model.train()
        epoch_time = time.time()
        
        
        if epoch%2:
            for i, (batch_x_reg,batch_y) in enumerate(train_data_loader):
                #start = time.time()
                adjust_learning_rate(optimizer=model_optim_reg,current_epoch=epoch,max_epoch=max_epoch,lr_min=lr_min,lr_max=lr_max,warmup=True)
                mask = create_mask(L,F,mask_len,mask_ratio).complt_mask()
                mask = torch.from_numpy(mask)
                model_optim_reg.zero_grad()
    
                batch_x = (batch_x_reg*mask).double().to(device)
    
                batch_y = (batch_x_reg*(1-mask)).double().to(device) 
                outputs_reg= model(batch_x)[1]*(1-mask).to(device) 
                loss_reg = criterion_mse(outputs_reg, batch_y)*6.66
                loss_reg.backward()
                train_loss_reg.append(loss_reg.item())
                model_optim_reg.step()
    
        else:
            for i, (batch_x_reg,batch_y) in enumerate(train_data_loader):
                #start = time.time()
                adjust_learning_rate(optimizer=model_optim_class,current_epoch=epoch,max_epoch=max_epoch,lr_min=lr_min,lr_max=lr_max,warmup=True)
                mask = create_mask(L,F,mask_len,mask_ratio).complt_mask()
                mask = torch.from_numpy(mask)
                model_optim_class.zero_grad()
    
                batch_x = (batch_x_reg*mask).double().to(device)
                batch_y = batch_y.to(torch.long).to(device)
                outputs_class = model(batch_x)[0].to(device)    
                loss_class = criterion_cross(outputs_class, batch_y)
                loss_class.backward()
                train_loss_class.append(loss_class.item())
                model_optim_class.step()
                
                #print(time.time()-start)
                #print("-------------")
        valid_loss_reg = validation_reg(valid_data_loader, criterion_mse,model,mask_par)
        valid_loss_class,acc,f_1,_,_ = validation_class(valid_data_loader, criterion_cross,model,mask_par)
        train_loss_class = np.average(train_loss_class)
        train_loss_reg = np.average(train_loss_reg)
        print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
    
        test_loss_record.append(round(valid_loss_class,2))
            
        early_stopping(valid_loss_class, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        '''
        if len(test_loss_record)>10 and test_loss_record[-10] == min(test_loss_record):
            break
        '''
        print("TEST: Epoch: {0}, Steps: {1} | Train Loss Reg: {2:.4f}  Test Loss Reg: {3:.4f}  Baseline Loss Reg: {4:.4f} \n                            Train Loss Class: {5:.4f} Test Loss Class: {6:.4f} Accuracy Class: {7:.4f} F_1 score Class: {8:.4f}".format(
            epoch + 1, train_steps, train_loss_reg, valid_loss_reg,loss_base_line,train_loss_class,valid_loss_class,acc,f_1))
