# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 06:15:54 2022

@author: weizh
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
#from Model.model import DeepConvLSTM_pre
from Dataloader.uci_har_loader import uci_har_loader
from math import cos, pi
import os 
from torch import optim
import time

device = torch.device("cuda"if torch.cuda.is_available() else "cpu")

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
        fig.savefig('C:/Users\weizh\桌面\Model/Record/Record.png',dpi=100)

    plt.show()
    

def validation_class(data_loader, criterion,model):
    
    model.eval()
    total_loss = []
    preds = []
    trues = []
    with torch.no_grad():
        for i, (batch_x,batch_y) in enumerate(data_loader):

            batch_x = batch_x.double().to(device)
            batch_y = batch_y.long().to(device)

            outputs = model(batch_x)

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
        

def training(model,dataset='uci',train_epochs =100):
    if dataset=='uci':
        dataloader = uci_har_loader().dataloader()
    train_data_loader,valid_data_loader,test_data_loader = dataloader['train_data_loader'],dataloader['valid_data_loader'],dataloader['test_data_loader']
    train_steps = len(train_data_loader)
    # def a mask  (length,mask_length,ratio,features)
    valid_loss_record=[]
    preds = []
    trues = []
    f_score =[]
    # Loss function
    criterion_cross =  nn.CrossEntropyLoss(reduction="mean").to(device)
    
    
    #正式预训练
    lr_max=0.001
    lr_min=0.0001
    max_epoch=100
    
    model_optim_class =optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(train_epochs):
        adjust_learning_rate(optimizer=model_optim_class,current_epoch=epoch,max_epoch=max_epoch,lr_min=lr_min,lr_max=lr_max,warmup=True)
        train_loss_class = []
        model.train()
        epoch_time = time.time()
        
        for i, (batch_x,batch_y) in enumerate(train_data_loader):
            #start = time.time()
    
            model_optim_class.zero_grad()
    
            batch_x = batch_x.double().to(device)
            
            batch_y_class = batch_y.to(torch.long).to(device)
            outputs_class = model(batch_x).to(device)
            
            loss_class = criterion_cross(outputs_class, batch_y_class)
            
            loss_class.backward()
            train_loss_class.append(loss_class.item())
            model_optim_class.step()
            
            preds.extend(list(np.argmax(outputs_class.detach().cpu().numpy(),axis=1)))
            trues.extend(list(batch_y_class.detach().cpu().numpy()))
        train_acc = accuracy_score(preds,trues)
        valid_loss,acc,f_1,_,_ = validation_class(valid_data_loader, criterion_cross,model)
        train_loss_class = np.average(train_loss_class)
        print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
    
        valid_loss_record.append(round(valid_loss,2))
        if valid_loss_record[-1]==min(valid_loss_record):
            save_path = os.getcwd() +'\Log_model\\Reg_model'
            torch.save(model, save_path)
        if len(valid_loss_record)>20 and valid_loss_record[-10] == min(valid_loss_record):
            break
        f_score.append(f_1)
        print("RESULT: Epoch: {0}, Steps: {1} | \n Train Loss: {2:.4f} Train Accuracy:{3:.4f}  Valid Loss: {4:.4f} Accuracy: {5:.4f} F_1 score: {6:.4f}".format(
            epoch + 1, train_steps,train_loss_class,train_acc,valid_loss,acc,f_1))
    model = torch.load(save_path)
    test_loss,test_acc,test_f_1,_,_ = validation_class(test_data_loader, criterion_cross,model)
    print('\n***Final Test Result:  Test loss:{0:.4f}  Accuracy:{1:.4f}  F1_score:{2:.4f}***'.format(test_loss,test_acc,test_f_1))
    return test_f_1
