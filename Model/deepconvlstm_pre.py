# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 21:16:12 2022

@author: weizh
"""
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

class RegBlock_ncnn(nn.Module):
    """
    4层卷积加2个线性层，是数据还原size
    """
    def __init__(self,nb_units_lstm, nb_channels, filter_width, dilation=1):  # input_width:lstm输出对应的特征数（最后一维度）
                                                                         #features：输出的特征数，这里是9，dilation：只是对应原conv，可以不用设置
        super(RegBlock_ncnn, self).__init__()                            #filter_width：卷积核宽度
        self.filter_width = filter_width
        self.dilation = dilation
        self.linear = nn.Linear(nb_units_lstm,nb_channels,bias=False)
        self.conv1 = nn.ConvTranspose2d(1, 1, (self.filter_width, 1), dilation=(self.dilation, 1))
        self.conv2 = nn.ConvTranspose2d(1, 1, (self.filter_width, 1), dilation=(self.dilation, 1))
        self.conv3 = nn.ConvTranspose2d(1, 1, (self.filter_width, 1), dilation=(self.dilation, 1))
        self.conv4 = nn.ConvTranspose2d(1, 1, (self.filter_width, 1), dilation=(self.dilation, 1))
        #self.tanh = nn.Tanh() #这里不加激活函数结果更好,所以删掉了
        
    def forward(self, x):

        out=torch.unsqueeze(x, 1)

        out = self.conv1(out)

        #out = self.tanh(out)
        out = self.conv2(out)

        #out = self.tanh(out)
        out = self.conv3(out)

        #out = self.tanh(out)
        out = self.conv4(out)

        #out = self.tanh(out)
        out = torch.squeeze(out, 1)
        out = self.linear(out)
        return out
    
class ConvBlock(nn.Module):
    """
    Normal convolution block
    """
    def __init__(self, filter_width, input_filters, nb_filters, dilation, batch_norm):
        super(ConvBlock, self).__init__()
        self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_filters = nb_filters
        self.dilation = dilation
        self.batch_norm = batch_norm

        self.conv1 = nn.Conv2d(self.input_filters, self.nb_filters, (self.filter_width, 1), dilation=(self.dilation, 1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1), dilation=(self.dilation, 1))
        if self.batch_norm:
            self.norm1 = nn.BatchNorm2d(self.nb_filters)
            self.norm2 = nn.BatchNorm2d(self.nb_filters)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm1(out)

        out = self.conv2(out)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm2(out)

        return out
class DeepConvLSTM_pre(nn.Module):
    def __init__(self, 
                 input_shape, 
                 nb_classes,
                 filter_scaling_factor,
                 config):
        """
        DeepConvLSTM model based on architecture suggested by Ordonez and Roggen (https://www.mdpi.com/1424-8220/16/1/115)
        
        """
        super(DeepConvLSTM_pre, self).__init__()
        self.nb_conv_blocks = config["nb_conv_blocks"]
        self.nb_filters     = int(filter_scaling_factor*config["nb_filters"])
        self.dilation       = config["dilation"]
        self.batch_norm     = bool(config["batch_norm"])
        self.filter_width   = config["filter_width"]
        self.nb_layers_lstm = config["nb_layers_lstm"]
        self.drop_prob      = config["drop_prob"]
        self.nb_units_lstm  = int(filter_scaling_factor*config["nb_units_lstm"])
        
        print(self.filter_width)
        self.nb_channels    = input_shape[3]
        self.nb_classes     = nb_classes

    
        self.conv_blocks = []

        for i in range(self.nb_conv_blocks):
            if i == 0:
                input_filters = input_shape[1]
            else:
                input_filters = self.nb_filters
    
            self.conv_blocks.append(ConvBlock(self.filter_width, input_filters, self.nb_filters, self.dilation, self.batch_norm))

        
        self.conv_blocks = nn.ModuleList(self.conv_blocks)
        
        # define dropout layer
        self.dropout = nn.Dropout(self.drop_prob)
        # define lstm layers
        self.lstm_layers = []
        for i in range(self.nb_layers_lstm):
            if i == 0:
                self.lstm_layers.append(nn.LSTM(self.nb_channels * self.nb_filters, self.nb_units_lstm, batch_first =True))
            else:
                self.lstm_layers.append(nn.LSTM(self.nb_units_lstm, self.nb_units_lstm, batch_first =True))
        self.lstm_layers = nn.ModuleList(self.lstm_layers)
        
        # define classifier
        self.fc = nn.Linear(self.nb_units_lstm, self.nb_classes, bias=True)
        self.reg = RegBlock_ncnn(self.nb_units_lstm,self.nb_channels,self.filter_width)  #用ncnn做Regression任务的网络部分
    def forward(self, x):
        # reshape data for convolutions
        # B,L,C = x.shape
        # x = x.view(B, 1, L, C)
        x = torch.unsqueeze(x, 1)
        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)
        final_seq_len = x.shape[2]

        # permute dimensions and reshape for LSTM
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, final_seq_len, self.nb_filters * self.nb_channels)
        
        x = self.dropout(x)

        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)

        out_reg = self.reg(x)
        x = x[:, -1, :]
        
        out_class = self.fc(x)
        

        return out_class, out_reg