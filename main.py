# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 05:26:14 2022

@author: weizh
"""
import argparse
from Training.pre_training import pretraining
from Training.training import training
from Model.deepconvlstm import DeepConvLSTM
from Model.deepconvlstm_pre import DeepConvLSTM_pre
import torch
import os
import numpy as np

# limination of seed and speed up the training processes
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

#model parameters
BATCH_NORM = False
REDUCE_LAYER = False
POOLING = False
NO_LSTM = False
WEIGHTS_INIT = 'xavier_normal'
POOL_TYPE = 'max',
CONV_BLOCK_TYPE = 'normal'
SEED= 2
REDUCE_LAYER_OUTPUT = 8
POOL_KERNEL_WIDTH = 2
FILTER_WIDTH = 11
NB_UNITS_LSTM = 128
NB_LAYERS_LSTM = 1
NB_CONV_BLOCKS = 2
NB_FILTERS = 64
DILATION = 1
DROP_PROB = 0.5
NB_CLASSES = 6
WINDOW_SIZE = 128
NB_CHANNELS = 9

#train parameters
DATASET_NAME = 'uci'
PRETRAIN_EPOCHS = 0
TRAIN_EPOCHS = 100
FILTER_SCALING_FACTOR = 1

def main(args):
    
    setup_seed(args.seed)
    device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
    config ={'batch_norm': args.batch_norm, 
             'seed': args.seed,  #
             'filter_width': args.filter_width,  
             'nb_units_lstm': args.nb_units_lstm, 
             'nb_layers_lstm': args.nb_layers_lstm, 
             'nb_conv_blocks': args.nb_conv_blocks,  
             'nb_filters': args.nb_filters,  
             'dilation': args.dilation,  
             'drop_prob': args.drop_prob, 
             'nb_classes': args.nb_classes, 
             'window_size': args.window_size, 
             'nb_channels': args.nb_channels,
             'filter_scaling_factor': args.filter_scaling_factor}
    
    #pretraining
    input_shape = (1,1,args.window_size,args.nb_channels)
    nb_classes = args.nb_classes
    filter_scaling_factor = args.filter_scaling_factor
    '''
    model_pre = DeepConvLSTM_pre(input_shape,nb_classes,filter_scaling_factor,config).double().to(device)
    
    print("Parameter :", np.sum([para.numel() for para in model_pre.parameters()]))
    pretraining(model_pre,dataset=args.dataset_name,pretrain_epochs =args.pretraining_epochs)
    '''
    # 5 times formal training
    test_record=[]
    for i in range(5):
        setup_seed(i)
        
        model = DeepConvLSTM(input_shape,nb_classes,filter_scaling_factor,config)
        '''
        path = os.getcwd()+'\Log_model\\Reg_model_alt'
        model_dict = model.state_dict() 
        save_model = torch.load(path)
        pretrained_dict =  save_model.state_dict()
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict} 
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        '''
        model = model.double().to(device)
        
        print("Parameter :", np.sum([para.numel() for para in model.parameters()]))
        test_loss=training(model,dataset=args.dataset_name,train_epochs =args.training_epochs)
        test_record.append(round(test_loss,4))
        print('Test_time {}'.format(i))
    print('Test_results_record:{0}\nAverage:{1:.4f}\nStd:{2:.4f}'.format(test_record,np.average(test_record),np.std(test_record)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--batch_norm', default=BATCH_NORM, action='store_true',
                        help='Flag indicating to use batch normalisation after each convolution')
    parser.add_argument('--reduce_layer', default=REDUCE_LAYER, action='store_true',
                        help='Flag indicating to use reduce layer after convolutions')
   
    parser.add_argument('--pooling', default=POOLING, action='store_true',
                        help='Flag indicating to apply pooling after convolutions')
    
    parser.add_argument('--no_lstm', default=NO_LSTM, action='store_true',
                        help='Flag indicating whether to omit LSTM from architecture')
    
    parser.add_argument('-wi', '--weights_init', default=WEIGHTS_INIT, type=str,
                        help='weight initialization method used. Options: normal, orthogonal, xavier_uniform, '
                             'xavier_normal, kaiming_uniform, kaiming_normal. '
                             'Default: xavier_normal')
    parser.add_argument('-pt', '--pool_type', default=POOL_TYPE, type=str,
                        help='type of pooling applied. Options: max, average. '
                             'Default: max')
 
    parser.add_argument('-cbt', '--conv_block_type', default=CONV_BLOCK_TYPE, type=str,
                        help='type of convolution blocks used. Options: normal, skip, fixup.'
                             'Default: normal')
    parser.add_argument('-dm', '--dataset_name', default=DATASET_NAME, type=str,
                        help='The dataset we choose to training'
                             'Default: uci')

  
    parser.add_argument('-s', '--seed', default=SEED, type=int,
                        help='Seed to be employed. '
                             'Default: 5')

    parser.add_argument('-rlo', '--reduce_layer_output', default=REDUCE_LAYER_OUTPUT, type=int,
                        help='Size of reduce layer output. '
                        'Default: 8')
    parser.add_argument('-nol', '--nb_layers_lstm', default=NB_LAYERS_LSTM, type=int,
                        help='amount LSTM layer. '
                        'Default: 1')
    parser.add_argument('-pre_epoch', '--pretraining_epochs', default=PRETRAIN_EPOCHS, type=int,
                        help='The max number of pretraining epoch '
                        'Default: 100')
    parser.add_argument('-train_epoch', '--training_epochs', default=TRAIN_EPOCHS, type=int,
                        help='The max number of training epoch '
                        'Default: 100')
    
    parser.add_argument('-pkw', '--pool_kernel_width', default=POOL_KERNEL_WIDTH, type=int,
                        help='Size of pooling kernel.'
                             'Default: 2')
    parser.add_argument('-fw', '--filter_width', default=FILTER_WIDTH, type=int,
                        help='Filter size (convolutions).'
                             'Default: 11')

    parser.add_argument('-nbul', '--nb_units_lstm', default=NB_UNITS_LSTM, type=int,
                        help='Number of units within each LSTM layer. '
                             'Default: 128')

    parser.add_argument('-nbcb', '--nb_conv_blocks', default=NB_CONV_BLOCKS, type=int,
                        help='Number of convolution blocks. '
                             'Default: 2')
    parser.add_argument('-nbf', '--nb_filters', default=NB_FILTERS, type=int,
                        help='Number of convolution filters.'
                             'Default: 64')
    parser.add_argument('-dl', '--dilation', default=DILATION, type=int,
                        help='Dilation applied in convolution filters.'
                             'Default: 1')
    ###
    parser.add_argument('-nc', '--nb_classes', default=NB_CLASSES, type=int,
                        help='Classes num Classification'
                             'Default: 6')
    parser.add_argument('-nb_channel', '--nb_channels', default=NB_CHANNELS, type=int,
                        help='Channel num Classification.'
                             'Default: 9')
    parser.add_argument('-ws', '--window_size', default=WINDOW_SIZE, type=int,
                        help='The length of sliding window.'
                             'Default: 128')
    ###
    parser.add_argument('-dp', '--drop_prob', default=DROP_PROB, type=float,
                        help='Dropout probability.'
                             'Default 0.5')
    parser.add_argument('-fsf', '--filter_scaling_factor', default=FILTER_SCALING_FACTOR, type=int,
                        help='The parameters can be used to regular filter account. '
                             'Default: 1')
    
    args = parser.parse_args()
    main(args)
    