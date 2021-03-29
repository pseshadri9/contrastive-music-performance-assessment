from __future__ import print_function
import gc
import os
import sys
import math
import time
import scipy.stats as ss
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.PCConvNet import PCConvNet, PCConvNetCls, PCConvNetContrastive
from models.PCConvLstmNet import PCConvLstmNet, PCConvLstmNetCls
from dataLoaders.PitchContourDataset import PitchContourDataset
from dataLoaders.PitchContourDataloader import PitchContourDataloader
from dataLoaders.MASTDataset import MASTDataset
from dataLoaders.MASTDataloader import MASTDataloader
from tensorboard_logger import configure, log_value
from sklearn import metrics
import eval_utils
import train_utils
import dill
from contrastive_utils import ContrastiveLoss

# set manual random seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)
# check is cuda is available and print result
CUDA_AVAILABLE = torch.cuda.is_available()
print('Running on GPU: ', CUDA_AVAILABLE)
if CUDA_AVAILABLE != True:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

# initialize training parameters
RUN = 110
NUM_EPOCHS = 500 #250 #2000
NUM_BATCHES = 10
SEGMENT = '2'
MTYPE = 'conv'
CTYPE = 0
# initialize dataset, dataloader and created batched data

#SET CONSTANTS
metric_type = {0:'musicality', 1:'note accuracy',2:'Rhythm Accuracy',3:'tonality'}
instrument = 'clarinet'
cross_instrument = 'ALL'
experiment = '5-classification_CE_Only-LR=0.001'
METRIC = 0 # 0: Musicality, 1: Note Accuracy, 2: Rhythmic Accuracy, 3: Tone Quality
BAND = 'middle'
ADD_NOISE_TEST = False
ADD_NOISE_VALID = False
NOISE_SHAPE = 'triangular'  #triangular, normal, or uniform
INPUT_REP = 'Cepstrum'
NAME = '{0}_{1}_{2}_{3}_{4}'.format(BAND, instrument, metric_type[METRIC], INPUT_REP, experiment)

#SET TRAINING CONSTANTS
contrastive = True
MSE_LOSS_STR = 1
CONTR_LOSS_STR = 0
num_labels = 5
classification = True
#HYPERPARAMETERS
LR_RATE = 0.001 #0.01
W_DECAY = 1e-5 #5e-4 #1e-5
MOMENTUM = 0.9

datasets = {'flute':{'test':'/media/SSD/FBA/crossInstr/middle_Flute__test.dill', 'train':'/media/SSD/FBA/crossInstr/middle_Flute__train_fixed.dill', 'valid':'/media/SSD/FBA/crossInstr/middle_Flute__valid.dill'},
            'clarinet':{'test':'/media/SSD/FBA/crossInstr/middle_Bb Clarinet__test.dill', 'train':'/media/SSD/FBA/crossInstr/middle_Bb Clarinet__train.dill', 'valid':'/media/SSD/FBA/crossInstr/middle_Bb Clarinet__valid.dill'},
            'saxophone':{'test':'/media/SSD/FBA/crossInstr/middle_Alto Saxophone__test.dill', 'train':'/media/SSD/FBA/crossInstr/middle_Alto Saxophone__train.dill', 'valid':'/media/SSD/FBA/crossInstr/middle_Alto Saxophone__valid.dill'}}
datasets_all = {'flute': '/media/SSD/FBA/crossInstr/middle_Flute_.dill', 'saxophone': '/media/SSD/FBA/crossInstr/middle_Alto Saxophone_.dill', 'clarinet':'/media/SSD/FBA/crossInstr/middle_Bb Clarinet_.dill', 
                'ALL':'/media/SSD/FBA/saved_dill/middle_2_new_dataPC.dill'}

file_name = BAND + '_' + str(SEGMENT) + '_data'

with open(datasets_all[instrument], 'rb') as f:
    NUM_DATA_POINTS = len(dill.load(f))
if sys.version_info[0] < 3:
    data_path = 'dat/' + file_name + '.dill'
    mast_path = '/Users/Som/GitHub/Mastmelody_dataset/f0data'
else:
    if torch.cuda.is_available():
        data_path = '/home/data_share/FBA/fall19/data/pitch_contour/' + BAND + '_2_pc_3.dill'
    else:
        data_path = '/Volumes/Farren/python_stuff/dat/' + BAND + '_2_data_3.dill'

    mast_path = '/home/apati/MASTmelody_dataset/f0data'

if BAND == 'mast':
    dataset = MASTDataset(mast_path)
    dataloader = MASTDataloader(dataset)
    CTYPE = 1
else:
    dataset = PitchContourDataset(datasets_all[instrument])
    dataloader = PitchContourDataloader(dataset, NUM_DATA_POINTS, NUM_BATCHES)


tr1, v1, vef, te1, tef = dataloader.create_split_data(1000, 500) #1000, 500 | 1500, 500 | 2000, 1000
tr2, v2, _, te2, _ = dataloader.create_split_data(1500, 500)
tr3, v3, _, te3, _ = dataloader.create_split_data(2000, 1000)
#tr4, v4, _, te4, _ = dataloader.create_split_data(2500, 1000)
#tr5, v5, _, te5, _ = dataloader.create_split_data(3000, 1500)
#tr6, v6, vef, te6, tef = dataloader.create_split_data(4000, 2000)
training_data = tr1 + tr2 + tr3 #+ tr2 + tr3 #+ tr4 + tr5 + tr6     # this is the proper training data split
validation_data = vef #+ v2 + v3 + v4 + v5 + v6
testing_data = te1 + te2 + te3 #+ te4 + te5 + te6


## augment data
aug_training_data = train_utils.augment_data(training_data)
aug_training_data = train_utils.augment_data(aug_training_data)
aug_validation_data = validation_data  #train_utils.augment_data(validation_data)


## initialize model
if MTYPE == 'conv':
    if BAND == 'mast':
        perf_model = PCConvNetCls(1)
    else:
        perf_model = PCConvNetContrastive(0, num_classes=num_labels)
elif MTYPE == 'lstm':
    if BAND == 'mast':
        perf_model = PCConvLstmNetCls()
    else:
        perf_model = PCConvLstmNet()        
if torch.cuda.is_available():
    perf_model.cuda()
if BAND == 'mast':
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.MSELoss()
if contrastive:
    criterion_contrastive = ContrastiveLoss(num_labels=num_labels)
    criterion = nn.CrossEntropyLoss()
else:
    criterion_contrastive = None
perf_optimizer = optim.SGD(perf_model.parameters(), lr= LR_RATE, momentum=MOMENTUM, weight_decay=W_DECAY)
#perf_optimizer = optim.Adam(perf_model.parameters(), lr = LR_RATE, weight_decay = W_DECAY)
print(perf_model)

# declare save file name
file_info = str(NUM_DATA_POINTS) + '_' + str(NUM_EPOCHS) + '_' + BAND + '_' + str(METRIC) + '_' + str(RUN) + '_' + MTYPE + '_onlyATest'       

# configure tensor-board logger
configure('pc_contrastive_runs/' + NAME + '_Reg' , flush_secs = 2)

## define training parameters
PRINT_EVERY = 1
ADJUST_EVERY = 1000
START = time.time()
#best_val_loss = 1.0
best_loss_contrastive_val = float('inf')
best_valrsq = .20
best_epoch = 0
best_acc_contrastive_val = float('inf')
# train and validate
try:
    print("Training for %d epochs..." % NUM_EPOCHS)
    for epoch in range(1, NUM_EPOCHS + 1):
        # perform training and validation
        train_loss, train_r_sq, train_accu, train_accu2, val_loss, val_r_sq, val_accu, val_accu2 = train_utils.train_and_validate(perf_model, criterion, perf_optimizer, aug_training_data, aug_validation_data, METRIC, MTYPE, CTYPE, contrastive=criterion_contrastive, strength=(MSE_LOSS_STR, CONTR_LOSS_STR))
        if contrastive:
            loss_contrastive_val, acc_contrastive_val, ce_loss_val = eval_utils.eval_acc_contrastive(perf_model, criterion_contrastive, vef, METRIC, MTYPE, CTYPE, criterion_CE=criterion)
            loss_contrastive_train, acc_contrastive_train, ce_loss_train = eval_utils.eval_acc_contrastive(perf_model, criterion_contrastive, aug_training_data, METRIC, MTYPE, CTYPE, criterion_CE=criterion)
        # adjut learning rate
        # train_utils.adjust_learning_rate(perf_optimizer, epoch, ADJUST_EVERY)
        # log data for visualization later

        ####
        log_value('train_loss', train_loss, epoch)
        log_value('val_loss', val_loss, epoch)
        log_value('train_r_sq', train_r_sq, epoch)
        log_value('val_r_sq', val_r_sq, epoch)
        log_value('train_accu', train_accu, epoch)
        log_value('val_accu', val_accu, epoch)
        log_value('train_accu2', train_accu2, epoch)
        log_value('val_accu2', val_accu2, epoch)
        if contrastive:
            log_value('Validation contrastive loss', loss_contrastive_val, epoch)
            log_value('Validation CrossEntropy Loss', ce_loss_val, epoch)
            log_value('Validation acc', acc_contrastive_val, epoch)
            log_value('Training contrastive loss', loss_contrastive_train, epoch)
            log_value('Training CrossEntropy Loss', ce_loss_train, epoch)
            log_value('Training acc', acc_contrastive_train, epoch)
        #####

        # print loss
        if epoch % PRINT_EVERY == 0:
            print('[%s (%d %.1f%%)]' % (train_utils.time_since(START), epoch, float(epoch) / NUM_EPOCHS * 100))
            #print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Train Loss: ', train_loss, ' R-sq: ', train_r_sq, ' Accu:', train_accu, train_accu2))
            #print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Valid Loss: ', val_loss, ' R-sq: ', val_r_sq, ' Accu:', val_accu, val_accu2))

            if contrastive:
                print('[%s %0.5f, %s %0.5f, %s %0.5f]'%('Train Contrastive Loss: ', loss_contrastive_train,'Train CE Loss:', ce_loss_train, 'Train Accuracy: ', acc_contrastive_train))
                print('[%s %0.5f, %s %0.5f, %s %0.5f]'%('Validation Contrastive Loss: ', loss_contrastive_val,'Validation CE Loss:', ce_loss_val, 'Validation Accuracy: ', acc_contrastive_val))
        # save model if best validation accuracy
        if acc_contrastive_val > best_acc_contrastive_val #loss_contrastive_val.item() < best_loss_contrastive_val:
            n = 'pc_contrastive_runs/' + NAME + '_best'
            train_utils.save(n, perf_model)
            best_acc_contrastive_val = acc_contrastive_val
            #best_loss_contrastive_val = loss_contrastive_val.item()
            best_epoch = epoch
        # store the best r-squared value from training
        if val_r_sq > best_valrsq:
            best_valrsq = val_r_sq
        if best_epoch < epoch - 100:
            break
    print("Saving...")
    train_utils.save('pc_contrastive_runs/'+NAME, perf_model)
except KeyboardInterrupt:
    print("Saving before quit...")
    train_utils.save('pc_contrastive_runs/'+NAME, perf_model)

print('BEST R^2 VALUE: ' + str(best_valrsq))

# test
# test of full length data
if contrastive:
    loss_contrastive_test, acc_contrastive_test, ce_loss_test = eval_utils.eval_acc_contrastive(perf_model, criterion_contrastive, testing_data, METRIC, MTYPE, CTYPE, criterion_CE=criterion)
    print('[%s %0.5f, %s %0.5f, %s %0.5f]'%('Testing Contrastive Loss: ', loss_contrastive_test,'Testing CE Loss:', ce_loss_test, 'Testing Accuracy: ', acc_contrastive_test))
else:
    test_loss, test_r_sq, test_accu, test_accu2 = eval_utils.eval_model(perf_model, criterion, testing_data, METRIC, MTYPE, CTYPE)
    print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Testing Loss: ', test_loss, ' R-sq: ', test_r_sq, ' Accu:', test_accu, test_accu2))

# validate and test on best validation model
# read the model
#filename = file_info + '_Reg'
filename = NAME + '_best'
if torch.cuda.is_available():
    perf_model.cuda()
    #perf_model.load_state_dict(torch.load('/Users/michaelfarren/Desktop/MusicPerfAssessment-master/src/runs/' + filename + '.pt'))
    perf_model.load_state_dict(torch.load('pc_contrastive_runs/' + filename + '.pt'))
else:
    perf_model.load_state_dict(torch.load('pc_contrastive_runs/' + filename + '.pt', map_location=lambda storage, loc: storage))

if contrastive:
    loss_contrastive_val, acc_contrastive_val, ce_loss_val = eval_utils.eval_acc_contrastive(perf_model, criterion_contrastive, vef, METRIC, MTYPE, CTYPE, criterion_CE=criterion)
    print('[%s %0.5f, %s %0.5f, %s %0.5f]'%('Validation Contrastive Loss: ', loss_contrastive_val,'Validation CE Loss:', ce_loss_val, 'Validation Accuracy: ', acc_contrastive_val))
else:
    val_loss, val_r_sq, val_accu, val_accu2 = eval_utils.eval_model(perf_model, criterion, vef, METRIC, MTYPE, CTYPE)
    print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Valid Loss: ', val_loss, ' R-sq: ', val_r_sq, ' Accu:', val_accu, val_accu2))

if contrastive:
    loss_contrastive_test, acc_contrastive_test, ce_loss_test = eval_utils.eval_acc_contrastive(perf_model, criterion_contrastive, testing_data, METRIC, MTYPE, CTYPE, criterion_CE=criterion)
    print('[%s %0.5f, %s %0.5f, %s %0.5f]'%('Testing Contrastive Loss: ', loss_contrastive_test,'Testing CE Loss:', ce_loss_test, 'Testing Accuracy: ', acc_contrastive_test))
else:
    test_loss, test_r_sq, test_accu, test_accu2 = eval_utils.eval_model(perf_model, criterion, testing_data, METRIC, MTYPE, CTYPE)
    print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Testing Loss: ', test_loss, ' R-sq: ', test_r_sq, ' Accu:', test_accu, test_accu2))
