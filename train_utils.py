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
import eval_utils
from contrastive_utils import ContrastiveLoss

def augment_data(data):
    """
    Augments the data using pitch shifting
    Args:
        data: batched data
    """
    num_batches = len(data)
    aug_data = [None] * num_batches
    # create augmented data
    for batch_idx in range(num_batches):
        mini_batch_size, seq_len = data[batch_idx]['pitch_tensor'].size()
        pitch_shift = ((torch.rand(mini_batch_size, 1) * 4) - 2) / 127.0
        pitch_shift = pitch_shift.expand(mini_batch_size, seq_len)
        pitch_tensor = data[batch_idx]['pitch_tensor'].clone()
        pitch_tensor[pitch_tensor != 0] = pitch_tensor[pitch_tensor != 0] + pitch_shift[pitch_tensor != 0]
        new_data = {}
        new_data['pitch_tensor'] = pitch_tensor
        new_data['score_tensor'] = data[batch_idx]['score_tensor'].clone()
        aug_data[batch_idx] = new_data
    # combine with orignal data
    aug_data = data + aug_data
    return aug_data

def train(model, criterion, optimizer, data, metric, mtype, ctype, contrastive=None, strength = None, encoder = True, classification = True):
    """
    Returns the model performance metrics
    Args:
        model:          object, trained model of PitchContourAssessor class
        criterion:      object, of torch.nn.Functional class which defines the loss 
        optimizer:      object, of torch.optim class which defines the optimization algorithm
        data:           list, batched testing data
        metric:         int, from 0 to 3, which metric to evaluate against
        mtype:          string, 'conv' for fully convolutional model, 'lstm' for lstm based model
        ctype:          int, 0 for reg, 1 for classification
    """
    # Put the model in training mode
    model.train() 
    # Initializations
    num_batches = len(data)
    loss_avg = 0
    # iterate over batches for training
    for batch_idx in range(num_batches):
        # clear gradients and loss
        model.zero_grad()
        loss = 0
        # extract pitch tensor and score for the batch
        pitch_tensor = data[batch_idx]['pitch_tensor']
        score_tensor = data[batch_idx]['score_tensor'][:, metric]
        # prepare data for input to model
        model_input = pitch_tensor.clone()
        model_target = score_tensor.clone()
        model_target = torch.unsqueeze(model_target, 1)
        if ctype == 1:
            #model_input = model_input.long()
            model_target = model_target.long()
        # convert to cuda tensors if cuda available
        if torch.cuda.is_available():
            model_input = model_input.cuda()
            model_target = model_target.cuda()
        # wrap all tensors in pytorch Variable
        model_input = Variable(model_input)
        model_target = Variable(model_target)
        # compute forward pass for the network
        mini_batch_size = model_input.size(0)
        if mtype == 'lstm':
            model.init_hidden(mini_batch_size)
        #model_output = model(model_input)
        if model_input.shape[0] % 2 == 1:
            model_input = model_input[:-1]
            model_target = model_target[:-1]
        num = model_input.shape[0] // 2
        r1=torch.randperm(num)
        r2=torch.randperm(num)
        model_input1 = model_input[:num][r1]
        model_input2 = model_input[num:][r2]
        model_target1 = model_target[:num][r1]
        model_target2 = model_target[num:][r2]
        #conv_out1 = model.forward_conv(model_input1)
        #conv_out2 = model.forward_conv(model_input2)
        if encoder:
            model_output1 = model.forward_conv(model_input1)
            model_output2 = model.forward_conv(model_input2)
            loss = contrastive(model_target1, model_target2, model_output1, model_output2)
            #loss += criterion(model_output1, contrastive.label_map(model_target1.squeeze())) + criterion(model_output2, contrastive.label_map(model_target2.squeeze()))
        else:
            model_output1 = model.forward_once(model_input1)
            model_output2 = model.forward_once(model_input2)
            if classification:
                t1 = contrastive.label_map(model_target1.squeeze())
                t2 = contrastive.label_map(model_target2.squeeze())
            else: 
                t1 = model_target1
                t2 = model_target2
            loss = criterion(model_output1, t1) + criterion(model_output2, t2) + contrastive(model_target1, model_target2, model.forward_conv(model_input1), model.forward_conv(model_input2))
            
        # compute loss
        #loss = criterion(model_output, model_target)
        """ if contrastive:
            #print('model_output:', model_output.size(), 'targets', model_target.size())
            if strength:
                mse_str, c_str = strength
            else:
                mse_str, c_str = (1,1)
            c_loss = contrastive(model_target1, model_target2, model_output1, model_output2)
            loss = criterion(model_output1, contrastive.label_map(model_target1.squeeze())) + criterion(model_output2, contrastive.label_map(model_target2.squeeze()))
            if mse_str == 0:
                loss = c_loss
            elif c_str == 0:
                loss = loss
            else:
                loss = mse_str*loss + c_str*c_loss """
        # compute backward pass and step
        loss.backward()
        optimizer.step()
        # add loss
        #loss_avg += loss.data[0]
        loss_avg += loss.data
    loss_avg /= num_batches
    return loss_avg


# define training and validate method
def train_and_validate(model, criterion, optimizer, train_data, val_data, metric, mtype, ctype = 0, contrastive=None, strength=None, encoder = True, classification = True):
    """
    Defines the training and validation cycle for the input batched data for the conv model
    Args:
        model:          object, trained model of PitchContourAssessor class
        criterion:      object, of torch.nn.Functional class which defines the loss 
        optimizer:      object, of torch.optim class which defines the optimization algorithm
        train_data:     list, batched training data
        val_data:       list, batched validation data
        metric:         int, from 0 to 3, which metric to evaluate against
        mtype:          string, 'conv' for fully convolutional model, 'lstm' for lstm based model
        ctype:          int, 0 for regression, 1 for classification
    """
    # train the network
    train(model, criterion, optimizer, train_data, metric, mtype, ctype,contrastive=contrastive, strength=strength, encoder=encoder, classification= classification)   
    # evaluate the network on train data
    if classification:
        train_loss_avg, train_r_sq, train_accu, train_accu2 = (0,0,0,0) #eval_utils.eval_model(model, criterion, train_data, metric, mtype, ctype)
    else:
        train_loss_avg, train_r_sq, train_accu, train_accu2 = eval_utils.eval_model(model, criterion, train_data, metric, mtype, ctype)
    # evaluate the network on validation data
    if classification:
        val_loss_avg, val_r_sq, val_accu, val_accu2 = (0,0,0,0) #eval_utils.eval_model(model, criterion, val_data, metric, mtype, ctype)
    else:
        val_loss_avg, val_r_sq, val_accu, val_accu2 = eval_utils.eval_model(model, criterion, val_data, metric, mtype, ctype)
    # return values
    return train_loss_avg, train_r_sq, train_accu, train_accu2, val_loss_avg, val_r_sq, val_accu, val_accu2

def save(filename, perf_model):
    """
    Saves the saved model
    Args:
        filename:   name of the file 
        model:      torch.nn model 
    """
    #save_filename = '/Users/michaelfarren/Desktop/models/' + filename + '_Reg.pt'
    #save_filename = f'/content/gdrive/My Drive/Colab Notebooks/' + filename + '_Reg.pt'
    #save_filename = 'pc_runs/' + filename
    save_filename = filename
    torch.save(perf_model.state_dict(), save_filename)
    print('Saved as %s' % save_filename)

def time_since(since):
    """
    Returns the time elapsed between now and 'since'
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def adjust_learning_rate(optimizer, epoch, adjust_every):
    """
    Adjusts the learning rate of the optimizer based on the epoch
    Args:
       optimizer:      object, of torch.optim class 
       epoch:          int, epoch number
       adjust_every:   int, number of epochs after which adjustment is to done
    """
    if epoch > 1:
        if epoch % adjust_every == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5


