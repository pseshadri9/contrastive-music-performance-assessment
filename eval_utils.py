import os
import sys
import math
import time
import scipy.stats as ss
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn import metrics
from contrastive_utils import ContrastiveLoss
from models.PCConvNet import PCConvNet, PCConvNetCls, PCConvNetContrastive

"""
Contains standard utility functions for training and testing evaluations
"""

def eval_regression(target, pred):
    """
    Calculates the standard regression metrics
    Args:
        target:     (N x 1) torch Float tensor, actual ground truth
        pred:       (N x 1) torch Float tensor, predicted values from the regression model
    """
    if torch.cuda.is_available():
        pred_np = pred.clone().cpu().numpy()
        target_np = target.clone().cpu().numpy()
    else:
        pred_np = pred.clone().numpy()
        target_np = target.clone().numpy()
    # compute r-sq score 
    #print(pred_np)
    pred_np[pred_np < 0] = 0
    pred_np[pred_np > 1] = 1
    r_sq = metrics.r2_score(target_np, pred_np)
    # compute 11-class classification accuracy
    pred_class = np.rint(pred_np * 10)
    pred_class[pred_class < 0] = 0
    pred_class[pred_class > 10] = 10
    target_class = np.rint(target_np * 10)
    pred_class.astype(int)
    target_class.astype(int)
    accu = metrics.accuracy_score(target_class, pred_class, normalize=True) 
    pred_class[pred_class < 5] = 0
    pred_class[pred_class >= 5] = 1
    target_class[target_class < 5] = 0
    target_class[target_class >= 5] = 1
    #print(target_class)
    accu2 = metrics.accuracy_score(target_class, pred_class, normalize=True)
    return r_sq, accu, accu2

def eval_model(model, criterion, data, metric, mtype, ctype, extra_outs = 0):
    """
    Returns the model performance metrics
    Args:
        model:          object, trained model of PitchContourAssessor class
        criterion:      object, of torch.nn.Functional class which defines the loss 
        data:           list, batched testing data
        metric:         int, from 0 to 3, which metric to evaluate against
        mtype:          string, 'conv' for fully convolutional model, 'lstm' for lstm based model
        extra_outs:     returns the target and predicted values if true
        ctype:          int, 0 for regression, 1 for classification
    """
    # put the model in eval mode
    model.eval()
    # intialize variables
    num_batches = len(data)
    pred = np.array([])
    target = np.array([])
    loss_avg = 0
    # iterate over batches for validation
    for batch_idx in range(num_batches):
        # extract pitch tensor and score for the batch
        pitch_tensor = torch.Tensor(data[batch_idx]['pitch_tensor'])
        score_tensor = torch.Tensor(data[batch_idx]['score_tensor'][:, metric])
        #score_tensor = torch.unsqueeze(score_tensor, 1)
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
        try:
            if type(model) is PCConvNetContrastive:
                model_output = model.forward_once(model_input)
            else:
                model_output = model.forward(model_input)
        except Exception as  e:
            print(e)
            continue
        # compute loss
        loss = criterion(model_output, model_target)
        #loss_avg += loss.data[0]
        loss_avg += loss.data
        # concatenate target and pred for computing validation metrics
        if ctype == 0:
            pred = torch.cat((pred, model_output.data.view(-1)), 0) if pred.size else model_output.data.view(-1)
        else:
            pred = torch.cat((pred, model_output.data), 0) if pred.size else model_output.data
        target = torch.cat((target, model_target), 0) if target.size else model_target
        #print(model_output.shape,pred.shape, target.shape, model_target.shape)
    #ctype = 1
    if ctype == 1:
        #pred = nn.functional.softmax(pred).data
        #_, pred = torch.max(pred, 1) 
        print(score_tensor.shape)
        pred = torch.argmax(model_output, dim=1)
        print(pred.shape)
    #print(model_output.shape, pred.shape)
    r_sq, accu, accu2 = eval_regression(target.T.flatten(), pred)
    loss_avg /= num_batches
    if extra_outs:
        return loss_avg, r_sq, accu, accu2, pred, target
    else:
        return loss_avg, r_sq, accu, accu2

def compute_saliency_maps(X, y, model):
    """
    Computes a regression score saliency map using the model for pitch contour X and score y.
    Args:
        X: Input pitch contour, torch tensor of shape (N, W)
        y: Regression score for X, Float tensor of shape (N,)
        model: A pretrained model that will be used to compute the saliency map.
    """
    # Set the model is in "test" mode
    model.eval()
    
    # Wrap the input tensors in Variables
    X_var = Variable(X, requires_grad=True)
    y_var = Variable(y, requires_grad=False)
    saliency = None
    
    # compute forward pass and class scores
    pred_scores = model.forward(X_var)
    print(pred_scores)

    # compute gradient wrt input
    pred_scores.sum().backward()
    saliency = X_var.grad

    saliency = saliency.data
    return saliency

def eval_model_preds(model, criterion, data, metric, mtype, ctype, extra_outs = 0, latent = False):
    """
    Returns the model performance metrics
    Args:
        model:          object, trained model of PitchContourAssessor class
        criterion:      object, of torch.nn.Functional class which defines the loss 
        data:           list, batched testing data
        metric:         int, from 0 to 3, which metric to evaluate against
        mtype:          string, 'conv' for fully convolutional model, 'lstm' for lstm based model
        extra_outs:     returns the target and predicted values if true
        ctype:          int, 0 for regression, 1 for classification
    """
    # put the model in eval mode
    model.eval()
    # intialize variables
    num_batches = len(data)
    pred = np.array([])
    target = np.array([])
    loss_avg = 0
    # iterate over batches for validation
    for batch_idx in range(num_batches):
        # extract pitch tensor and score for the batch
        pitch_tensor = data[batch_idx]['pitch_tensor']
        score_tensor = data[batch_idx]['score_tensor'][:, metric]
        score_tensor = torch.unsqueeze(score_tensor, 1)
        # prepare data for input to model
        model_input = pitch_tensor.clone()
        model_target = score_tensor.clone()
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
        if latent:
            model_output = model.forward_conv(model_input, latent_test = True)
        else:
            if type(model) is PCConvNet:
                model_output = torch.mean(model.forward_conv(model_input), 2)
            else:
                model_output = model.forward_conv(model_input)
        # compute loss
        if not latent:
            loss = criterion(model_output, model_target)
        #loss_avg += loss.data[0]
            loss_avg += loss.data
        # concatenate target and pred for computing validation metrics
        if latent:
            if pred.size and False:
                print(pred.size())
            pred = torch.cat((pred, model_output.data), dim=0) if pred.size else model_output.data
        elif ctype == 0:
            pred = torch.cat((pred, model_output.data), 0) if pred.size else model_output.data
        else:
            pred = torch.cat((pred, model_output.data), 0) if pred.size else model_output.data
        target = torch.cat((target, score_tensor), 0) if target.size else score_tensor
    if ctype == 1:
        pred = nn.functional.softmax(pred).data
        _, pred = torch.max(pred, 1) 
    return pred, target

def eval_acc_contrastive(model, criterion, data, metric, mtype, ctype, extra_outs = 0, classification = True, criterion_CE = None, preds = False):
        model.eval()
        # intialize variables
        num_batches = len(data)
        pred = np.array([])
        target = np.array([])
        preds_out = None
        loss_avg = 0
        ce_loss_avg = 0
        if classification:
            correct = 0
            total = 0
        # iterate over batches for validation
        for batch_idx in range(1, num_batches - 1, 2):
            # extract pitch tensor and score for the batch
            pitch_tensor = torch.Tensor(data[batch_idx]['pitch_tensor'])
            score_tensor = torch.Tensor(data[batch_idx]['score_tensor'][:, metric])
            score_tensor = torch.unsqueeze(score_tensor, 1)
            pitch_tensor2 = torch.Tensor(data[batch_idx-1]['pitch_tensor'])
            score_tensor2 = torch.Tensor(data[batch_idx-1]['score_tensor'][:, metric])
            score_tensor2 = torch.unsqueeze(score_tensor2, 1)
            # prepare data for input to model
            model_input = pitch_tensor.clone()
            model_target = score_tensor.clone()
            model_input2 = pitch_tensor2.clone()
            model_target2 = score_tensor2.clone()
            if ctype == 1:
                #model_input = model_input.long()
                model_target = model_target.long()
            # convert to cuda tensors if cuda available
            if torch.cuda.is_available():
                model_input = model_input.cuda()
                model_target = model_target.cuda()
                model_input2 = pitch_tensor2.cuda()
                model_target2 = score_tensor2.cuda()
            # wrap all tensors in pytorch Variable
            model_input = Variable(model_input)
            model_target = Variable(model_target)
            model_input2 = Variable(model_input2)
            model_target2 = Variable(model_target2)
            # compute forward pass for the network
            mini_batch_size = model_input.size(0)
            if mtype == 'lstm':
                model.init_hidden(mini_batch_size)
            try:
                conv_out1 = model.forward_conv(model_input)
                conv_out2 = model.forward_conv(model_input2)
                model_output1 = model.forward_once(model_input)
                model_output2 = model.forward_once(model_input2)
            except Exception as e:
                print(e)
                continue
            # compute loss
            #print(model_target2.shape, conv_out2.shape)
            loss = criterion(model_target, model_target2, conv_out1, conv_out2)
            if criterion_CE:
                ce_loss = criterion_CE(model_output1, criterion.label_map(model_target.squeeze(dim=1))) + criterion_CE(model_output2, criterion.label_map(model_target2.squeeze(dim=1)))
                ce_loss_avg += ce_loss.data
            #print(loss)
            #loss_avg += loss.data[0]
            loss_avg += loss.data
            # concatenate target and pred for computing validation metrics
            if ctype == 0:
                #print(conv_out1.shape)
                if preds_out is None and preds:
                    #print(preds_out.shape)
                    preds_out = torch.cat((model_output1, model_output2), dim=0)
                elif preds:
                    conv = torch.cat((model_output1, model_output2), dim=0)
                    preds_out = torch.cat((preds_out, conv), dim=0)
                #print(preds_out.shape)
                target = torch.cat((target, score_tensor), 0) if target.size else score_tensor
                target = torch.cat((target, score_tensor2), 0) if target.size else score_tensor2
                pred = model.classifier(torch.cat((model_output1, model_output2), dim=0))
                pred = torch.argmax(pred, dim=1)
                correct_s = torch.sum(torch.eq(pred, criterion.label_map((torch.cat((model_target, model_target2), dim=0)).squeeze())).float())
                correct += correct_s
                total += pred.shape[0]
                #print(correct.data, total)
            else:
                pred = torch.cat((pred, model_output1.data), 0) if pred.size else model_output1.data
        acc = correct/total
        loss_avg /= num_batches
        ce_loss_avg /= num_batches
        return (loss_avg, acc, ce_loss_avg) if not preds else (preds_out, target)

