import torch
import time
import numpy as np
from torch import nn
from scipy.stats import pearsonr
from torch.autograd import Variable
from models.SpectralCRNN import SpectralCRNN_Reg_Dropout as SpectralCRNN #SpectralCRNN_Reg_Dropout
from tensorboard_logger import configure, log_value
from dataLoaders.SpectralDataset import SpectralDataset, SpectralDataLoader
from sklearn import metrics
from torch.optim import lr_scheduler

criterion = nn.MSELoss()
def evaluate_classification(targets, predictions):
    r2 = metrics.r2_score(targets, predictions)
    targets = np.round(targets*10).astype(int)
    predictions = predictions * 10
    predictions[predictions < 0] = 0
    predictions[predictions > 10] = 10
    predictions = np.round(predictions).astype(int)
    accuracy = metrics.accuracy_score(targets, predictions)
    return r2, accuracy

def evaluate_model(model, dataloader):
    model.eval()
    all_predictions = []
    all_targets = []
    for i, (data) in enumerate(dataloader):
        inputs, targets = data
        inputs = Variable(inputs.cuda(), requires_grad = False)
        targets = Variable(targets.cuda(), requires_grad = False)
        targets = targets.view(-1,1)
        model.init_hidden(inputs.size(0))
        out = model(inputs)
        all_predictions.extend(out.data.cpu().numpy())
        all_targets.extend(targets.data.cpu().numpy())
    return evaluate_classification(np.array(all_targets), np.array(all_predictions))

metric_type = {0:'musicality', 1:'note accuracy',2:'Rhythm Accuracy',3:'tonality'}
cross_test = True
instrument = 'flute'
test_instr = 'saxophone'
test_instr = instrument if not cross_test else test_instr
experiment = 'all-run'
metric = 0
BAND = 'middle'
ADD_NOISE_TEST = False
ADD_NOISE_VALID = False
NOISE_SHAPE = 'triangular'  #triangular, normal, or uniform
INPUT_REP = 'Cepstrum'
NAME = '{0}_{1}_{2}_{3}_{4}'.format(BAND, instrument, metric_type[metric], INPUT_REP, experiment)  

rep_params = {'method':INPUT_REP, 'n_fft':2048, 'n_mels': 96, 'hop_length': 1024, 'normalize': True}
datasets = {'flute':{'test':'/media/SSD/FBA/crossInstr/middle_Flute__test.dill', 'train':'/media/SSD/FBA/crossInstr/middle_Flute__train.dill', 'valid':'/media/SSD/FBA/crossInstr/middle_Flute__valid.dill'},
            'clarinet':{'test':'/media/SSD/FBA/crossInstr/middle_Bb Clarinet__test.dill', 'train':'/media/SSD/FBA/crossInstr/middle_Bb Clarinet__train.dill', 'valid':'/media/SSD/FBA/crossInstr/middle_Bb Clarinet__valid.dill'},
            'saxophone':{'test':'/media/SSD/FBA/crossInstr/middle_Alto Saxophone__test.dill', 'train':'/media/SSD/FBA/crossInstr/middle_Alto Saxophone__train.dill', 'valid':'/media/SSD/FBA/crossInstr/middle_Alto Saxophone__valid.dill'}}

train_dataset = SpectralDataset(datasets[test_instr]['train'], metric, rep_params)
train_dataloader = SpectralDataLoader(train_dataset, batch_size = 10, num_workers = 4, shuffle = True)

test_dataset = SpectralDataset(datasets[test_instr]['test'], metric, rep_params)
test_dataloader = SpectralDataLoader(test_dataset, batch_size = 10, num_workers = 1, shuffle = True)

valid_dataset = SpectralDataset(datasets[test_instr]['valid'], metric, rep_params)
valid_dataloader = SpectralDataLoader(valid_dataset, batch_size = 10, num_workers = 4, shuffle = True)

def eval_total(test, train, valid, model_name):
    model_path = 'runs/model_' + model_name
    model = SpectralCRNN().cuda()
    model = torch.load(model_path)

    criterion = nn.MSELoss()

    train_metrics = evaluate_model(model, train_dataloader)
    val_metrics = evaluate_model(model, valid_dataloader)
    test_metrics = evaluate_model(model, test_dataloader)

    print("Training Rsq:")
    print('Train r^2 : {}'.format(train_metrics[0]))
    print('Test r^2 : {}'.format(test_metrics[0]))
    print('Valid r^2 : {}'.format(val_metrics[0]))
    return test_metrics

eval_total(test_dataloader, train_dataloader, valid_dataloader, NAME)