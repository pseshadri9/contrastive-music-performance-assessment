import torch
import time
import numpy as np
from torch import nn
from torch.autograd import Variable
from models.SpectralCRNN import SpectralCRNN_Reg_Clarinet as SpectralCRNN #SpectralCRNN_Reg_Dropout
from tensorboard_logger import configure, log_value
from dataLoaders.SpectralDataset import SpectralDataset, SpectralDataLoader
from sklearn import metrics
from torch.optim import lr_scheduler
from predict_one_mel import predict_one_mel
from test_spectral import eval_total

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate_classification(targets, predictions):
    r2 = metrics.r2_score(targets, predictions)
    targets = np.round(targets*10).astype(int)
    predictions = predictions * 10
    predictions[predictions < 0] = 0
    predictions[predictions > 10] = 10
    predictions = np.round(predictions).astype(int)
    accuracy = metrics.accuracy_score(targets, predictions)
    return r2, accuracy


#SET CONSTANTS
metric_type = {0:'musicality', 1:'note accuracy',2:'Rhythm Accuracy',3:'tonality'}
instrument = 'saxophone'
cross_instrument = 'saxophone'
experiment = 'all-run'
metric = 3
BAND = 'middle'
ADD_NOISE_TEST = False
ADD_NOISE_VALID = False
NOISE_SHAPE = 'triangular'  #triangular, normal, or uniform
INPUT_REP = 'Cepstrum'
NAME = '{0}_{1}_{2}_{3}_{4}'.format(BAND, instrument, metric_type[metric], INPUT_REP, experiment)  

#NAME = INPUT_REP + '_' + NAME

if ADD_NOISE_TEST:
    NAME += '_' + NOISE_SHAPE + 'Noise_test'
    if ADD_NOISE_VALID:
        NAME += '_valid'
datasets = {'flute':{'test':'/media/SSD/FBA/crossInstr/middle_Flute__test.dill', 'train':'/media/SSD/FBA/crossInstr/middle_Flute__train_fixed.dill', 'valid':'/media/SSD/FBA/crossInstr/middle_Flute__valid.dill'},
            'clarinet':{'test':'/media/SSD/FBA/crossInstr/middle_Bb Clarinet__test.dill', 'train':'/media/SSD/FBA/crossInstr/middle_Bb Clarinet__train.dill', 'valid':'/media/SSD/FBA/crossInstr/middle_Bb Clarinet__valid.dill'},
            'saxophone':{'test':'/media/SSD/FBA/crossInstr/middle_Alto Saxophone__test.dill', 'train':'/media/SSD/FBA/crossInstr/middle_Alto Saxophone__train.dill', 'valid':'/media/SSD/FBA/crossInstr/middle_Alto Saxophone__valid.dill'}}
# Configure tensorboard logger
configure('runs/' + NAME, flush_secs = 2)

# Parameteres for Spectral Representation
rep_params = {'method':INPUT_REP, 'n_fft':2048, 'n_mels': 96, 'hop_length': 1024, 'normalize': True, 'n_bins': 144, 'bins_per_octave': 24}

# Load Datasets
if torch.cuda.is_available():
    train_dataset = SpectralDataset(datasets[instrument]['train'], metric, rep_params) # /media/SSD/FBA/trains/middle_2_data_4_train.dill
    test_dataset = SpectralDataset(datasets[instrument]['test'], metric, rep_params)
    valid_dataset = SpectralDataset(datasets[instrument]['valid'], metric, rep_params)
    cross_instrument_dataloaders = dict()
    cross_instrument_datasets = dict()
    for instr in ['flute', 'clarinet', 'saxophone']:
        if instr != instrument:
            cross_instrument_datasets[instr] = SpectralDataset(datasets[instr]['test'], metric, rep_params)
            cross_instrument_dataloaders[instr] = SpectralDataLoader(cross_instrument_datasets[instr], batch_size = 10, num_workers = 6, shuffle = True)
else:
    train_dataset = SpectralDataset('/Volumes/Farren/Python_stuff/dat/train_' + BAND + '.dill', 0, rep_params)
    test_dataset = SpectralDataset('/Volumes/Farren/Python_stuff/dat/test_' + BAND + '.dill', 0, rep_params)
    valid_dataset = SpectralDataset('/Volumes/Farren/Python_stuff/dat/valid_' + BAND + '.dill', 0, rep_params)

train_dataloader = SpectralDataLoader(train_dataset, batch_size = 10, num_workers = 6, shuffle = True)
test_dataloader = SpectralDataLoader(test_dataset, batch_size = 10, num_workers = 6, shuffle = True)
valid_dataloader = SpectralDataLoader(valid_dataset, batch_size = 10, num_workers = 6, shuffle = True)
#cross_test_dataloader = SpectralDataLoader(cross_test_dataset, batch_size = 10, num_workers = 4, shuffle = True)


# Define Model
model = SpectralCRNN().cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001) #0.0001
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma = 0.1) #gamma=0.1

batch_time = AverageMeter()
data_time = AverageMeter()

train_loss = 0
validation_loss = 0

num_epochs = 400
best_val = 0.0
epoch_time = time.time()
np.random.seed(1)
best_epoch = 0
for epoch in range(num_epochs):
    try:
        model.train()
        # scheduler.step()
        avg_loss = 0.0
        end = time.time()
        all_predictions = []
        all_targets = []
        losses = AverageMeter()
        for i, (data) in enumerate(train_dataloader):
            inputs, targets = data
            # Add any desired noise
            if ADD_NOISE_TEST:
                if NOISE_SHAPE == 'triangular':
                    noise = np.random.triangular(left=-0.1, mode=0, right=0.1, size=targets.size()[0])
                elif NOISE_SHAPE == 'normal':
                    noise = np.random.normal(loc=0.0, scale=0.1/3, size=targets.size()[0])
                elif NOISE_SHAPE == 'uniform':
                    noise = np.random.uniform(low=-0.1, high=0.1, size=targets.size()[0])
                targets = torch.Tensor(targets.data.numpy() + noise)

            data_time.update(time.time() - end)
            inputs = Variable(inputs.cuda(), requires_grad = False)
            targets = Variable(targets.cuda(), requires_grad = False)
            targets = targets.view(-1,1)
            model.init_hidden(inputs.size(0))
            out = model(inputs)
            all_predictions.extend(out.data.cpu().numpy())
            all_targets.extend(targets.data.cpu().numpy())
            loss = criterion(out, targets)
            loss_value = loss.data.item()
            losses.update(loss_value, inputs.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                (epoch + 1), (i + 1), len(train_dataloader), batch_time=batch_time,
                data_time=data_time, loss=losses))
        print('Epoch Completed. Validating')
        train_loss = losses.avg
        train_r2, train_accuracy = evaluate_classification(np.array(all_targets), np.array(all_predictions))
        print('Train r^2: ', str(train_r2))
        
        model.eval()
        losses = AverageMeter()
        all_predictions = []
        all_targets = []
        for i, (data) in enumerate(valid_dataloader):
            inputs, targets = data
            # Add any desired noise to the labels
            if ADD_NOISE_VALID:
                if NOISE_SHAPE == 'triangular':
                    noise = np.random.triangular(left=-0.1, mode=0, right=0.1, size=targets.size()[0])
                elif NOISE_SHAPE == 'normal':
                    noise = np.random.normal(loc=0.0, scale=0.1/3, size=targets.size()[0])
                elif NOISE_SHAPE == 'uniform':
                    noise = np.random.uniform(low=-0.1, high=0.1, size=targets.size()[0])
                targets = torch.Tensor(targets.data.numpy() + noise)

            data_time.update(time.time() - end)
            inputs = Variable(inputs.cuda(), requires_grad = False)
            targets = Variable(targets.cuda(), requires_grad = False)
            targets = targets.view(-1,1)
            model.init_hidden(inputs.size(0))
            out = model(inputs)
            all_predictions.extend(out.data.cpu().numpy())
            all_targets.extend(targets.data.cpu().numpy())
            loss = criterion(out, targets)
            loss_value = loss.data.item()
            losses.update(loss_value, inputs.size(0))
        print('Validating Completed. Loss: {}'.format(losses.avg))
        valid_loss = losses.avg
        val_r2, val_accuracy = evaluate_classification(np.array(all_targets), np.array(all_predictions))
        try:
            for instr in cross_instrument_dataloaders.keys():
                test_r2, test_acc = eval_total(cross_instrument_dataloaders[instr], train_dataloader, valid_dataloader, NAME)
                log_value('Test R2- '+instr, test_r2, epoch)
                #log_value('Test Loss- ' +instr, test_loss, epoch)
        except:
            pass
        print('Val r^2: ', str(val_r2))
        log_value('Train Loss', train_loss, epoch)
        log_value('Validation Loss', valid_loss, epoch)
        log_value('Training Accuracy', train_accuracy, epoch)
        log_value('Validation Accuracy', val_accuracy, epoch)
        log_value('Training R2', train_r2, epoch)
        log_value('Validation R2', val_r2, epoch)
        if val_r2 > best_val:
            best_epoch = epoch
            best_val = val_r2
            torch.save(model, 'runs/model_' + NAME)
        if best_epoch < epoch - 100:
            break
    except KeyboardInterrupt:
        torch.save(model, 'runs/lastmodel_')
        break
eval_total(test_dataloader, train_dataloader, valid_dataloader, NAME)

