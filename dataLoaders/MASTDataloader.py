import os
import sys
import collections
import numpy as np
import random
from random import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as multiprocessing
from dataLoaders.MASTDataset import MASTDataset

class MASTDataloader(DataLoader):
    """
	Dataloader class for pitch contour music performance assessment data
	"""

    def __init__(self, dataset):
        """
        Initializes the class, defines the number of batches and other parameters
        Args:
                dataset:  		object of the MASTDataset class, should be properly initialized
        """
        np.random.seed(1)
        self.dataset = dataset
        self.num_data_pts = 260
        self.border_entry = 266
        self.num_batches = 10
        self.indices = np.arange(self.num_data_pts)
        np.random.shuffle(self.indices)
        #print(self.indices)
        self.mini_batch_size = int(np.floor(self.num_data_pts / self.num_batches)) * 2


    def create_split_data(self):
        """
        Returns batched data after sorting as a list
        """


    def create_split_data(self, chunk_len, hop):
        """
        Returns batched data which is split into chunks
        Args:
            chunk_len:  legnth of the chunk in samples
            hop:	hop length in samples
        """
        random.seed(0)
        indices = self.indices
        #print(indices)
        num_training_songs = int(0.8 * self.num_data_pts)
        num_validation_songs = int(0.1 * self.num_data_pts)
        num_testing_songs = num_validation_songs

        #### Create Training Data
        train_split = []
        for i in range(num_training_songs):
            f0, rating = self.dataset.__getitem__(indices[i])
            pc = f0
            gt = torch.from_numpy(np.ones((1, 1)) * rating).float()
            count = 0
            if len(pc) < chunk_len:
                zeropad_pc = np.zeros((chunk_len,))
                zeropad_pc[:pc.shape[0],] = pc
                pc = zeropad_pc 
            while count + chunk_len <= len(pc):
                d = {}
                d['pitch_contour'] = pc[count: count+chunk_len]
                d['ratings'] = gt
                train_split.append(d)
                count += hop
            f0, rating = self.dataset.__getitem__(indices[i] + self.border_entry)
            pc = f0
            gt = torch.from_numpy(np.ones((1, 1)) * rating).float()
            count = 0
            if len(pc) < chunk_len:
                zeropad_pc = np.zeros((chunk_len,))
                zeropad_pc[:pc.shape[0],] = pc
                pc = zeropad_pc 
            while count + chunk_len <= len(pc):
                d = {}
                d['pitch_contour'] = pc[count: count+chunk_len]
                d['ratings'] = gt
                train_split.append(d)
                count += hop
        shuffle(train_split)
        num_data_pts = len(train_split)
        batched_data = [None] * self.num_batches
        mini_batch_size = int(np.floor(num_data_pts / self.num_batches))
        count = 0
        for batch_num in range(self.num_batches):
            batched_data[batch_num] = list()
            pitch_tensor = torch.zeros(mini_batch_size, chunk_len)
            score_tensor = torch.zeros(mini_batch_size, 1)
            for seq_num in range(mini_batch_size):
                # convert pitch contour to torch tensor
                pc_tensor = torch.from_numpy(train_split[count]['pitch_contour'])
                pitch_tensor[seq_num, :] = pc_tensor.float()
                # place the score tensor
                score_tensor[seq_num, :] = train_split[count]['ratings']
                count += 1
            dummy = {}
            dummy['pitch_tensor'] = pitch_tensor
            dummy['score_tensor'] = score_tensor
            batched_data[batch_num] = dummy 
        
        #### Create Validation Data
        val_split = []
        val_batch_full = []
        for i in range(num_training_songs, num_training_songs + num_validation_songs):
            f0, rating = self.dataset.__getitem__(indices[i])
            pc = f0
            gt = torch.from_numpy(np.ones((1, 1)) * rating).float()
            count = 0
            d = {}
            if len(pc) < chunk_len:
                zeropad_pc = np.zeros((chunk_len,))
                zeropad_pc[:pc.shape[0],] = pc
                pc = zeropad_pc
                d['pitch_tensor'] = torch.from_numpy(pc).float().view(1,-1)
                d['score_tensor'] = gt
                val_batch_full.append(d)
            while count + chunk_len <= len(pc):
                d = {}
                d['pitch_contour'] = pc[count: count+chunk_len]
                d['ratings'] = gt
                val_split.append(d)
                count += hop
            f0, rating = self.dataset.__getitem__(indices[i] + self.border_entry)
            pc = f0
            gt = torch.from_numpy(np.ones((1, 1)) * rating).float()
            count = 0
            d = {}
            if len(pc) < chunk_len:
                zeropad_pc = np.zeros((chunk_len,))
                zeropad_pc[:pc.shape[0],] = pc
                pc = zeropad_pc
                d['pitch_tensor'] = torch.from_numpy(pc).float().view(1,-1)
                d['score_tensor'] = gt
                val_batch_full.append(d)
            while count + chunk_len <= len(pc):
                d = {}
                d['pitch_contour'] = pc[count: count+chunk_len]
                d['ratings'] = gt
                val_split.append(d)
                count += hop
        shuffle(val_split)
        num_data_pts = len(val_split)
        mini_batch_size = num_data_pts
        count = 0
        pitch_tensor = torch.zeros(mini_batch_size, chunk_len)
        score_tensor = torch.zeros(mini_batch_size, 1)
        for seq_num in range(mini_batch_size):
            # convert pitch contour to torch tensor
            pc_tensor = torch.from_numpy(val_split[count]['pitch_contour'])
            pitch_tensor[seq_num, :] = pc_tensor.float()
            # place the score tensor
            score_tensor[seq_num, :] = val_split[count]['ratings']
            count += 1
        dummy = {}
        dummy['pitch_tensor'] = pitch_tensor
        dummy['score_tensor'] = score_tensor
        val_batch = [dummy]
        
        #### Create Test Data
        test_split = []
        test_batch_full = []
        for i in range(num_training_songs + num_validation_songs, num_training_songs + num_validation_songs + num_testing_songs):
            f0, rating = self.dataset.__getitem__(indices[i])
            pc = f0
            gt = torch.from_numpy(np.ones((1, 1)) * rating).float()
            count = 0
            d = {}
            if len(pc) < chunk_len:
                zeropad_pc = np.zeros((chunk_len,))
                zeropad_pc[:pc.shape[0],] = pc
                pc = zeropad_pc
                d['pitch_tensor'] = torch.from_numpy(pc).float().view(1,-1)
                d['score_tensor'] = gt
                test_batch_full.append(d)
            while count + chunk_len <= len(pc):
                d = {}
                d['pitch_contour'] = pc[count: count+chunk_len]
                d['ratings'] = gt
                test_split.append(d)
                count += hop
            f0, rating = self.dataset.__getitem__(indices[i] + self.border_entry)
            pc = f0
            gt = torch.from_numpy(np.ones((1, 1)) * rating).float()
            count = 0
            d = {}
            if len(pc) < chunk_len:
                zeropad_pc = np.zeros((chunk_len,))
                zeropad_pc[:pc.shape[0],] = pc
                pc = zeropad_pc
                d['pitch_tensor'] = torch.from_numpy(pc).float().view(1,-1)
                d['score_tensor'] = gt
                test_batch_full.append(d)
            while count + chunk_len <= len(pc):
                d = {}
                d['pitch_contour'] = pc[count: count+chunk_len]
                d['ratings'] = gt
                test_split.append(d)
                count += hop
        num_data_pts = len(test_split)
        mini_batch_size = num_data_pts
        count = 0
        pitch_tensor = torch.zeros(mini_batch_size, chunk_len)
        score_tensor = torch.zeros(mini_batch_size, 1)
        for seq_num in range(mini_batch_size):
            # convert pitch contour to torch tensor
            pc_tensor = torch.from_numpy(test_split[count]['pitch_contour'])
            pitch_tensor[seq_num, :] = pc_tensor.float()
            # place the score tensor
            score_tensor[seq_num, :] = test_split[count]['ratings']
            count += 1
        dummy = {}
        dummy['pitch_tensor'] = pitch_tensor
        dummy['score_tensor'] = score_tensor
        test_batch = [dummy]           
        return batched_data, val_batch, val_batch_full, test_batch, test_batch_full