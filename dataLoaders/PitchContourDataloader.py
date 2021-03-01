import os
import sys
import collections
import numpy as np
import random
from random import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as multiprocessing
from dataLoaders.PitchContourDataset import PitchContourDataset

class PitchContourDataloader(DataLoader):
    """
    Dataloader class for pitch contour music performance assessment data
    """

    def __init__(self, dataset, num_data_pts, num_batches=1):
        """
        Initializes the class, defines the number of batches and other parameters
        Args:
                dataset:        object of the PitchContourDataset class, should be properly initialized
                num_data_pts:   int, number of data points to be consideted while loading the data
                num_batches:    int, number of batches to be considered
        """
        # check if input parameters are accurate
        np.random.seed(1)
        length = len(dataset)
        assert num_data_pts <= dataset.__len__()
        assert num_batches <= num_data_pts
        self.dataset = dataset
        self.num_data_pts = num_data_pts
        self.num_batches = num_batches
        self.indices = np.arange(self.num_data_pts)
        np.random.shuffle(self.indices)
        self.mini_batch_size = int(np.floor(self.num_data_pts / self.num_batches))
        
    def get_sorted_data(self):
        """
        Returns data points sorted in descending order of pitch contour length
        """
        # get the lengths of the 1st num_songs
        song_len = np.empty([self.num_data_pts])
        for i in range(self.num_data_pts):
            data_point = self.dataset.__getitem__(i)
            song_len[i] = data_point['length']

        # get the sorted indices
        sorted_idx = (-song_len).argsort(kind='mergesort')

        # iterate and collect data
        sorted_data = list()
        for i in range(self.num_data_pts):
            sorted_data.append(self.dataset.__getitem__(sorted_idx[i]))
        return sorted_data

    def create_batched_data(self):
        """
        Returns batched data after sorting as a list
        """
        # sort the data first
        sorted_data = self.get_sorted_data()
        # batch the sorted data
        batched_data = [None] * self.num_batches
        count = 0
        for batch_num in range(self.num_batches):
            batched_data[batch_num] = list()
            longest_seq_len = sorted_data[count]['length']
            pitch_tensor = torch.zeros(self.mini_batch_size, longest_seq_len)
            score_tensor = torch.zeros(self.mini_batch_size, len(sorted_data[count]['ratings']))
            class_score_tensor = torch.zeros(self.mini_batch_size, len(sorted_data[count]['class_ratings'])).long()
            zero_pad = ZeroPad(longest_seq_len)
            for seq_num in range(self.mini_batch_size):
                # convert pitch contour to torch tensor
                pc_tensor = torch.from_numpy(sorted_data[count]['pitch_contour'])
                pitch_tensor[seq_num, :] = zero_pad.apply_pad(pc_tensor.float())
                # convert score tuple to torch tensor
                s_tensor = torch.from_numpy(np.asarray(sorted_data[count]['ratings']))
                c_tensor = torch.from_numpy(np.asarray(sorted_data[count]['class_ratings'])).long()
                score_tensor[seq_num, :] = s_tensor
                class_score_tensor[seq_num, :] = c_tensor
                count += 1
            data = {}
            data['pitch_tensor'] = pitch_tensor
            data['score_tensor'] = score_tensor
            data['class_tensor'] = class_score_tensor
            batched_data[batch_num] = data
        return batched_data

    def create_split_data(self, chunk_len, hop):
        """
        Returns batched data which is split into chunks
        Args:
            chunk_len:  legnth of the chunk in samples
            hop:    hop length in samples
        """
        random.seed(0)
        indices = self.indices
        #print(indices)
        #np.random.shuffle(indices)
        num_training_songs = int(0.8 * self.num_data_pts)
        num_validation_songs = int(0.1 * self.num_data_pts)
        num_testing_songs = num_validation_songs
        train_split = []
        for i in range(num_training_songs):
            data = self.dataset.__getitem__(indices[i])
            pc = data['pitch_contour']
            #print(data['student_id'])
            gt = data['ratings']
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
            score_tensor = torch.zeros(mini_batch_size, len(train_split[count]['ratings']))
            for seq_num in range(mini_batch_size):
                # convert pitch contour to torch tensor
                pc_tensor = torch.from_numpy(train_split[count]['pitch_contour'])
                pitch_tensor[seq_num, :] = pc_tensor.float()
                # convert score tuple to torch tensor
                s_tensor = torch.from_numpy(np.asarray(train_split[count]['ratings']))
                score_tensor[seq_num, :] = s_tensor
                count += 1
            dummy = {}
            dummy['pitch_tensor'] = pitch_tensor
            dummy['score_tensor'] = score_tensor
            batched_data[batch_num] = dummy 
        
        val_split = []
        val_batch_full = []
        for i in range(num_training_songs, num_training_songs + num_validation_songs):
            data = self.dataset.__getitem__(indices[i])
            pc = data['pitch_contour']
            #print(data['student_id'])
            gt = data['ratings']
            count = 0
            d = {}
            d['pitch_tensor'] = torch.from_numpy(pc).float().view(1,-1)
            d['score_tensor'] = torch.from_numpy(np.asarray(gt)).float().view(1,-1)
            val_batch_full.append(d)
            if len(pc) < chunk_len:
                zeropad_pc = np.zeros((chunk_len,))
                zeropad_pc[:pc.shape[0],] = pc
                pc = zeropad_pc
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
        score_tensor = torch.zeros(mini_batch_size, len(val_split[count]['ratings']))
        for seq_num in range(mini_batch_size):
            # convert pitch contour to torch tensor
            pc_tensor = torch.from_numpy(val_split[count]['pitch_contour'])
            pitch_tensor[seq_num, :] = pc_tensor.float()
            # convert score tuple to torch tensor
            s_tensor = torch.from_numpy(np.asarray(val_split[count]['ratings']))
            score_tensor[seq_num, :] = s_tensor
            count += 1
        dummy = {}
        dummy['pitch_tensor'] = pitch_tensor
        dummy['score_tensor'] = score_tensor
        val_batch = [dummy]
        
        test_split = []
        test_batch_full = []
        for i in range(num_training_songs + num_validation_songs, num_training_songs + num_validation_songs + num_testing_songs):
            data = self.dataset.__getitem__(indices[i])
            pc = data['pitch_contour']
            #print(data['student_id'])
            gt = data['ratings']
            count = 0
            d = {}
            d['pitch_tensor'] = torch.from_numpy(pc).float().view(1,-1)
            d['score_tensor'] = torch.from_numpy(np.asarray(gt)).float().view(1,-1)
            test_batch_full.append(d)
            if len(pc) < chunk_len:
                zeropad_pc = np.zeros((chunk_len,))
                zeropad_pc[:pc.shape[0],] = pc
                pc = zeropad_pc
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
        score_tensor = torch.zeros(mini_batch_size, len(test_split[count]['ratings']))
        for seq_num in range(mini_batch_size):
            # convert pitch contour to torch tensor
            pc_tensor = torch.from_numpy(test_split[count]['pitch_contour'])
            pitch_tensor[seq_num, :] = pc_tensor.float()
            # convert score tuple to torch tensor
            s_tensor = torch.from_numpy(np.asarray(test_split[count]['ratings']))
            score_tensor[seq_num, :] = s_tensor
            count += 1
        dummy = {}
        dummy['pitch_tensor'] = pitch_tensor
        dummy['score_tensor'] = score_tensor
        test_batch = [dummy]           
        return batched_data, val_batch, val_batch_full, test_batch, test_batch_full


class ZeroPad(object):
    """
    Class to perform zero padding of input sequences
    """

    def __init__(self, seq_length):
        """
        Initializes the ZeroPad class
        Args:
            seq_length:     int, length of the final zero padded sequence
        """
        assert isinstance(seq_length, int)
        self.seq_length = seq_length

    def apply_pad(self, sample):
        """
        Zero-Pads the input 1-D float tensor so that it becomes the same length as the seq_length
             member of the class
        Args:
            sample: 1-D double tensor
        """
        assert self.seq_length >= sample.size(0)
        if self.seq_length == sample.size(0):
            return sample
        zero_pad = torch.zeros(self.seq_length - int(sample.size(0)))
        zero_padded_sample = torch.cat((sample, zero_pad), 0)
        return zero_padded_sample
