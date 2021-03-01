import os
import dill
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

class PitchContourDataset(Dataset):
    """Dataset class for pitch contour based music performance assessment data"""

    def __init__(self, data_path):
        """
        Initializes the class, defines the number of datapoints
        Args:
            data_path:  full path to the file which contains the pitch contour data
        """
        self.perf_data = dill.load(open(data_path, 'rb'))
        #print(self.perf_data[0])
        print(len(self.perf_data))
        self.length = len(self.perf_data)

        # perform a few pre-processing steps
        for i in range(self.length):
            # store the length of the pitch contours for use later
            self.perf_data[i]['length'] = len(
                self.perf_data[i]['pitch_contour'])
            # store the length of the pitch contours for use later
            self.perf_data[i]['pitch_contour'] = self.normalize_pitch_contour(
                self.perf_data[i]['pitch_contour'])
        print(self.perf_data[0])

    def __getitem__(self, idx):
        """
        Returns a datapoint for a particular index
        Args:
            idx:        int, must range within [0, length of dataset)
        """
        return self.perf_data[idx]

    def __len__(self):
        """
        Return the size of the dataset
        """
        return self.length

    def plot_pitch_contour(self, idx):
        """
        Plots the pitch contour for visualization
        """
        pitch_contour = self.perf_data[idx]['pitch_contour']
        plt.plot(pitch_contour)
        plt.ylabel('pYin Pitch Contour (in Hz)')
        plt.show()

    def normalize_pitch_contour(self, pitch_contour):
        """
        Returns the normalized pitch contour after converting to floating point MIDI
        Args:
            pitch_contour:      np 1-D array, contains pitch in Hz
        """
        # convert to MIDI first
        a4 = 440.0
        pitch_contour[pitch_contour != 0] = 69 + 12 * \
            np.log2(pitch_contour[pitch_contour != 0] / a4)
        # normalize pitch (restrict between 36 to 108 MIDI notes)
        normalized_pitch = pitch_contour #/ 127.0
        normalized_pitch[normalized_pitch != 0] = (normalized_pitch[normalized_pitch != 0] - 36.0)/72.0 
        return normalized_pitch
