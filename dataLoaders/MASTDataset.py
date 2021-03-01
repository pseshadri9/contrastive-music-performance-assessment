import os
import torch
from torch.utils.data import Dataset
import numpy as np

class MASTDataset(Dataset):
    # f0_path is the path to the f0data directory in MAST_dataset
    def __init__(self, f0_path):
        super(MASTDataset, self).__init__()
        all_f0s = os.listdir(f0_path)
        self.data = [(a,1) for a in all_f0s if 'pass' in a] # 266 samples
        self.data.extend([(a,0) for a in all_f0s if 'fail' in a]) # 730 samples
        self.f0_path = f0_path
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        file, target = self.data[index]
        f0 = np.loadtxt(os.path.join(self.f0_path, file))[:,1]
        # downsample
        f0 = f0[0::2]
        f0 = self.normalize_pitch_contour(f0)
        return f0, target

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