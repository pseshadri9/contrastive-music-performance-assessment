import os
import dill
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from scipy.linalg import toeplitz
from scipy.signal import resample

class AudioToSpectralRep:
    def __init__(self, rep_params):
        self.params = rep_params
    
    def extract_features(self, y, sr):
        try:
            if self.params['normalize']:
                rms = np.sqrt(np.mean(y*y))
                if rms > 1e-4:
                    y = y/rms
        except KeyError:
            pass
        try:
            if self.params['remove_silence']:
                y = self.remove_silence(y, window=32, hop=32, threshold=self.params['sil_threshold'])
        except KeyError:
            pass
        if self.params['method'] == 'FFT':
            x = np.abs(librosa.stft(y, n_fft=self.params['n_fft'], hop_length=self.params['hop_length']))
            x = librosa.logamplitude(x**2)
            #x = np.abs(librosa.stft(y,n_fft=320))

        elif self.params['method'] == 'MelSpectrogram':
            x = librosa.feature.melspectrogram(y, sr, n_fft=self.params['n_fft'], hop_length=self.params['hop_length'], n_mels=self.params['n_mels'])
            x = librosa.amplitude_to_db(x)
            #x = librosa.logamplitude(x**2)
            #x = librosa.feature.melspectrogram(y,sr,n_fft=320,n_mels=160)
            #
            print(x.shape)
            raise ValueError

        elif self.params['method'] == 'ACF':
            numZeros = self.params['hop_length'] - (len(y) % self.params['hop_length'])
            ## Use center padding
            y2 = np.insert(y, 0, np.zeros(numZeros // 2))
            y2 = np.append(y2, np.zeros(numZeros - (numZeros // 2)))

            ind = 0
            blockSize = self.params['n_fft']
            while ((ind + blockSize) <= len(y2)):
                if ind == 0:
                    x = librosa.autocorrelate(y2[ind : ind + blockSize], max_size=(self.params['hop_length']//2))
                    x = np.expand_dims(x, 0)
                else:
                    x = np.vstack((x, librosa.autocorrelate(y2[ind : ind + blockSize], max_size=(self.params['hop_length']//2))))
                ind += blockSize
            #x = x.transpose()
            x = resample(x, 96, t=None, axis=0)
            #print(x.shape)


        elif self.params['method'] == 'Cepstrum':
            x = librosa.stft(y, n_fft=self.params['n_fft'], hop_length=self.params['hop_length'])
            x = np.log(x)
            for i in range(len(x)):
                x[i] = np.absolute(np.fft.ifft(x[i]))
            x = x.real.astype('float32')
            x = resample(x, 96, t=None, axis=0)


        elif self.params['method'] == 'CQT':
            #x = librosa.hybrid_cqt(y,sr, hop_length=self.params['hop_length'], n_bins=self.params['n_bins'], bins_per_octave=self.params['bins_per_octave'])
            x = librosa.hybrid_cqt(y,sr, hop_length=128, n_bins=144, bins_per_octave=24)
            x = librosa.amplitude_to_db(x**2)

        elif self.params['method'] == 'MFCC':
            x = librosa.feature.mfcc(y, sr, n_fft=self.params['n_fft'], n_mels=self.params['n_mels'], n_mfcc = self.params['n_mfcc'], hop_length=self.params['hop_length'])
            delta = librosa.feature.delta(x)
            d_delta = librosa.feature.delta(x, order=2)
            x = np.concatenate([x, delta, d_delta], axis=0)

        return torch.FloatTensor(x)

    def remove_silence(self, y, window, hop, threshold):
        frames = librosa.util.frame(y, window, hop)
        sil_frames = []
        for i, f in enumerate(frames.transpose()):
            if np.sqrt(np.mean(f*f)) < threshold:
                sil_frames.append(i)
        y = np.delete(frames, sil_frames, 1)
        y = y.reshape(-1)
        return y

class SpectralDataset(Dataset, AudioToSpectralRep):
    """Dataset class for spectral feature based music performance assessment data"""

    def __init__(self, data_path, label_id, rep_params):
        """
        Initializes the class, defines the number of datapoints
        Args:
            data_path:  full path to the file which contains the pitch contour data
            label_id:   the label to use for training
            rep_params: parameters for spectral representation
        """
        super(SpectralDataset, self).__init__(rep_params)
        self.perf_data = dill.load(open(data_path, 'rb'))
        self.label_id = label_id
        self.length = len(self.perf_data)
            
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        y, sr = self.perf_data[idx]['audio']
        X = self.extract_features(y, sr)
        label = self.perf_data[idx]['ratings'][self.label_id]
        # For one-hot labels
        # y = np.zeros(11)
        # y[label] = 1

        # For class labels
        y = label
        return torch.unsqueeze(X, 0), y

def _collate_fn(batch):
    def func(p):
        return p[0].size(2)

    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(1)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(2)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(2)
        inputs[x].narrow(2, 0, seq_length).copy_(tensor)
        targets.append(target)
    targets = torch.Tensor(targets)
    return inputs, targets

class SpectralDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(SpectralDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn