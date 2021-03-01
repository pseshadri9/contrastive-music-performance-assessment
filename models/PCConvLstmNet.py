import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PCConvLstmNet(nn.Module):
    """
    Class to implement a deep neural model for music performance assessment using
	 pitch contours as input
    """

    def __init__(self):
        """
        Initializes the PitchContourAssessor class with internal parameters for the different layers
        """
        super(PCConvLstmNet, self).__init__()
        # initialize interal parameters
        self.kernel_size = 7
        #self.kernel_size = 15
        self.stride = 3
        self.hidden_size = 16
        self.n_layers = 1
        self.n0_features = 4
        self.n1_features = 8
        self.n2_features = 16
        # define the different convolutional modules
        self.conv = nn.Sequential(
            # define the 1st convolutional layer
            nn.Conv1d(1, self.n0_features, self.kernel_size, self.stride),
            nn.BatchNorm1d(self.n0_features),
            nn.ReLU(),
            #nn.Dropout(),
            # define the 2nd convolutional layer
            nn.Conv1d(self.n0_features, self.n1_features, self.kernel_size, self.stride),
            nn.BatchNorm1d(self.n1_features),
            nn.ReLU(),
            #nn.Dropout(),
            # define the 3rd convolutional layer
            nn.Conv1d(self.n1_features, self.n2_features, self.kernel_size, self.stride),
            nn.BatchNorm1d(self.n2_features),
            nn.ReLU(),
            #nn.Dropout()
        )
        # define the LSTM module
        self.lstm = nn.GRU(self.n2_features, self.hidden_size,
                            self.n_layers, batch_first=True)
        # intialize the hidden state
        self.init_hidden(1)
        # define the final linear layer
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, input):
        """
        Defines the forward pass of the PitchContourAssessor module
        Args:
                input: 	torch Variable (mini_batch_size x zero_pad_len), of input pitch contours
                		mini_batch_size: 	size of the mini batch during one training iteration
            			zero_pad_len: 		length to which each input sequence is zero-padded
                		seq_lengths:		torch tensor (mini_batch_size x 1), length of each pitch contour
        """
        # get mini batch size from input and reshape input
        mini_batch_size, zero_pad_len = input.size()
        input = input.view(mini_batch_size, 1, zero_pad_len)
        # compute the output of the convolutional layer
        conv_out = self.conv(input)
        # compute the output of the lstm layer
        # transpose to ensure sequence length is dim 1 now
        lstm_out, self.hidden = self.lstm(conv_out.transpose(1, 2))
        # extract final output of the lstm layer
        mini_batch_size, lstm_seq_len, num_features = lstm_out.size()
        final_lstm_out = lstm_out[:, lstm_seq_len - 1, :]
        # compute output of the linear layer
        final_output = F.relu(self.linear(final_lstm_out))

        # return output
        return final_output

    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the PitchContourAssessor module
        Args:
                mini_batch_size: 	number of data samples in the mini-batch
        """
        self.hidden = Variable(torch.zeros(self.n_layers, mini_batch_size, self.hidden_size))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()



class PCConvLstmNetCls(nn.Module):
    """
    Class to implement a deep neural model for music performance assessment using
	 pitch contours as input for classification tasks
    """

    def __init__(self, num_classes = 2):
        """
        Initializes the PitchContourAssessor class with internal parameters for the different layers
        Args:
            num_classes:    int, number of classes
        """
        super(PCConvLstmNetCls, self).__init__()
        # initialize interal parameters
        self.num_classes = num_classes
        self.kernel_size = 7
        #self.kernel_size = 15
        self.stride = 3
        self.hidden_size = 16
        self.n_layers = 1
        self.n0_features = 4
        self.n1_features = 8
        self.n2_features = 16
        # define the different convolutional modules
        self.conv = nn.Sequential(
            # define the 1st convolutional layer
            nn.Conv1d(1, self.n0_features, self.kernel_size, self.stride),
            nn.BatchNorm1d(self.n0_features),
            nn.ReLU(),
            #nn.Dropout(),
            # define the 2nd convolutional layer
            nn.Conv1d(self.n0_features, self.n1_features, self.kernel_size, self.stride),
            nn.BatchNorm1d(self.n1_features),
            nn.ReLU(),
            #nn.Dropout(),
            # define the 3rd convolutional layer
            nn.Conv1d(self.n1_features, self.n2_features, self.kernel_size, self.stride),
            nn.BatchNorm1d(self.n2_features),
            nn.ReLU(),
            #nn.Dropout()
        )
        # define the LSTM module
        self.lstm = nn.GRU(self.n2_features, self.hidden_size,
                            self.n_layers, batch_first=True)
        # intialize the hidden state
        self.init_hidden(1)
        # define the final linear layer
        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, input):
        """
        Defines the forward pass of the PitchContourAssessor module
        Args:
                input: 	torch Variable (mini_batch_size x zero_pad_len), of input pitch contours
                		mini_batch_size: 	size of the mini batch during one training iteration
            			zero_pad_len: 		length to which each input sequence is zero-padded
                		seq_lengths:		torch tensor (mini_batch_size x 1), length of each pitch contour
        """
        # get mini batch size from input and reshape input
        mini_batch_size, zero_pad_len = input.size()
        input = input.view(mini_batch_size, 1, zero_pad_len)
        # compute the output of the convolutional layer
        conv_out = self.conv(input)
        # compute the output of the lstm layer
        # transpose to ensure sequence length is dim 1 now
        lstm_out, self.hidden = self.lstm(conv_out.transpose(1, 2))
        # extract final output of the lstm layer
        mini_batch_size, lstm_seq_len, num_features = lstm_out.size()
        final_lstm_out = lstm_out[:, lstm_seq_len - 1, :]
        # compute output of the linear layer
        final_output = F.relu(self.linear(final_lstm_out))

        # return output
        return final_output

    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the PitchContourAssessor module
        Args:
                mini_batch_size: 	number of data samples in the mini-batch
        """
        self.hidden = Variable(torch.zeros(self.n_layers, mini_batch_size, self.hidden_size))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()