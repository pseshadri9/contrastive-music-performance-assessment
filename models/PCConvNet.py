import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PCConvNet(nn.Module):
    """
    Class to implement a deep neural model for music performance assessment using
	 pitch contours as input
    """

    def __init__(self, mode):
        """
        Initializes the class with internal parameters for the different layers
        Args:
            mode:       int, 0,1 specifying different minimum input size, 0: 1000, 1:500
        """
        super(PCConvNet, self).__init__()
        if mode == 0: # for minimum input size of 1000
            # initialize model internal parameters
            self.kernel_size = 7
            self.stride = 3
            self.n0_features = 4
            self.n1_features = 8
            self.n2_features = 16
            # define the different convolutional modules
            if False:
                self.conv = nn.Sequential(
                    # define the 1st convolutional layer
                    nn.Conv1d(1, self.n0_features, self.kernel_size, self.stride),# output is (1000 - 7)/3 + 1 = 332
                    nn.BatchNorm1d(self.n0_features),
                    nn.ReLU(),
                    #nn.Dropout(),
                    # define the 2nd convolutional layer
                    nn.Conv1d(self.n0_features, self.n1_features, self.kernel_size, self.stride), # output is (332 - 7)/3 + 1 = 109
                    nn.BatchNorm1d(self.n1_features),
                    nn.ReLU(),
                    #nn.Dropout(),
                    # define the 3rd convolutional layer
                    nn.Conv1d(self.n1_features, self.n2_features, self.kernel_size, self.stride), # output is (109 - 7)/3 + 1 = 35
                    nn.BatchNorm1d(self.n2_features),
                    nn.ReLU(),
                    #nn.Dropout(),
                    # define the final fully connected layer (fully convolutional)
                    nn.Conv1d(self.n2_features, 1, 35, 1),
                    nn.BatchNorm1d(1),
                    nn.ReLU(),
                    #nn.Dropout()
                )
            else:
                self.conv = nn.Sequential(
                    # define the 1st convolutional layer
                    nn.Conv1d(1, self.n0_features, self.kernel_size, self.stride),# output is (1000 - 7)/3 + 1 = 332
                    nn.BatchNorm1d(self.n0_features),
                    nn.ReLU(),
                    #nn.Dropout(),
                    # define the 2nd convolutional layer
                    nn.Conv1d(self.n0_features, self.n1_features, self.kernel_size, self.stride), # output is (332 - 7)/3 + 1 = 109
                    nn.BatchNorm1d(self.n1_features),
                    nn.ReLU(),
                    #nn.Dropout(),
                    # define the 3rd convolutional layer
                    nn.Conv1d(self.n1_features, self.n2_features, self.kernel_size, self.stride), # output is (109 - 7)/3 + 1 = 35
                    nn.BatchNorm1d(self.n2_features),
                    nn.ReLU()
                )
                self.conv2 = nn.Sequential(
                    nn.Conv1d(self.n2_features, 1, 35, 1),
                    nn.BatchNorm1d(1),
                    nn.ReLU())
        elif mode == 1: # for minimum input size of 500
            # initialize model internal parameters
            self.kernel_size = 5
            self.stride = 2
            self.n0_features = 4
            self.n1_features = 8
            self.n2_features = 16
            # define the convolutional modelues
            self.conv = nn.Sequential(
                # define the 1st convolutional layer
                nn.Conv1d(1, self.n0_features, self.kernel_size, self.stride), # output is (500 - 5)/2 + 1 = 248
                nn.BatchNorm1d(self.n0_features),
                nn.LeakyReLU(),
                # define the 2nd convolutional layer
                nn.Conv1d(self.n0_features, self.n1_features, self.kernel_size, self.stride), # output is (248 - 5)/2 + 1 = 122
                nn.BatchNorm1d(self.n1_features),
                nn.LeakyReLU(),
                # define the 3rd convolutional layer
                nn.Conv1d(self.n1_features, self.n2_features, 7, 4), # output is (122 - 7)/4 + 1 = 29
                nn.BatchNorm1d(self.n2_features),
                nn.LeakyReLU(),
                # define the final fully connected layer (fully convolutional)
                nn.Conv1d(self.n2_features, 1, 28, 1),
                nn.BatchNorm1d(1),
                nn.ReLU()
            )
    def forward_conv(self, input, latent_test = False):
        # get mini batch size from input and reshape
        mini_batch_size, sig_size = input.size()
        input = input.view(mini_batch_size, 1, sig_size)
        # compute the forward pass through the convolutional layer
        conv_out = self.conv(input)
        if latent_test:
            final_output = torch.mean(conv_out, 2)
            #final_output = torch.unsqueeze(final_output.view(-1), dim=0)
            #print(final_output.size(), '1')
        else:
            final_output = conv_out
        return final_output

    def forward(self, input, conv_out = None):
        """
        Defines the forward pass of the PitchContourAssessor module
        Args:
                input: 	torch Variable (mini_batch_size x zero_pad_len), of input pitch contours
                		mini_batch_size: 	size of the mini batch during one training iteration
            			zero_pad_len: 		length to which each input sequence is zero-padded
                		seq_lengths:		torch tensor (mini_batch_size x 1), length of each pitch contour
        """
        if conv_out is None:
            conv_out = self.forward_conv(input)
        final_output = self.conv2(conv_out)
        # compute final output
        final_output = torch.mean(final_output, 2)
            #print(final_output.size())
        #print(final_output.size())
        # return output
        return final_output


class PCConvNetCls(nn.Module):
    """
    Class to implement a deep neural model for music performance assessment using
	 pitch contours as input for classification tasks
    """

    def __init__(self, mode, num_classes = 2, latent = False):
        """
        Initializes the class with internal parameters for the different layers
        Args:
            mode:           int, 0,1 specifying different minimum input size, 0: 1000, 1:500
            num_classes:    int, number of classes for the classification task
        """
        super(PCConvNetCls, self).__init__()
        self.num_classes = 2
        if mode == 0: # for minimum input size of 1000
            # initialize model internal parameters
            self.kernel_size = 7
            self.stride = 3
            self.n0_features = 4
            self.n1_features = 8
            self.n2_features = 16
            self.latent = latent
            # define the different convolutional modules
            if not self.latent:
                self.conv = nn.Sequential(
                    # define the 1st convolutional layer
                    nn.Conv1d(1, self.n0_features, self.kernel_size, self.stride),# output is (1000 - 7)/3 + 1 = 332
                    nn.BatchNorm1d(self.n0_features),
                    nn.ReLU(),
                    #nn.Dropout(),
                    # define the 2nd convolutional layer
                    nn.Conv1d(self.n0_features, self.n1_features, self.kernel_size, self.stride), # output is (332 - 7)/3 + 1 = 109
                    nn.BatchNorm1d(self.n1_features),
                    nn.ReLU(),
                    #nn.Dropout(),
                    # define the 3rd convolutional layer
                    nn.Conv1d(self.n1_features, self.n2_features, self.kernel_size, self.stride), # output is (109 - 7)/3 + 1 = 35
                    nn.BatchNorm1d(self.n2_features),
                    nn.ReLU(),
                    #nn.Dropout(),
                    # define the final fully connected layer (fully convolutional)
                    nn.Conv1d(self.n2_features, self.num_classes, 35, 1),
                    nn.BatchNorm1d(self.num_classes),
                    nn.ReLU(),
                    #nn.Dropout()
                )
            else:
                self.conv = nn.Sequential(
                    # define the 1st convolutional layer
                    nn.Conv1d(1, self.n0_features, self.kernel_size, self.stride), # output is (500 - 5)/2 + 1 = 248
                    nn.BatchNorm1d(self.n0_features),
                    nn.LeakyReLU(),
                    # define the 2nd convolutional layer
                    nn.Conv1d(self.n0_features, self.n1_features, self.kernel_size, self.stride), # output is (248 - 5)/2 + 1 = 122
                    nn.BatchNorm1d(self.n1_features),
                    nn.LeakyReLU(),
                    # define the 3rd convolutional layer
                    nn.Conv1d(self.n1_features, self.n2_features, 7, 4), # output is (122 - 7)/4 + 1 = 29
                    nn.BatchNorm1d(self.n2_features),
                    nn.LeakyReLU(),
                )

        elif mode == 1: # for minimum input size of 500
            # initialize model internal parameters
            self.kernel_size = 5
            self.stride = 2
            self.n0_features = 4
            self.n1_features = 8
            self.n2_features = 16
            # define the convolutional modelues
            if not self.latent:
                self.conv = nn.Sequential(
                    # define the 1st convolutional layer
                    nn.Conv1d(1, self.n0_features, self.kernel_size, self.stride), # output is (500 - 5)/2 + 1 = 248
                    nn.BatchNorm1d(self.n0_features),
                    nn.LeakyReLU(),
                    # define the 2nd convolutional layer
                    nn.Conv1d(self.n0_features, self.n1_features, self.kernel_size, self.stride), # output is (248 - 5)/2 + 1 = 122
                    nn.BatchNorm1d(self.n1_features),
                    nn.LeakyReLU(),
                    # define the 3rd convolutional layer
                    nn.Conv1d(self.n1_features, self.n2_features, 7, 4), # output is (122 - 7)/4 + 1 = 29
                    nn.BatchNorm1d(self.n2_features),
                    nn.LeakyReLU(),
                    # define the final fully connected layer (fully convolutional)
                    nn.Conv1d(self.n2_features, self.num_classes, 28, 1),
                    nn.BatchNorm1d(self.num_classes),
                    nn.ReLU()
                )
            else:
                self.conv = nn.Sequential(
                    # define the 1st convolutional layer
                    nn.Conv1d(1, self.n0_features, self.kernel_size, self.stride), # output is (500 - 5)/2 + 1 = 248
                    nn.BatchNorm1d(self.n0_features),
                    nn.LeakyReLU(),
                    # define the 2nd convolutional layer
                    nn.Conv1d(self.n0_features, self.n1_features, self.kernel_size, self.stride), # output is (248 - 5)/2 + 1 = 122
                    nn.BatchNorm1d(self.n1_features),
                    nn.LeakyReLU(),
                    # define the 3rd convolutional layer
                    nn.Conv1d(self.n1_features, self.n2_features, 7, 4), # output is (122 - 7)/4 + 1 = 29
                    nn.BatchNorm1d(self.n2_features),
                    nn.LeakyReLU(),
                )


    def forward(self, input):
        """
        Defines the forward pass of the PitchContourAssessor module
        Args:
                input: 	torch Variable (mini_batch_size x zero_pad_len), of input pitch contours
                		mini_batch_size: 	size of the mini batch during one training iteration
            			zero_pad_len: 		length to which each input sequence is zero-padded
                		seq_lengths:		torch tensor (mini_batch_size x 1), length of each pitch contour
        """
        # get mini batch size from input and reshape
        mini_batch_size, sig_size = input.size()
        input = input.view(mini_batch_size, 1, sig_size)

        # compute the forward pass through the convolutional layer
        conv_out = self.conv(input)
        # compute final output
        if not self.latent:
            final_output = torch.mean(conv_out, 2)
        # return output
        return final_output