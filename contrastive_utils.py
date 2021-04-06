import torch
import torch.nn

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    
    follows form loss = (1 - Y)(0.5)(Dw)^2 + Y(0.5)(max(0, m - (Dw)^2))
    Where:
        Y = {1 if prediction_label = target_label else 0}
        Dw = Distance between 2 labels
        m = margin, using label accuracy per each batch = margin
          
    """

    def __init__(self, num_labels = 5, margin = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.num_labels = num_labels
        self.margin = margin
    
    # map each data point to a label corresponding to equal length section of the scores
    def label_map(self, x):
        label_width = float(1/self.num_labels)
        labels = torch.floor(torch.div(x, label_width))
        labels[labels == self.num_labels] -= 1
        return labels.long()

    #euclidean vector distance
    def euclid(self, x, y):
        z = x - y
        return torch.pow(torch.sum(torch.pow(z, 2), axis=1), 0.5)
    
    #cosine similarity 
    def cosine_sim(self, x, y):
        x_mag = torch.pow(torch.sum(torch.pow(x, 2), axis=1), 0.5)
        y_mag = torch.pow(torch.sum(torch.pow(y, 2), axis=1), 0.5)
        dist = torch.matmul(x, y.T)
        if len(dist.size()) > 1:
            dist = torch.diagonal(dist)
        return torch.div(dist, (x_mag * y_mag))

    def forward(self, targets1, targets2, conv_out1, conv_out2):
        targets1 = torch.squeeze(targets1)
        targets2 = torch.squeeze(targets2)

        # Use cosine similarity or euclidean in scores as distance function 
        #print(conv_out1.shape, conv_out2.shape)
        diff = self.euclid(conv_out1, conv_out2)
        dist_sq = torch.pow(diff, 2)

        #map predictions and targets to labels
        Y_targ = self.label_map(targets1)
        Y_targ2 = self.label_map(targets2)
        #compare labels and compute accuracyd
        #Y_diff = torch.eq(Y_pred, Y_targ).float()
        #acc = torch.sum(Y_diff) / Y_diff.shape[0]

        #compare labels for contrastive pairs
        Y_cont = torch.eq(Y_targ, Y_targ2).long()

        #compute loss
        loss = 0.5*(Y_cont)*dist_sq + 0.5*(1 - Y_cont)*torch.pow(torch.clamp(self.margin - diff, min=0.0), 2)
        return torch.mean(loss)