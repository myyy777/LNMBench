import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    if torch.is_tensor(ind):
        ind = ind.cpu().numpy()
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    loss_2 = F.cross_entropy(y_2, t, reduction='none')

    # Sort and select smallest losses (remembered samples)
    ind_1_sorted = torch.argsort(loss_1).cpu().numpy()
    ind_2_sorted = torch.argsort(loss_2).cpu().numpy()

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1))

    # Select indices of clean data based on sorted loss
    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    # Compute pure ratio
    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_update]]) / float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_update]]) / float(num_remember)

    # Convert to tensors and move to device
    ind_1_update = torch.tensor(ind_1_update, dtype=torch.long).to(y_1.device)
    ind_2_update = torch.tensor(ind_2_update, dtype=torch.long).to(y_2.device)

    # Cross loss on exchanged indices
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    # return loss_1_update, loss_2_update, pure_ratio_1, pure_ratio_2
    return loss_1_update, loss_2_update, ind_1_update, ind_2_update


