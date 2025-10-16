import numpy as np
import os
import numpy as np
import torch
import torchvision
from math import inf
from scipy import stats
from torchvision.transforms import transforms
import torch.nn.functional as F
import torch.nn as nn
import random



def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target  

def set_seed(seed):
    # Python 
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    # PyTorch GPU
    torch.cuda.manual_seed(seed)

    #  CuDNN 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False