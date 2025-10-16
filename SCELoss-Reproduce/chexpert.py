import os
import pandas as pd
from PIL import Image
import torch.utils.data as data
import numpy as np
from PIL import Image
import torch.utils.data as data

class chexpert(data.Dataset):
    def __init__(self, 
                 transform=None, target_transform=None,
                 download=False,
                 noise_type='symmetric', noise_rate=0.2, 
                  random_seed=0, num_class=5,device=None):

        
        self.transform = transform
        self.target_transform = target_transform
        self.df = pd.read_csv("/home/user/label_noise/Chexpert/train.csv")
        self.image_root = "/home/user/label_noise/Chexpert/"


        self.random_seed = random_seed


  
        self.labels = self.df['Label'].values  # numpy array
        self.labels_clean = self.df['Label'].values
        self.image_names = self.df['Path'].values

        # Grade == Grade_gt
        self.noise_or_not = (self.df['Label'] == self.df['Label']).values
     



    
    def __getitem__(self, index):
        filename = self.image_names[index]
        img_path = os.path.join(self.image_root, filename)

        image = Image.open(img_path).convert('RGB')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label, index

    def __len__(self):
        return len(self.df)


class chexpert_val(data.Dataset):
    def __init__(self, 
                 transform=None, target_transform=None,
                 download=False,
                 noise_type='symmetric', noise_rate=0.2, 
                  random_seed=0, num_class=5,device=None):

        
        self.transform = transform
        self.target_transform = target_transform
        self.df = pd.read_csv("/home/user/label_noise/Chexpert/val.csv")
        self.image_root = "/home/user/label_noise/Chexpert/"


        self.random_seed = random_seed


        #  PathMNIST  train split
        #  noisy label
        self.labels = self.df['Label'].values  # numpy array
        self.image_names = self.df['Path'].values

     



    
    def __getitem__(self, index):
        filename = self.image_names[index] 
        img_path = os.path.join(self.image_root, filename)

        image = Image.open(img_path).convert('RGB')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label, index

    def __len__(self):
        return len(self.df)




class chexpert_test(data.Dataset):
    def __init__(self, 
                 transform=None, target_transform=None,
                 download=False,
                 noise_type='symmetric', noise_rate=0.2, 
                  random_seed=0, num_class=5,device=None):

        
        self.transform = transform
        self.target_transform = target_transform
        self.df = pd.read_csv("/home/user/label_noise/Chexpert/test.csv")
        self.image_root = "/home/user/label_noise/Chexpert/"


        self.random_seed = random_seed


        #  PathMNIST  train split
        #  noisy label
        self.labels = self.df['Label'].values  # numpy array
        self.image_names = self.df['Path'].values



     



    
    def __getitem__(self, index):
        filename = self.image_names[index]
        img_path = os.path.join(self.image_root, filename)

        image = Image.open(img_path).convert('RGB')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label, index

    def __len__(self):
        return len(self.df)