import os
import pandas as pd
from PIL import Image
import torch.utils.data as data
import numpy as np
from PIL import Image
import torch.utils.data as data



class DRTID(data.Dataset):
    def __init__(self, 
                 transform=None, target_transform=None,
                 download=False,
                 noise_type='symmetric', noise_rate=0.2, 
                  random_seed=0, num_class=9,device=None):

        
        self.transform = transform
        self.target_transform = target_transform
        self.df = pd.read_csv("/home/user/label_noise/drtid/final_train.csv")
        self.image_root = "/home/user/label_noise/drtid/image/newall_/"


        self.random_seed = random_seed


        #  PathMNIST  train split
        #  noisy label
        self.labels = self.df['Grade'].values  # numpy array
        self.labels_clean = self.df['Grade_gt'].values
        self.image_names = self.df['image_name'].values

        # Grade == Grade_gt
        self.noise_or_not = (self.df['Grade'] == self.df['Grade_gt']).values
     



    
    def __getitem__(self, index):
        filename = self.image_names[index] + ".jpg"
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


class DRTID_val(data.Dataset):
    def __init__(self, 
                 transform=None, target_transform=None,
                 download=False,
                 noise_type='symmetric', noise_rate=0.2, 
                  random_seed=0, num_class=9,device=None):

        
        self.transform = transform
        self.target_transform = target_transform
        self.df = pd.read_csv("/home/user/label_noise/drtid/final_val.csv")
        self.image_root = "/home/user/label_noise/drtid/image/newall_/"


        self.random_seed = random_seed


        #  PathMNIST  train split
        #  noisy label
        self.labels = self.df['Grade'].values  # numpy array
        self.image_names = self.df['image_name'].values

     



    
    def __getitem__(self, index):
        filename = self.image_names[index] + ".jpg"
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




class DRTID_test(data.Dataset):
    def __init__(self, 
                 transform=None, target_transform=None,
                 download=False,
                 noise_type='symmetric', noise_rate=0.2, 
                  random_seed=0, num_class=9,device=None):

        
        self.transform = transform
        self.target_transform = target_transform
        self.df = pd.read_csv("/home/user/label_noise/drtid/final_test.csv")
        self.image_root = "/home/user/label_noise/drtid/image/newall_/"


        self.random_seed = random_seed


        #  PathMNIST  train split
        #  noisy label
        self.labels = self.df['Grade'].values  # numpy array
        self.image_names = self.df['image_name'].values



     



    
    def __getitem__(self, index):
        filename = self.image_names[index] + ".jpg"
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