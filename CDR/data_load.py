import numpy as np
import torch.utils.data as Data
from PIL import Image
import tools
import torch
from random import choice
import random 
from torch.utils.data import Dataset
from medmnist import PathMNIST, DermaMNIST, BloodMNIST,OrganCMNIST
from utils import noisify
import os
import pandas as pd


class organcmnist_dataset(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None,
                  noise_type='symmetric', noise_rate=0.5,
                random_seed=0, num_class=8,device=None):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        
        base_dataset = OrganCMNIST(split='train', download=False, size=224)
        original_images = base_dataset.imgs  
        original_labels = base_dataset.labels.flatten()


        train_labels = noisify(
                    train_or_val='train',
                    train_labels=original_labels,
                    train_images=original_images,
                    dataset='organcmnist',
                    noise_type=noise_type,
                    noise_rate=noise_rate,
                    random_state=random_seed,
                    nb_classes=num_class,
                    device=device
                )
        self.train_data = original_images
        self.train_labels = train_labels
    def __getitem__(self, index):
        
        img, label = self.train_data[index], self.train_labels[index]
     

        img = Image.fromarray(img.astype(np.uint8))  
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.train_data)
 


class OrganCMNIST_VAL(Dataset):
    def __init__(self, 
                 transform=None, target_transform=None,
                 download=False,
                 noise_type='symmetric', noise_rate=0.2, 
                  random_seed=0, num_class=8,device=None):

        
        self.transform = transform
        self.target_transform = target_transform
        self.noise_type = noise_type
        self.noise_rate = noise_rate

        self.random_seed = random_seed
        self.num_class = num_class

        #  PathMNIST  train split
        base_dataset = OrganCMNIST(split='val', download=False, size=224)
        original_images = base_dataset.imgs  # (N, 3, 224, 224)
        original_labels = base_dataset.labels.flatten()

        #  train / val 
        train_labels = noisify(
            train_or_val='val',
            train_labels=original_labels,
            train_images=original_images,
            dataset='organcmnist',
            noise_type=noise_type,
            noise_rate=noise_rate,
            random_state=random_seed,
            nb_classes=num_class,
            device=device
        )


        self.data = original_images
        self.labels = train_labels
    
    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.data)


class organcmnist_test_dataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        base_dataset = OrganCMNIST(split='test', download=False, size=224)
        self.test_data = base_dataset.imgs # to (N, H, W, C)
        self.test_labels = base_dataset.labels.flatten()

    def __getitem__(self, index):
        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.test_data)


class chexpert(Data.Dataset):
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


        #  PathMNIST  train split
        #  noisy label
        self.labels = self.df['Label'].values  # numpy array
        self.labels_clean = self.df['Label'].values
        self.image_names = self.df['Path'].values

        # Grade == Grade_gt
        self.t = self.labels_clean
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


class chexpert_val(Data.Dataset):
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

        return image, label,index

    def __len__(self):
        return len(self.df)




class chexpert_test(Data.Dataset):
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

        return image, label,index

    def __len__(self):
        return len(self.df)

class kaggledr(Data.Dataset):
    def __init__(self, 
                 transform=None, target_transform=None,
                 download=False,
                 noise_type='symmetric', noise_rate=0.2, 
                  random_seed=0, num_class=5,device=None):

        
        self.transform = transform
        self.target_transform = target_transform
        self.df = pd.read_csv("/home/user/label_noise/Kaggle_DR/kaggle_train.csv")
        self.image_root = "/home/user/label_noise/Kaggle_DR/diabetic-retinopathy-detection/image/"


        self.random_seed = random_seed


        #  PathMNIST  train split
        #  noisy label
        self.labels = self.df['level'].values  # numpy array
        self.labels_clean = self.df['vote'].values
        self.image_names = self.df['image'].values

        # Grade == Grade_gt
        self.t = self.labels_clean
        self.noise_or_not = (self.df['level'] == self.df['vote']).values
     



    
    def __getitem__(self, index):
        filename = self.image_names[index] + ".jpeg"
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


class kaggledr_val(Data.Dataset):
    def __init__(self, 
                 transform=None, target_transform=None,
                 download=False,
                 noise_type='symmetric', noise_rate=0.2, 
                  random_seed=0, num_class=5,device=None):

        
        self.transform = transform
        self.target_transform = target_transform
        self.df = pd.read_csv("/home/user/label_noise/Kaggle_DR/kaggle_val.csv")
        self.image_root = "/home/user/label_noise/Kaggle_DR/diabetic-retinopathy-detection/image/"



        self.random_seed = random_seed


        #  PathMNIST  train split
        #  noisy label
        self.labels = self.df['level'].values  # numpy array
        self.image_names = self.df['image'].values

     



    
    def __getitem__(self, index):
        filename = self.image_names[index] + ".jpeg"
        img_path = os.path.join(self.image_root, filename)

        image = Image.open(img_path).convert('RGB')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label,index

    def __len__(self):
        return len(self.df)




class kaggledr_test(Data.Dataset):
    def __init__(self, 
                 transform=None, target_transform=None,
                 download=False,
                 noise_type='symmetric', noise_rate=0.2, 
                  random_seed=0, num_class=5,device=None):

        
        self.transform = transform
        self.target_transform = target_transform
        self.df = pd.read_csv("/home/user/label_noise/Kaggle_DR/golden.csv")
        self.image_root = "/home/user/label_noise/Kaggle_DR/diabetic-retinopathy-detection/image/"


        self.random_seed = random_seed


        #  PathMNIST  train split
        #  noisy label
        self.labels = self.df['level'].values  # numpy array
        self.image_names = self.df['image'].values



    
    def __getitem__(self, index):
        filename = self.image_names[index] + ".jpeg"
        img_path = os.path.join(self.image_root, filename)

        image = Image.open(img_path).convert('RGB')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label,index

    def __len__(self):
        return len(self.df)


class DRTID(Data.Dataset):
    def __init__(self, 
                 transform=None, target_transform=None,
                 download=False,
                 noise_type='symmetric', noise_rate=0.2, 
                  random_seed=0, num_class=5,device=None):

        
        self.transform = transform
        self.target_transform = target_transform
        self.df = pd.read_csv("/home/user/label_noise/drtid/final_train.csv")
        self.image_root = "/home/user/label_noise/drtid/image/newall_/"


        self.random_seed = random_seed


 
        self.labels = self.df['Grade'].values  # numpy array
        self.labels_clean = self.df['Grade_gt'].values
        self.image_names = self.df['image_name'].values

      
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

        return image, label,index

    def __len__(self):
        return len(self.df)


class DRTID_val(Data.Dataset):
    def __init__(self, 
                 transform=None, target_transform=None,
                 download=False,
                 noise_type='symmetric', noise_rate=0.2, 
                  random_seed=0, num_class=5,device=None):

        
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

        return image, label,index

    def __len__(self):
        return len(self.df)




class DRTID_test(Data.Dataset):
    def __init__(self, 
                 transform=None, target_transform=None,
                 download=False,
                 noise_type='symmetric', noise_rate=0.2, 
                  random_seed=0, num_class=5,device=None):

        
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

        return image, label,index

    def __len__(self):
        return len(self.df)





class dermamnist_dataset(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None,
                  noise_type='symmetric', noise_rate=0.5,
                random_seed=0, num_class=7,device=None):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        
        base_dataset = DermaMNIST(split='train', download=False, size=224)
        original_images = base_dataset.imgs  
        original_labels = base_dataset.labels.flatten()


        train_labels = noisify(
                    train_or_val='train',
                    train_labels=original_labels,
                    train_images=original_images,
                    dataset='dermamnist',
                    noise_type=noise_type,
                    noise_rate=noise_rate,
                    random_state=random_seed,
                    nb_classes=num_class,
                    device=device
                )
        self.train_data = original_images
        self.train_labels = train_labels
    def __getitem__(self, index):
        
        img, label = self.train_data[index], self.train_labels[index]
     

        img = Image.fromarray(img.astype(np.uint8))  
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.train_data)
 


class DERMAMNIST_VAL(Dataset):
    def __init__(self, 
                 transform=None, target_transform=None,
                 download=False,
                 noise_type='symmetric', noise_rate=0.2, 
                  random_seed=0, num_class=7,device=None):

        
        self.transform = transform
        self.target_transform = target_transform
        self.noise_type = noise_type
        self.noise_rate = noise_rate

        self.random_seed = random_seed
        self.num_class = num_class

        #  PathMNIST  train split
        base_dataset = DermaMNIST(split='val', download=False, size=224)
        original_images = base_dataset.imgs  # (N, 3, 224, 224)
        original_labels = base_dataset.labels.flatten()

        #  train / val 
        train_labels = noisify(
            train_or_val='val',
            train_labels=original_labels,
            train_images=original_images,
            dataset='dermamnist',
            noise_type=noise_type,
            noise_rate=noise_rate,
            random_state=random_seed,
            nb_classes=num_class,
            device=device
        )


        self.data = original_images
        self.labels = train_labels
    
    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.data)


class dermamnist_test_dataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        base_dataset = DermaMNIST(split='test', download=False, size=224)
        self.test_data = base_dataset.imgs # to (N, H, W, C)
        self.test_labels = base_dataset.labels.flatten()

    def __getitem__(self, index):
        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.test_data)


class bloodmnist_dataset(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None,
                  noise_type='symmetric', noise_rate=0.5,
                random_seed=0, num_class=8,device=None):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        
        base_dataset = BloodMNIST(split='train', download=False, size=224)
        original_images = base_dataset.imgs  
        original_labels = base_dataset.labels.flatten()


        train_labels = noisify(
                    train_or_val='train',
                    train_labels=original_labels,
                    train_images=original_images,
                    dataset='bloodmnist',
                    noise_type=noise_type,
                    noise_rate=noise_rate,
                    random_state=random_seed,
                    nb_classes=num_class,
                    device=device
                )
        self.train_data = original_images
        self.train_labels = train_labels
    def __getitem__(self, index):
        
        img, label = self.train_data[index], self.train_labels[index]
     

        img = Image.fromarray(img.astype(np.uint8))  
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.train_data)
 


class BloodMNIST_VAL(Dataset):
    def __init__(self, 
                 transform=None, target_transform=None,
                 download=False,
                 noise_type='symmetric', noise_rate=0.2, 
                  random_seed=0, num_class=8,device=None):

        
        self.transform = transform
        self.target_transform = target_transform
        self.noise_type = noise_type
        self.noise_rate = noise_rate

        self.random_seed = random_seed
        self.num_class = num_class

        #  PathMNIST  train split
        base_dataset = BloodMNIST(split='val', download=False, size=224)
        original_images = base_dataset.imgs  # (N, 3, 224, 224)
        original_labels = base_dataset.labels.flatten()

        #  train / val 
        train_labels = noisify(
            train_or_val='val',
            train_labels=original_labels,
            train_images=original_images,
            dataset='bloodmnist',
            noise_type=noise_type,
            noise_rate=noise_rate,
            random_state=random_seed,
            nb_classes=num_class,
            device=device
        )


        self.data = original_images
        self.labels = train_labels
    
    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.data)


class bloodmnist_test_dataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        base_dataset = BloodMNIST(split='test', download=False, size=224)
        self.test_data = base_dataset.imgs # to (N, H, W, C)
        self.test_labels = base_dataset.labels.flatten()

    def __getitem__(self, index):
        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.test_data)


class pathmnist_dataset(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None,
                  noise_type='symmetric', noise_rate=0.5,
                random_seed=0, num_class=9,device=None):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        
        base_dataset = PathMNIST(split='train', download=False, size=224)
        original_images = base_dataset.imgs  
        original_labels = base_dataset.labels.flatten()


        train_labels = noisify(
                    train_or_val='train',
                    train_labels=original_labels,
                    train_images=original_images,
                    dataset='pathmnist',
                    noise_type=noise_type,
                    noise_rate=noise_rate,
                    random_state=random_seed,
                    nb_classes=num_class,
                    device=device
                )
        self.train_data = original_images
        self.train_labels = train_labels
    def __getitem__(self, index):
        
        img, label = self.train_data[index], self.train_labels[index]
     

        img = Image.fromarray(img.astype(np.uint8))  
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.train_data)
 


class PATHMNIST_VAL(Dataset):
    def __init__(self, 
                 transform=None, target_transform=None,
                 download=False,
                 noise_type='symmetric', noise_rate=0.2, 
                  random_seed=0, num_class=9,device=None):

        
        self.transform = transform
        self.target_transform = target_transform
        self.noise_type = noise_type
        self.noise_rate = noise_rate

        self.random_seed = random_seed
        self.num_class = num_class

        #  PathMNIST  train split
        base_dataset = PathMNIST(split='val', download=False, size=224)
        original_images = base_dataset.imgs  # (N, 3, 224, 224)
        original_labels = base_dataset.labels.flatten()

        #  train / val 
        train_labels = noisify(
            train_or_val='val',
            train_labels=original_labels,
            train_images=original_images,
            dataset='pathmnist',
            noise_type=noise_type,
            noise_rate=noise_rate,
            random_state=random_seed,
            nb_classes=num_class,
            device=device
        )


        self.data = original_images
        self.labels = train_labels
    
    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.data)


class pathmnist_test_dataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        base_dataset = PathMNIST(split='test', download=False, size=224)
        self.test_data = base_dataset.imgs # to (N, H, W, C)
        self.test_labels = base_dataset.labels.flatten()

    def __getitem__(self, index):
        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.test_data)



