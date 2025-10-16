import numpy as np
import torch.utils.data as Data
from PIL import Image
from transformer import *
import tools, pdb
from collections import Counter
from collections import defaultdict
from medmnist import PathMNIST,DermaMNIST, BloodMNIST, OrganCMNIST
from datasets.randaugment import RandAugmentwogeo
from torch.utils.data import Dataset
from collections import defaultdict
from PIL import Image
import numpy as np
import torch
import os
import pandas as pd
from utils_ import  noisify
import torch.utils.data as data

class organcmnist_dataset(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None,
                 noise_rate=0.5, split_per=0.9, random_seed=0, num_class=11,
                 noise_type='instance',  device = None):
         
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
       
        device = device
        base_dataset = OrganCMNIST(split='train', download=False, size=224)
        original_images = base_dataset.imgs  # (N, 3, 224, 224)
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
        self.train_labels = train_labels
        self.t = original_labels
        self.train_data = original_images



    def __getitem__(self, index):
        img, label = self.train_data[index], self.train_labels[index]
        true = self.t[index]
        
            


        img = Image.fromarray(img.astype(np.uint8))

        if self.transform:
            w, s = self.transform(img)
       
        if self.target_transform:
            label = self.target_transform(label)
            true = self.target_transform(true)

       
        return (w,s), label, index
    

        

    def __len__(self):
        return len(self.train_data)


class organcmnist_val_dataset(Dataset):
    def __init__(self, 
                 transform=None, target_transform=None,
                 download=False,
                 noise_type='symmetric', noise_rate=0.2, 
                  random_seed=0, num_class=11,device=None):

        
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
        # self.train_noisy_labels = np.array([i[0] for i in train_labels])
        # _train_labels = np.array([i[0] for i in train_clean_labels])
        self.train_noisy_labels = train_labels
        self.train_labels_clean = original_labels

        # 
        self.noise_or_not = self.train_noisy_labels == self.train_labels_clean

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

        return img, label

    def __len__(self):
        return len(self.data)

class organcmnist_test_dataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        base_dataset = OrganCMNIST( split='test', download=False, size=224)
        self.test_data = base_dataset.imgs
        self.test_labels = base_dataset.labels.flatten()
        


    def __getitem__(self, index):
        img, label = self.test_data[index], self.test_labels[index]
        img = Image.fromarray(img.astype(np.uint8))

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)


class bloodmnist_dataset(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None,
                 noise_rate=0.5, split_per=0.9, random_seed=0, num_class=8,
                 noise_type='instance',  device = None):
         
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
       
        device = device
        base_dataset = BloodMNIST(split='train', download=False, size=224)
        original_images = base_dataset.imgs  # (N, 3, 224, 224)
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
        self.train_labels = train_labels
        self.t = original_labels
        self.train_data = original_images



    def __getitem__(self, index):
        img, label = self.train_data[index], self.train_labels[index]
        true = self.t[index]
        
            


        img = Image.fromarray(img.astype(np.uint8))

        if self.transform:
            w, s = self.transform(img)
       
        if self.target_transform:
            label = self.target_transform(label)
            true = self.target_transform(true)

       
        return (w,s), label, index
    

        

    def __len__(self):
        return len(self.train_data)


class bloodmnist_val_dataset(Dataset):
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
        # self.train_noisy_labels = np.array([i[0] for i in train_labels])
        # _train_labels = np.array([i[0] for i in train_clean_labels])
        self.train_noisy_labels = train_labels
        self.train_labels_clean = original_labels

        # 
        self.noise_or_not = self.train_noisy_labels == self.train_labels_clean

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

        return img, label

    def __len__(self):
        return len(self.data)

class bloodmnist_test_dataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        base_dataset = BloodMNIST( split='test', download=False, size=224)
        self.test_data = base_dataset.imgs
        self.test_labels = base_dataset.labels.flatten()
        


    def __getitem__(self, index):
        img, label = self.test_data[index], self.test_labels[index]
        img = Image.fromarray(img.astype(np.uint8))

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)





class dermamnist_dataset(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None,
                 noise_rate=0.5, split_per=0.9, random_seed=0, num_class=7,
                 noise_type='instance',  device = None):
         
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
       
        device = device
        base_dataset = DermaMNIST(split='train', download=False, size=224)
        original_images = base_dataset.imgs  # (N, 3, 224, 224)
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
        self.train_labels = train_labels
        self.t = original_labels
        self.train_data = original_images



    def __getitem__(self, index):
        img, label = self.train_data[index], self.train_labels[index]
        true = self.t[index]
        
            


        img = Image.fromarray(img.astype(np.uint8))

        if self.transform:
            w, s = self.transform(img)
       
        if self.target_transform:
            label = self.target_transform(label)
            true = self.target_transform(true)

       
        return (w,s), label, index
    

        

    def __len__(self):
        return len(self.train_data)


class dermamnist_val_dataset(Dataset):
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
        # self.train_noisy_labels = np.array([i[0] for i in train_labels])
        # _train_labels = np.array([i[0] for i in train_clean_labels])
        self.train_noisy_labels = train_labels
        self.train_labels_clean = original_labels

        # 
        self.noise_or_not = self.train_noisy_labels == self.train_labels_clean

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

        return img, label

    def __len__(self):
        return len(self.data)

class dermamnist_test_dataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        base_dataset = DermaMNIST( split='test', download=False, size=224)
        self.test_data = base_dataset.imgs
        self.test_labels = base_dataset.labels.flatten()
        
        # if self.test_data.shape[1] == 3:  # (N, 3, 28, 28)
        #     self.test_data = self.test_data.transpose((0, 2, 3, 1))  # to (N, 28, 28, 3)

    def __getitem__(self, index):
        img, label = self.test_data[index], self.test_labels[index]
        img = Image.fromarray(img.astype(np.uint8))

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)







class pathmnist_dataset(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None,
                 noise_rate=0.5, split_per=0.9, random_seed=0, num_class=9,
                 noise_type='instance',  device = None):
         
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
       
        device = device
        base_dataset = PathMNIST(split='train', download=False, size=224)
        original_images = base_dataset.imgs  # (N, 3, 224, 224)
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
        self.train_labels = train_labels
        self.t = original_labels
        self.train_data = original_images



    def __getitem__(self, index):
        img, label = self.train_data[index], self.train_labels[index]
        true = self.t[index]
        
            


        img = Image.fromarray(img.astype(np.uint8))

        if self.transform:
            w, s = self.transform(img)
       
        if self.target_transform:
            label = self.target_transform(label)
            true = self.target_transform(true)

       
        return (w,s), label, index
    

        

    def __len__(self):
        return len(self.train_data)


class pathmnist_val_dataset(Dataset):
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
        # self.train_noisy_labels = np.array([i[0] for i in train_labels])
        # _train_labels = np.array([i[0] for i in train_clean_labels])
        self.train_noisy_labels = train_labels
        self.train_labels_clean = original_labels

        # 
        self.noise_or_not = self.train_noisy_labels == self.train_labels_clean

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

        return img, label

    def __len__(self):
        return len(self.data)

class pathmnist_test_dataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        base_dataset = PathMNIST( split='test', download=False, size=224)
        self.test_data = base_dataset.imgs
        self.test_labels = base_dataset.labels.flatten()
        
        # if self.test_data.shape[1] == 3:  # (N, 3, 28, 28)
        #     self.test_data = self.test_data.transpose((0, 2, 3, 1))  # to (N, 28, 28, 3)

    def __getitem__(self, index):
        img, label = self.test_data[index], self.test_labels[index]
        img = Image.fromarray(img.astype(np.uint8))

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)







class drtid_dataset(data.Dataset):
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
        self.t = self.labels_clean
        self.noise_or_not = (self.df['Grade'] == self.df['Grade_gt']).values
     



    
    def __getitem__(self, index):
        filename = self.image_names[index] + ".jpg"
        img_path = os.path.join(self.image_root, filename)

        image = Image.open(img_path).convert('RGB')
        label = self.labels[index]

        if self.transform:
                   
            w, s = self.transform(image)
       
 
        if self.target_transform:
            label = self.target_transform(label)

        return (w,s), label, index

    def __len__(self):
        return len(self.df)


class drtid_val_dataset(data.Dataset):
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

        return image, label

    def __len__(self):
        return len(self.df)




class drtid_test_dataset(data.Dataset):
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

        return image, label

    def __len__(self):
        return len(self.df)










class TransformFixMatch_PathMNIST:
    def __init__(self, size=224):
        #  resize&crop + 
        self.weak = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
        ])
        #  RandAugment
        self.strong = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            RandAugmentwogeo(n=2, m=5)
        ])
        #  ToTensor + Normalize
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std =[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        """
        img: PIL.Image 或者 HWC numpy array
        返回: (weak_tensor, strong_tensor)，两者 shape 都是 [3, size, size]
        """
        w = self.weak(img)
        s = self.strong(img)
        return self.normalize(w), self.normalize(s)
    

    

    


class kaggledr(data.Dataset):
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
                   
            w, s = self.transform(image)
       
 
        if self.target_transform:
            label = self.target_transform(label)

        return (w,s), label, index

    def __len__(self):
        return len(self.df)


class kaggledr_val(data.Dataset):
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

        return image, label

    def __len__(self):
        return len(self.df)




class kaggledr_test(data.Dataset):
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

        return image, label

    def __len__(self):
        return len(self.df)




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
                   
            w, s = self.transform(image)
       
 
        if self.target_transform:
            label = self.target_transform(label)

        return (w,s), label, index

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

        return image, label

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

        return image, label

    def __len__(self):
        return len(self.df)







class TransformFixMatch_PathMNIST:
    def __init__(self, size=224):
        #  resize&crop + 
        self.weak = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
        ])
        #  RandAugment
        self.strong = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            RandAugmentwogeo(n=2, m=5)
        ])
        #  ToTensor + Normalize
        self.normalize = transforms.Compose([
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std =[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        """
        img: PIL.Image 或者 HWC numpy array
        返回: (weak_tensor, strong_tensor)，两者 shape 都是 [3, size, size]
        """
        w = self.weak(img)
        s = self.strong(img)
        return self.normalize(w), self.normalize(s)
    

    

class TransformFixMatch_DRTID:
    def __init__(self, size=512):
        #  resize&crop + 
        self.weak = transforms.Compose([
            transforms.Resize((520, 520)),  # 
            transforms.RandomResizedCrop(size=512, scale=(0.95, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),  # 
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # ±10%
            transforms.RandomRotation(degrees=5),  #  ±5°
        ])
        #  RandAugment
        self.strong = transforms.Compose([
            transforms.Resize((520, 520)),  # 
            transforms.RandomResizedCrop(size=512, scale=(0.95, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),  # 
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # ±10%
            transforms.RandomRotation(degrees=5),  #  ±5°
            RandAugmentwogeo(n=2, m=5)
        ])
        #  ToTensor + Normalize
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std =[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        """
        img: PIL.Image 或者 HWC numpy array
        返回: (weak_tensor, strong_tensor)，两者 shape 都是 [3, size, size]
        """
        w = self.weak(img)
        s = self.strong(img)
        return self.normalize(w), self.normalize(s)
    
    
class TransformFixMatch_chexpert:
    def __init__(self, size=224):
        #  resize&crop + 
        self.weak = transforms.Compose([
            transforms.Resize((230, 230)),  # 
            transforms.RandomResizedCrop(size=224, scale=(0.95, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),  # 
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # ±10%
            transforms.RandomRotation(degrees=5),  #  ±5°
        ])
        #  RandAugment
        self.strong = transforms.Compose([
            transforms.Resize((230, 230)),  # 
            transforms.RandomResizedCrop(size=224, scale=(0.95, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),  # 
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # ±10%
            transforms.RandomRotation(degrees=5),  #  ±5°
            RandAugmentwogeo(n=2, m=5)
        ])
        #  ToTensor + Normalize
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std =[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        """
        img: PIL.Image 或者 HWC numpy array
        返回: (weak_tensor, strong_tensor)，两者 shape 都是 [3, size, size]
        """
        w = self.weak(img)
        s = self.strong(img)
        return self.normalize(w), self.normalize(s)