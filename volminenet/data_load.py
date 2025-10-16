import numpy as np
from PIL import Image
import torch.utils.data as data
from medmnist import PathMNIST,DermaMNIST, BloodMNIST, OrganCMNIST
import os
import pandas as pd
from utils import  noisify

class ORGANCMNIST(data.Dataset):
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
        base_dataset = OrganCMNIST(split='train', download=False, size=224)
        original_images = base_dataset.imgs  # (N, 3, 224, 224)
        original_labels = base_dataset.labels.flatten()

        #  train / val 
        train_labels,self.t  = noisify(
            train_or_val='train',
            train_labels=original_labels,
            train_images=original_images,
            dataset='organcmnist',
            noise_type=noise_type,
            noise_rate=noise_rate,
            random_state=random_seed,
            nb_classes=self.num_class,
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


class ORGANCMNIST_VAL(data.Dataset):
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
        train_labels,P = noisify(
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
        self.train_noisy_labels = np.array(train_labels)
        self.train_labels_clean = np.array(original_labels)

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




class ORGANCMNIST_TEST(data.Dataset):
    def __init__(self,  transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform

        #  medmnist 
        base_dataset = OrganCMNIST( split='test', download=False, size=224)
        self.test_data = base_dataset.imgs
        self.test_labels = base_dataset.labels.flatten()

    def __getitem__(self, index):
        img = self.test_data[index]
        label = self.test_labels[index]

        img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)


class BLOODMNIST(data.Dataset):
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
        base_dataset = BloodMNIST(split='train', download=False, size=224)
        original_images = base_dataset.imgs  # (N, 3, 224, 224)
        original_labels = base_dataset.labels.flatten()

        #  train / val 
        train_labels,self.t  = noisify(
            train_or_val='train',
            train_labels=original_labels,
            train_images=original_images,
            dataset='bloodmnist',
            noise_type=noise_type,
            noise_rate=noise_rate,
            random_state=random_seed,
            nb_classes=self.num_class,
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


class BLOODMNIST_VAL(data.Dataset):
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
        train_labels,P  = noisify(
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
        self.train_noisy_labels = np.array(train_labels)
        self.train_labels_clean = np.array(original_labels)

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




class BLOODMNIST_TEST(data.Dataset):
    def __init__(self,  transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform

        #  medmnist 
        base_dataset = BloodMNIST( split='test', download=False, size=224)
        self.test_data = base_dataset.imgs
        self.test_labels = base_dataset.labels.flatten()

    def __getitem__(self, index):
        img = self.test_data[index]
        label = self.test_labels[index]

        img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)

class DERMAMNIST(data.Dataset):
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
        base_dataset = DermaMNIST(split='train', download=False, size=224)
        original_images = base_dataset.imgs  # (N, 3, 224, 224)
        original_labels = base_dataset.labels.flatten()

        #  train / val 
        train_labels,self.t  = noisify(
            train_or_val='train',
            train_labels=original_labels,
            train_images=original_images,
            dataset='dermamnist',
            noise_type=noise_type,
            noise_rate=noise_rate,
            random_state=random_seed,
            nb_classes=self.num_class,
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


class DERMAMNIST_VAL(data.Dataset):
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
        train_labels, P = noisify(
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
        self.train_noisy_labels = np.array(train_labels)
        self.train_labels_clean = np.array(original_labels)

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




class DERMAMNIST_TEST(data.Dataset):
    def __init__(self,  transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform

        #  medmnist 
        base_dataset = DermaMNIST( split='test', download=False, size=224)
        self.test_data = base_dataset.imgs
        self.test_labels = base_dataset.labels.flatten()

    def __getitem__(self, index):
        img = self.test_data[index]
        label = self.test_labels[index]

        img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)


class PATHMNIST(data.Dataset):
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
        base_dataset = PathMNIST(split='train', download=False, size=224)
        original_images = base_dataset.imgs  # (N, 3, 224, 224)
        original_labels = base_dataset.labels.flatten()

        #  train / val 
        train_labels,self.t = noisify(
            train_or_val='train',
            train_labels=original_labels,
            train_images=original_images,
            dataset='pathmnist',
            noise_type=noise_type,
            noise_rate=noise_rate,
            random_state=random_seed,
            nb_classes=self.num_class,
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


class PATHMNIST_VAL(data.Dataset):
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
        train_labels,P = noisify(
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
        self.train_noisy_labels = np.array(train_labels)
        self.train_labels_clean = np.array(original_labels)

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




class PATHMNIST_TEST(data.Dataset):
    def __init__(self,  transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform

        #  medmnist 
        base_dataset = PathMNIST( split='test', download=False, size=224)
        self.test_data = base_dataset.imgs
        self.test_labels = base_dataset.labels.flatten()

    def __getitem__(self, index):
        img = self.test_data[index]
        label = self.test_labels[index]

        img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)
    




    

class DRTID(data.Dataset):
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

        return image, label

    def __len__(self):
        return len(self.df)


class DRTID_val(data.Dataset):
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

        return image, label

    def __len__(self):
        return len(self.df)




class DRTID_test(data.Dataset):
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

        return image, label

    def __len__(self):
        return len(self.df)
    

class Chexpert(data.Dataset):
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

        return image, label

    def __len__(self):
        return len(self.df)


class Chexpert_val(data.Dataset):
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




class Chexpert_test(data.Dataset):
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
    

class Kaggledr(data.Dataset):
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

        return image, label

    def __len__(self):
        return len(self.df)


class Kaggledr_val(data.Dataset):
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




class Kaggledr_test(data.Dataset):
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

        return image, label

    def __len__(self):
        return len(self.df)