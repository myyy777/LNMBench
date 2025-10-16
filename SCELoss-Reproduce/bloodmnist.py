import numpy as np
from PIL import Image
import torch.utils.data as data
from medmnist import DermaMNIST, BloodMNIST
from utils.utils import  noisify
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

        return img, label, index

    def __len__(self):
        return len(self.data)


class BloodMNIST_VAL(data.Dataset):
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

        return img, label, index

    def __len__(self):
        return len(self.data)




class BloodMNIST_TEST(data.Dataset):
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

        return img, label, index

    def __len__(self):
        return len(self.test_data)