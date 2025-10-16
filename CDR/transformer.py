import torch
import numpy as np
import torchvision.transforms as transforms

def transform_train(dataset_name):
    
    if dataset_name == 'pathmnist' or dataset_name == 'dermamnist' or dataset_name == 'bloodmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std =[0.229, 0.224, 0.225])
        ])
    if dataset_name == 'drtid' or dataset_name == 'kaggledr':
        transform = transforms.Compose([
            transforms.Resize((520, 520)),  # 
            transforms.RandomResizedCrop(size=512, scale=(0.95, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),  # 
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # ±10%
            transforms.RandomRotation(degrees=5),  #  ±5°
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std =[0.229, 0.224, 0.225])
        ])
        
    if dataset_name == 'chexpert':
           
        transform = transforms.Compose([
            transforms.Resize((230, 230)),  # 
            transforms.RandomResizedCrop(size=224, scale=(0.95, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),  # 
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # ±10%
            transforms.RandomRotation(degrees=5),  #  ±5°
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std =[0.229, 0.224, 0.225])
        ])

    return transform


def transform_test(dataset_name):
    
    if dataset_name == 'pathmnist' or dataset_name == 'dermamnist' or dataset_name == 'bloodmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std =[0.229, 0.224, 0.225])
        ])
        
    if dataset_name == 'drtid' or dataset_name == 'kaggledr':
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std =[0.229, 0.224, 0.225])
        ])
    if dataset_name == 'chexpert':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std =[0.229, 0.224, 0.225])
        ])
    
    
    return transform


def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target    
