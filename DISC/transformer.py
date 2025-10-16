import torch
import numpy as np
import torchvision.transforms as transforms

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def transform_train(dataset_name):
    if dataset_name == 'fmnist':
        transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.2860, ),(0.3530, ))
        transforms.Normalize((0.1307, ),(0.3081, )),
       ])


    elif dataset_name == 'pathmnist':
        transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                    std =[0.229, 0.224, 0.225])
        # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
   ])
    else:
        transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
   ])
    
    return transform


def transform_test(dataset_name):
    if dataset_name == 'fmnist':
        transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.2860, ),(0.3530, )),
        transforms.Normalize((0.1307, ),(0.3081, )),
       ])

    elif dataset_name == 'pathmnist' or dataset_name == 'dermamnist' or dataset_name == 'bloodmnist':
        transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std =[0.229, 0.224, 0.225])
        # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
   ])
    elif dataset_name == 'organcmnist':
         transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std =[0.229, 0.224, 0.225])
        # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
   ])

    elif dataset_name == 'chexpert':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std =[0.229, 0.224, 0.225])
    ])
    elif dataset_name == 'drtid' or dataset_name == 'kaggledr':
        transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std =[0.229, 0.224, 0.225])
   ])
    
    return transform



def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target    
