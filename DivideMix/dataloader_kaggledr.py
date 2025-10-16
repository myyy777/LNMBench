from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
from medmnist import PathMNIST 
from utils import  noisify  
from collections import defaultdict
import pandas as pd

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class kaggledr_dataset(Dataset): 
    def __init__(self, dataset, noise_rate, noise_mode, transform, mode, random_seed=0,num_class=5,device=None, pred=[], probability=[], log=''): 
        
        self.r = noise_rate # noise ratio
        self.transform = transform
        self.mode = mode  
        self.image_root = "/home/user/label_noise/Kaggle_DR/diabetic-retinopathy-detection/image/" 
     
        if self.mode=='test':
            if dataset=='kaggledr': 
                df = pd.read_csv("/home/user/label_noise/Kaggle_DR/golden.csv")
                
                
                self.test_data = df['image'].values
                self.test_labels = df['level'].values
            
        elif self.mode=='val':    
            train_data=[]
            train_label=[]
            if dataset=='kaggledr': 
                df = pd.read_csv("/home/user/label_noise/Kaggle_DR/kaggle_val.csv")
            self.val_data = df['image'].values
            self.val_labels = df['level'].values
        else:     
            train_data=[]
            train_label=[]
            if dataset=='kaggledr': 
                df = pd.read_csv("/home/user/label_noise/Kaggle_DR/kaggle_train.csv")
                train_labels = df['level'].values
                clean_labels = df['vote'].values
                train_data = df['image'].values


              
   

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = train_labels
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    clean = (np.array(train_labels)==np.array(clean_labels))                                                       
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value()   

                    # ===  precision ===
                    num_selected = pred.sum()  # 
                    if num_selected > 0:
                        num_true_clean = clean[pred.astype(bool)].sum()  # 
                        precision_clean = num_true_clean / num_selected
                    else:
                        precision_clean = 0.0  #  0          
                    total_true_clean = clean.sum()
                    if total_true_clean > 0:
                        recall_clean = num_true_clean / total_true_clean
                    else:
                        recall_clean = 0.0

                    log.write('Number of labeled samples: %d   AUC: %.3f   Precision (true clean ratio): %.2f%%   Recall (selected in all clean): %.2f%%\n' %
                        (num_selected, auc, precision_clean * 100, recall_clean * 100))
                    log.flush() 


                    train_labels_np = np.array(train_labels)
                    clean_labels_np = np.array(clean_labels)
                    metrics = defaultdict(lambda: {"selected": 0, "true_clean": 0, "total_clean": 0})

                    for i in range(len(train_labels)):
                        cls = train_labels_np[i]
                        is_pred = pred[i]
                        is_clean = clean[i]
                        metrics[cls]["total_clean"] += is_clean
                        if is_pred:
                            metrics[cls]["selected"] += 1
                            metrics[cls]["true_clean"] += is_clean

                    log.write("=== Per-Class Statistics ===\n")
                    for cls in sorted(metrics.keys()):
                        m = metrics[cls]
                        selected = m["selected"]
                        true_clean = m["true_clean"]
                        total_clean = m["total_clean"]
                        precision = true_clean / selected if selected > 0 else 0.0
                        recall = true_clean / total_clean if total_clean > 0 else 0.0
                        log.write("Class %d - Selected: %d, TrueClean: %d, TotalClean: %d, Precision: %.2f%%, Recall: %.2f%%\n" %
                                (cls, selected, true_clean, total_clean, precision * 100, recall * 100))
                    log.flush()
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                             
                
                self.train_data = train_data[pred_idx]
                self.noise_label = [train_labels[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            filename = img + ".jpeg"
            img_path = os.path.join(self.image_root, filename)
            img = Image.open(img_path).convert('RGB')
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            filename = img + ".jpeg"
            img_path = os.path.join(self.image_root, filename)
            img = Image.open(img_path).convert('RGB')
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            filename = img + ".jpeg"
            img_path = os.path.join(self.image_root, filename)
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)            
            return img, target, index        
        elif self.mode=='test':
            filename = self.test_data[index] + ".jpeg"
            img_path = os.path.join(self.image_root, filename)
            img = Image.open(img_path).convert('RGB')
            target = self.test_labels[index]
            img = self.transform(img)            
            return img, target
        elif self.mode == 'val':
            filename = self.val_data[index] + ".jpeg"
            img_path = os.path.join(self.image_root, filename)
            img = Image.open(img_path).convert('RGB')
            target = self.val_labels[index]
            img = self.transform(img)           
            return img, target
    def __len__(self):
        if self.mode!='test':
            if self.mode=='val':
                return len(self.val_data)
            else:
                return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class kaggledr_dataloader():  
    def __init__(self, dataset, noise_rate, noise_mode, batch_size, num_workers, log, random_seed, num_class, device ):
        self.dataset = dataset
        self.r = noise_rate
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.root_dir = root_dir
        self.log = log
    
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])     
        elif self.dataset == 'kaggledr':
            self. transform_train = transforms.Compose([
                transforms.Resize((520, 520)),  # 
                transforms.RandomResizedCrop(size=512, scale=(0.95, 1.0), ratio=(0.95, 1.05)),
                transforms.RandomHorizontalFlip(p=0.5),  # 
                transforms.ColorJitter(brightness=0.1, contrast=0.1),  # ±10%
                transforms.RandomRotation(degrees=5),  #  ±5°
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std =[0.229, 0.224, 0.225])
            ])
            self.transform_test = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std =[0.229, 0.224, 0.225])
                ])
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = kaggledr_dataset(dataset=self.dataset, noise_mode=self.noise_mode, noise_rate=self.r,  transform=self.transform_train,
                                             mode="all",random_seed=0,num_class=5,device=None)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = kaggledr_dataset(dataset=self.dataset, noise_mode=self.noise_mode, noise_rate=self.r,  transform=self.transform_train, 
                                                mode="labeled", pred=pred, probability=prob,log=self.log,random_seed=0,num_class=5,device=None)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = kaggledr_dataset(dataset=self.dataset, noise_mode=self.noise_mode, noise_rate=self.r, 
                                                  transform=self.transform_train, mode="unlabeled", pred=pred,random_seed=0,num_class=5,device=None)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = kaggledr_dataset(dataset=self.dataset, noise_mode=self.noise_mode, noise_rate=self.r,  transform=self.transform_test,
                                              mode='test',random_seed=0,num_class=5,device=None)      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = kaggledr_dataset(dataset=self.dataset, noise_mode=self.noise_mode, noise_rate=self.r, transform=self.transform_test, mode='all',
                                             random_seed=0,num_class=5,device=None )      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        
        elif mode=='val':
            val_dataset = kaggledr_dataset(dataset=self.dataset, noise_mode=self.noise_mode, noise_rate=self.r, transform=self.transform_test, 
                                            mode='val',random_seed=0,num_class=5,device=None)      
            val_loader = DataLoader(
                dataset=val_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return val_loader