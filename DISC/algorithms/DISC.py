# -*- coding: utf-8 -*-
# @Author : Jack
# @Email  : liyifan20g@ict.ac.cn
# @File   : DISC.py 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet50, vgg19_bn
import torch.nn as nn
from models import InceptionResNetV2
import numpy as np
from utils import get_model
from losses import GCELoss, Mixup
from tqdm import tqdm
import pickle
import torchvision.models as models
import os
from typing import Optional

class DISC:
    def __init__(
            self, 
            config: dict = None, 
            input_channel: int = 3, 
            num_classes: int = 7,
            clean_label: Optional[np.ndarray] = None
        ):
        self.num_classes = num_classes
        device = torch.device('cuda:%s'%config['gpu']) if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.epochs = config['epochs']
        self.tmp_epoch = 0
        self.dataset = config['dataset']
        self.noise_type = config['noise_type']+'_'+str(config['noise_rate'])
        self.noise_rate = config['noise_rate']
        # self.gt_labels = clean_label
        self.gt_labels = torch.as_tensor(clean_label, dtype=torch.long, device=self.device)
        self.lr = config['lr']
        
        # Backbones for different datasets
  
            
    
        self.model_scratch = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model_scratch.fc = nn.Linear(self.model_scratch.fc.in_features, config['num_classes'])
        self.model_scratch = self.model_scratch.to(self.device)


        # Thresholds for different subsets
        self.adjust_lr = 1
        self.lamd_ce = 1
        self.lamd_h = 1
        self.sigma = 0.5
        self.momentum = config['momentum']

        if config['dataset'] == 'cifar-10':
            if config['noise_type']=='asym':
                self.momentum = 0.95
            self.start_epoch = 20
            
            if config['noise_type']=='ins':
                self.start_epoch = 15
                
            if config['noise_rate'] in [0.6, 0.8]:
                self.lamd_h = 0.2
                self.momentum = 0.95
        elif config['dataset'] == 'pathmnist':
            if config['noise_type']=='asym':
                self.momentum = 0.95
            self.start_epoch = 5
            
            if config['noise_type']=='instance':
                self.start_epoch = 5
                
            if config['noise_rate'] in [0.6, 0.8,0.9]:
                self.lamd_h = 0.2
                self.momentum = 0.95
        elif config['dataset'] == 'dermamnist' or config['dataset'] == 'bloodmnist' or config['dataset'] == 'organcmnist':
            if config['noise_type']=='asym':
                self.momentum = 0.95
            self.start_epoch = 5
            
            if config['noise_type']=='instance':
                self.start_epoch = 5
                
            if config['noise_rate'] in [0.6, 0.8,0.9]:
                self.lamd_h = 0.2
                self.momentum = 0.95
        elif config['dataset'] == 'drtid':
            if config['noise_type']=='asym':
                self.momentum = 0.95
            self.start_epoch = 5
            
            if config['noise_type']=='instance':
                self.start_epoch = 5
                
            if config['noise_rate'] in [0.6, 0.8,0.9]:
                self.lamd_h = 0.2
                self.momentum = 0.95
        elif config['dataset'] == 'kaggledr':
            if config['noise_type']=='asym':
                self.momentum = 0.95
            self.start_epoch = 5
            
            if config['noise_type']=='instance':
                self.start_epoch = 5
                
            if config['noise_rate'] in [0.6, 0.8,0.9]:
                self.lamd_h = 0.2
                self.momentum = 0.95   
                
        elif config['dataset'] == 'chexpert':
            if config['noise_type']=='asym':
                self.momentum = 0.95
            self.start_epoch = 5
            
            if config['noise_type']=='instance':
                self.start_epoch = 5
                
            if config['noise_rate'] in [0.6, 0.8,0.9]:
                self.lamd_h = 0.2
                self.momentum = 0.95   
        config['start_epoch'] = self.start_epoch
        config['momentum'] = self.momentum

        # Optimizers for different subsets
        if config['noise_rate'] > 0.6 and 'dermamnist' in config['dataset']:
            self.lr = 0.001
            # Adjust learning rate and betas for Adam Optimizer
            mom1 = 0.9
            mom2 = 0.1
            self.alpha_plan = [self.lr] * config['epochs']
            self.beta1_plan = [mom1] * config['epochs']
            self.epoch_decay_start = config['epoch_decay_start']
            for i in range(config['epoch_decay_start'], config['epochs']):
                self.alpha_plan[i] = float(config['epochs'] - i) / (config['epochs'] - config['epoch_decay_start']) * self.lr
                self.beta1_plan[i] = mom2

            self.optimizer = torch.optim.Adam(self.model_scratch.parameters(), lr=self.lr)
            config['optimizer'] = 'adam' 
            self.optim_type = 'adam'

        else:
            
            self.lr = 0.1
            self.weight_decay = 1e-3
            self.optimizer = torch.optim.SGD(self.model_scratch.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            config['optimizer'] = 'sgd' 
            self.optim_type = 'sgd'
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 40], verbose=True)

            config['weight_decay'] = self.weight_decay

        config['lr'] = self.lr
        
        if 'cifar' in config['dataset']:
            N = 50000
        elif 'pathmnist' in config['dataset']:
            N = 89996
        elif 'dermamnist' in config['dataset']:
            N = 7007
        elif 'bloodmnist' in config['dataset']:
            N = 11959

        elif 'organcmnist' in config['dataset']:
            N = 12975

        elif 'drtid' in config['dataset']:
            N = 1600
        elif 'kaggledr' in config['dataset']:
            N = 70247
        elif 'chexpert' in config['dataset']:
            N = 64208
        self.N = N
        
        # Variable definition
        self.s_prev_confidence = torch.ones(N).to(self.device)*1/N
        self.w_prev_confidence = torch.ones(N).to(self.device)*1/N
        self.ws_prev_confidence = torch.ones(N).to(self.device)*1/N

        self.w_probs = torch.zeros(N, config['num_classes']).to(self.device)
        self.s_probs = torch.zeros(N, config['num_classes']).to(self.device)
        self.labels = torch.ones(N).long().to(self.device)

        if 'cifar' in config['dataset']:
            self.gt_labels = torch.tensor(self.get_gt_labels(config['dataset'], config['root'])).to(self.device)
        self.weak_labels = self.labels.detach().clone()

        self.clean_flags = torch.zeros(N).bool().to(self.device)
        self.hard_flags = torch.zeros(N).bool().to(self.device)
        self.correction_flags = torch.zeros(N).bool().to(self.device)
        self.weak_flags = torch.zeros(N).bool().to(self.device)
        self.w_selected_flags = torch.zeros(N).bool().to(self.device)
        self.s_selected_flags = torch.zeros(N).bool().to(self.device)
        self.selected_flags = torch.zeros(N).bool().to(self.device)
        self.class_weight = torch.ones(self.num_classes).to(self.device)
        self.accs = np.zeros(self.epochs)
        self.acc_list = list()
        self.num_list = list()
        
        # Loss function definition
        self.GCE_loss = GCELoss(num_classes=num_classes, gpu=config['gpu'])
        self.mixup_loss = Mixup(gpu=config['gpu'], num_classes=num_classes, alpha=config['alpha'])
        self.criterion = nn.CrossEntropyLoss()
        self.feat_dim = 2048
        self.features = torch.zeros(self.N, self.feat_dim, dtype=torch.float32)
        self._last_feat = None



        # 
        self._register_feature_hook()

    def _register_feature_hook(self):
        def hook(module, inp, out):
            # out shape: [B, 2048, 1, 1] → flatten  [B, 2048]
            self._last_feat = torch.flatten(out, 1).detach()
        self._feat_hook = self.model_scratch.avgpool.register_forward_hook(hook)

    def train(self, train_loader, epoch):
        print('Training ...')
        self.model_scratch.train()
        self.tmp_epoch = epoch
        pbar = tqdm(train_loader)
        # pbar = train_loader
        if epoch < self.start_epoch:
            for (images, targets, indexes) in pbar:
                w_imgs, s_imgs = Variable(images[0]).to(self.device, non_blocking=True), \
                                Variable(images[1]).to(self.device, non_blocking=True)
                targets = Variable(targets).to(self.device)

                all_inputs = torch.cat([w_imgs, s_imgs], dim=0)
                bs = w_imgs.shape[0]
                logits = self.model_scratch(all_inputs)
                w_logits = logits[:bs]
                s_logits = logits[bs:]
                loss_sup = self.criterion(w_logits, targets) \
                         + self.criterion(s_logits, targets)
                self.optimizer.zero_grad()
                loss_sup.backward()
                self.optimizer.step()
                
                with torch.no_grad():
                    w_prob = F.softmax(w_logits, dim=1)
                    self.w_probs[indexes] = w_prob
                    s_prob = F.softmax(s_logits, dim=1)
                    self.s_probs[indexes] = s_prob
               
                pbar.set_description(
                        'Epoch [%d/%d], loss_sup: %.4f'
                        % (epoch + 1, self.epochs, loss_sup.data.item()))
        else:
            for (images, targets, indexes) in pbar:
                w_imgs, s_imgs = Variable(images[0]).to(self.device, non_blocking=True), \
                                 Variable(images[1]).to(self.device, non_blocking=True)
                targets = Variable(targets).to(self.device)
                
                #############CE+GCE############
                all_inputs = torch.cat([w_imgs, s_imgs], dim=0)
                bs = w_imgs.shape[0]
                logits = self.model_scratch(all_inputs)
                w_logits = logits[:bs]
                s_logits = logits[bs:]

                feats = self._last_feat             # [2*bs, 2048]
                w_feats = feats[:bs]

                #  weak 
                batch_feats = w_feats

                # 
                self.features[indexes] = batch_feats.cpu()
                
                loss_sup = torch.tensor(0).float().to(self.device)
                
                b_clean_flags = self.clean_flags[indexes]
                clean_num = b_clean_flags.sum()
                b_hard_flags = self.hard_flags[indexes]
                hard_num = b_hard_flags.sum()
                
                batch_size = len(w_imgs)

                if clean_num:
                    clean_loss_sup = self.criterion(w_logits[b_clean_flags], targets[b_clean_flags]) \
                            + self.criterion(s_logits[b_clean_flags], targets[b_clean_flags])
                    loss_sup += clean_loss_sup * self.lamd_ce * (clean_num/batch_size) 
                if hard_num:
                    hard_loss_sup = self.GCE_loss(w_logits[b_hard_flags], targets[b_hard_flags]) \
                        + self.GCE_loss(s_logits[b_hard_flags], targets[b_hard_flags])
                    loss_sup += hard_loss_sup * self.lamd_h * (hard_num/batch_size)
                ###########################
                
                ############Mixup##########
                weak_labels = self.weak_labels[indexes]      
                weak_flag = self.weak_flags[indexes] 
                weak_num = weak_flag.sum() 

                if weak_num:        
                    mixup_loss = self.mixup_loss(w_imgs[weak_flag], weak_labels[weak_flag], self.model_scratch)
                    mixup_loss += self.mixup_loss(s_imgs[weak_flag], weak_labels[weak_flag], self.model_scratch)
                    loss_sup += mixup_loss
                    
                #######################
                with torch.no_grad():
                    w_prob = F.softmax(w_logits, dim=1)
                    self.w_probs[indexes] = w_prob
                    s_prob = F.softmax(s_logits, dim=1)
                    self.s_probs[indexes] = s_prob

                if loss_sup:
                    self.optimizer.zero_grad()
                    loss_sup.backward()
                    self.optimizer.step()

                pbar.set_description(
                        'Epoch [%d/%d], loss_sup: %.4f'
                        % (epoch + 1, self.epochs, loss_sup.data.item()))

        with torch.no_grad():
            ws_probs = (self.w_probs+self.s_probs)/2
            w_prob_max, w_label= torch.max(self.w_probs, dim=1)
            s_prob_max, s_label= torch.max(self.s_probs, dim=1)
            ws_prob_max, ws_label= torch.max(ws_probs, dim=1)
            labels = w_label.clone()
            
            ###############Selection###############
            w_mask = self.w_probs[self.labels>=0, self.labels]>self.w_prev_confidence[self.labels>=0]
            s_mask = self.s_probs[self.labels>=0, self.labels]>self.s_prev_confidence[self.labels>=0]
            self.clean_flags = w_mask & s_mask
            self.selected_flags = w_mask + s_mask
            self.w_selected_flags = w_mask & (~self.clean_flags) #H_w
            self.s_selected_flags = s_mask & (~self.clean_flags) #H_s
            self.hard_flags = self.w_selected_flags + self.s_selected_flags #H       
            #######################################

            ###############Correction##############
            ws_threshold = (self.w_prev_confidence + self.s_prev_confidence)/2 + self.sigma
            ws_threshold = torch.min(ws_threshold, torch.tensor(0.99).to(self.device))
            self.correction_flags = ws_prob_max > ws_threshold
            self.correction_flags = self.correction_flags & (~ self.selected_flags) # P-(C+H)
            #######################################
            
            ###############Mix set###############
            self.weak_flags = self.correction_flags + self.selected_flags
            self.weak_labels[self.selected_flags] = self.labels[self.selected_flags]
            self.weak_labels[self.correction_flags] = ws_label[self.correction_flags]
            #######################################

            self.w_prev_confidence = self.momentum*self.w_prev_confidence + (1-self.momentum)*w_prob_max
            
            self.s_prev_confidence = self.momentum*self.s_prev_confidence + (1-self.momentum)*s_prob_max
            # #  clean noisy label  gt label 
            # print("labels ", self.labels.min().item(), self.labels.max().item())
            # print("clean_flags :")
            # print(torch.bincount(self.labels[self.clean_flags]))
            # print("laebl :")
            # print(torch.bincount(self.labels))

            true_clean_mask = (self.labels == self.gt_labels)

            #  clean TP
            selected_clean = (true_clean_mask & self.clean_flags).sum()

            # TP + FN
            total_true_clean = true_clean_mask.sum()

            # 
            clean_recall = selected_clean / total_true_clean

            
            # if 'pathmnist' in self.dataset:
            clean_acc = (self.labels[self.clean_flags]==self.gt_labels[self.clean_flags]).sum()/self.clean_flags.sum()
            # clean_recall = ((self.labels == self.gt_labels) & self.clean_flags).sum() / (self.labels == self.gt_labels).sum()
            hard_acc = (self.labels[self.hard_flags]==self.gt_labels[self.hard_flags]).sum()/self.hard_flags.sum()
            selection_acc = (self.labels[self.selected_flags]==self.gt_labels[self.selected_flags]).sum()/self.selected_flags.sum()
            w_selection_acc = (self.labels[self.w_selected_flags]==self.gt_labels[self.w_selected_flags]).sum()/self.w_selected_flags.sum()
            s_selection_acc = (self.labels[self.s_selected_flags]==self.gt_labels[self.s_selected_flags]).sum()/self.s_selected_flags.sum()
            correction_acc = (self.weak_labels[self.correction_flags]==self.gt_labels[self.correction_flags]).sum()/self.correction_flags.sum()
            weak_acc = (self.weak_labels[self.weak_flags]==self.gt_labels[self.weak_flags]).sum()/self.weak_flags.sum()
            total_acc = (labels == self.gt_labels).sum()/self.N
            num_classes = self.num_classes
            per_class_precision = torch.zeros(num_classes).to(self.device)
            per_class_recall = torch.zeros(num_classes).to(self.device)

            true_clean_mask = (self.labels == self.gt_labels)

            for cls in range(num_classes):
                #  gt = labels  label = cls  clean 
                cls_true_clean_mask = (self.labels == cls) & (self.gt_labels == cls)

                #  clean_flags  clean 
                cls_selected_clean = cls_true_clean_mask & self.clean_flags

                #  clean
                cls_clean_predicted = (self.clean_flags) & (self.labels == cls)

                # recall_c =  / 
                recall_denom = cls_true_clean_mask.sum()
                recall_c = cls_selected_clean.sum() / recall_denom if recall_denom > 0 else torch.tensor(0.0).to(self.device)

                # precision_c = clean_flags 
                precision_denom = cls_clean_predicted.sum()
                precision_c = cls_selected_clean.sum() / precision_denom if precision_denom > 0 else torch.tensor(0.0).to(self.device)

                per_class_recall[cls] = recall_c
                per_class_precision[cls] = precision_c

            # log_path = f"/home/user/label_noise/DISC/clean_logs/{self.dataset}_{self.noise_type}_{self.noise_rate}_perclass_stats.txt"
            # os.makedirs(os.path.dirname(log_path), exist_ok=True)

            # with open(log_path, "a") as f:
            #     f.write(f"Epoch {epoch}\n")
            #     f.write(f"Precision (true clean ratio): {100 * clean_acc.item():.2f}%   Recall (selected in all clean): {100 * clean_recall.item():.2f}%    \n")
            #     f.write(f"Hard ratio : {100 * hard_acc.item():.2f}%   Correction ratio: {100 * correction_acc.item():.2f}%    \n")
            #     f.write("=== Per-Class Statistics ===\n")

            #     for cls in range(self.num_classes):
            #         selected = int(((self.clean_flags) & (self.labels == cls)).sum().item())
            #         true_clean = int(((self.clean_flags) & (self.labels == cls) & (self.gt_labels == cls)).sum().item())
            #         total_clean = int(((self.labels == cls) & (self.gt_labels == cls)).sum().item())

            #         precision = (true_clean / selected * 100) if selected > 0 else 0.0
            #         recall = (true_clean / total_clean * 100) if total_clean > 0 else 0.0

            #         f.write(f"Class {cls} - Selected: {selected}, TrueClean: {true_clean}, TotalClean: {total_clean}, "
            #                 f"Precision: {precision:.2f}%, Recall: {recall:.2f}%\n")
                
            #     f.write("\n")

            #     f.write("=== Hard Sample Statistics ===\n")
            #     for cls in range(self.num_classes):
            #         selected = int(((self.hard_flags) & (self.labels == cls)).sum().item())
            #         true_clean = int(((self.hard_flags) & (self.labels == cls) & (self.gt_labels == cls)).sum().item())
            #         precision = (true_clean / selected * 100) if selected > 0 else 0.0

            #         f.write(f"Class {cls} - Selected: {selected}, TrueClean: {true_clean}, "
            #                 f"Precision: {precision:.2f}%\n")

            #     f.write("=== Correction Sample Statistics ===\n")
            #     for cls in range(self.num_classes):
            #         selected = int(((self.correction_flags) & (self.weak_labels == cls)).sum().item())
            #         true_clean = int(((self.correction_flags) & (self.weak_labels == cls) & (self.gt_labels == cls)).sum().item())
            #         precision = (true_clean / selected * 100) if selected > 0 else 0.0

            #         f.write(f"Class {cls} - Selected: {selected}, TrueClean: {true_clean}, "
            #                 f"Precision: {precision:.2f}%\n")

            # mask_clean_sel  =  true_clean_mask &  self.clean_flags
            # mask_clean_uns  =  true_clean_mask & (~self.clean_flags)
            # mask_noisy_sel  = (~true_clean_mask) &  self.clean_flags
            # mask_noisy_uns  = (~true_clean_mask) & (~self.clean_flags)
            # if (epoch ) > 5:
            #     out_dir = f"/home/user/label_noise/DISC/features/{self.dataset}_{self.noise_type}_{self.noise_rate}"
            #     os.makedirs(out_dir, exist_ok=True)
            #     torch.save({
            #         "features": self.features,  # [N, 2048]
            #         "labels_noisy": self.labels.cpu(),
            #         "labels_gt": self.gt_labels.cpu(),
            #         "mask_clean_sel": mask_clean_sel.cpu(),
            #         "mask_clean_uns": mask_clean_uns.cpu(),
            #         "mask_noisy_sel": mask_noisy_sel.cpu(),
            #         "mask_noisy_uns": mask_noisy_uns.cpu()
            #     }, f"{out_dir}/epoch_{epoch:03d}.pt")



            print("Clean Precision (acc) is %.4f, Clean Recall is %.4f, Clean num is %d" % (
                    clean_acc.item(), clean_recall.item(), self.clean_flags.sum().item()))

            print("Hard ratio is %.4f, hard num is %d" % (hard_acc.item(), self.hard_flags.sum().item()))
            print("Weak selection ratio is %.4f, weak selection num is %d" % (w_selection_acc.item(), self.w_selected_flags.sum().item()))
            print("Strong selection ratio is %.4f, strong selection num is %d" % (s_selection_acc.item(), self.s_selected_flags.sum().item()))
            print("Selection ratio is %.4f, selection num is %d" % (selection_acc.item(), self.selected_flags.sum().item()))
            print("Correction ratio is %.4f, correction num is %d" % (correction_acc.item(), self.correction_flags.sum().item()))
            print("Weak ratio is %.4f, weak num is %d" % (weak_acc.item(), self.weak_flags.sum().item()))
            print("Total ratio is %.4f" % total_acc.item())
            self.acc_list.append((
                clean_acc.cpu().item(),
                hard_acc.cpu().item(),
                w_selection_acc.cpu().item(),
                s_selection_acc.cpu().item(),
                selection_acc.cpu().item(),
                correction_acc.cpu().item(),
                weak_acc.cpu().item(),
                total_acc.cpu().item()
            ))

        self.num_list.append((self.clean_flags.sum().cpu().numpy(), self.hard_flags.sum().cpu().numpy(), self.w_selected_flags.sum(), self.s_selected_flags.sum(), self.correction_flags.sum().cpu().numpy(), self.weak_flags.sum().cpu().numpy()))

        if epoch==(self.epochs-1):
            self.save_results()
        if self.adjust_lr:
            if self.optim_type == 'sgd':
                self.scheduler.step()
            elif self.optim_type == 'adam':
                self.adjust_learning_rate(self.optimizer, epoch)
                print("lr is %.8f." % (self.alpha_plan[epoch]))
                
    def evaluate(self, test_loader):
        print('Evaluating ...')

        self.model_scratch.eval()  # Change model to 'eval' mode

        correct = 0
        correct_top5 = 0
        total = 0

        for images, labels in test_loader:
            images = Variable(images).to(self.device)
            logits = self.model_scratch(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()
        
        acc = 100 * float(correct) / float(total)
        self.accs[self.tmp_epoch] = acc
     
        return acc

    def save_checkpoints(self):
        checkpoint_root = 'checkpoints/%s/'%self.dataset
        filename = checkpoint_root + 'save_epoch199_%s'%self.noise_type
        
        if not os.path.exists(checkpoint_root):
            os.makedirs(checkpoint_root)

        state = {'weak_labels':self.weak_labels, 'weak_flags':self.weak_flags, 'weak selected flags':self.w_selected_flags, 
                 'strong selected flags':self.s_selected_flags, 'clean_flags':self.clean_flags, 'labels':self.labels,
                 'hard_flags':self.hard_flags, 'correction_flags':self.correction_flags, 'w_probs':self.w_probs, 's_probs':self.s_probs, 
                 'epoch': self.start_epoch,'model': self.model_scratch.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename+'.pth')
        print("The model has been saved !!!!!")
    
    def save_results(self, name='disc'):
        save_root = 'result_root/%s/'%self.dataset
        filename = save_root + self.noise_type + '_save_' + name + '.npy'

        if not os.path.exists(save_root):
            os.makedirs(save_root)
        if 'cifar' in self.dataset:
            results = {'num_list': self.num_list, 'acc_list': self.acc_list, 'test_acc': self.accs}
        else:
            results = {'num_list': self.num_list, 'test_acc': self.accs}
        np.save(filename, results)

    def load_checkpoints(self):
        filename = 'checkpoints/%s/start_epoch%d.pth'%(self.dataset, self.start_epoch)
        model_parameters = torch.load(filename)
        self.model_scratch.load_state_dict(model_parameters['model'])
        self.s_selected_flags = model_parameters['strong selected flags']      
        self.w_selected_flags = model_parameters['weak selected flags']      
        self.weak_labels = model_parameters['weak_labels'].to(self.device)
        self.weak_flags = model_parameters['weak_flags'].to(self.device)
        self.clean_flags = model_parameters['clean_flags'].to(self.device)
        self.correction_flags = model_parameters['correction_flags'].to(self.device)
        self.hard_flags = model_parameters['hard_flags'].to(self.device)
        self.w_probs = model_parameters['w_probs'].to(self.device)
        self.s_probs = model_parameters['s_probs'].to(self.device)
        print("The model has been loaded !!!!!")


    def get_labels(self, train_loader):
        print("Loading labels......")
        pbar = tqdm(train_loader)
        for (_, targets, indexes) in pbar:
            targets = targets.to(self.device)
            self.labels[indexes] = targets
        print("The labels are loaded!")

    def get_gt_labels(self, dataset, root):
        if dataset=='cifar-10':
            train_list = [
                ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
                ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
                ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
                ['data_batch_4', '634d18415352ddfa80567beed471001a'],
                ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
            ]
            base_folder = 'cifar-10-batches-py'
        elif dataset=='cifar-100':
            train_list = [
                ['train', '16019d7e3df5f24257cddd939b257f8d'],
            ]
            base_folder = 'cifar-100-python'
        targets = []
        for file_name, checksum in train_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])
        return targets

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
