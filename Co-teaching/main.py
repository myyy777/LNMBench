# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from data.pathmnist import PATHMNIST,PATHMNIST_TEST,PATHMNIST_VAL
from data.dermamnist import DERMAMNIST, DERMAMNIST_VAL, DERMAMNIST_TEST
from data.bloodmnist import BLOODMNIST, BloodMNIST_VAL, BloodMNIST_TEST
from data.organcmnist import ORGANCMNIST, OrganCMNIST_VAL, OrganCMNIST_TEST
from drtid import DRTID_val, DRTID, DRTID_test
from kaggledr import kaggledr, kaggledr_val,kaggledr_test
from chexpert import chexpert, chexpert_val, chexpert_test
import argparse, sys
import numpy as np
import datetime
import shutil
from data.utils import set_seed
from loss import loss_coteaching
import torchvision.models as models
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--log_dir', type = str, help = 'dir to save result txt files', default = 'logs_clean/')
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.38)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1.0, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'pathmnist')
parser.add_argument('--n_epoch', type=int, default=50)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--epoch_decay_start', type=int, default=20)
parser.add_argument('--split_per', type=float, default=0.9)
parser.add_argument('--gpu', type=int, help='ind of gpu', default=2)
parser.add_argument('--adjust_lr', type=int, help='adjust lr', default=1)
parser.add_argument('--batch', type=int, help='adjust lr', default=128)
args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
# Seed
set_seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = args.batch
learning_rate = args.lr 

# load dataset  

if args.dataset == 'pathmnist':
    input_channel=3
    num_classes=9
    args.top_bn = False
    args.epoch_decay_start = 20
    args.n_epoch = 50 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std =[0.229, 0.224, 0.225])
    ])
    train_dataset =PATHMNIST(
                                    transform=transform,

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)

    val_dataset = PATHMNIST_VAL(
                                    transform=transform,
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)

    test_dataset = PATHMNIST_TEST(transform=transform)


if args.dataset == 'dermamnist':
    input_channel=3
    num_classes=7
    args.top_bn = False
    args.epoch_decay_start = 20
    args.n_epoch = 50 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std =[0.229, 0.224, 0.225])
    ])
    train_dataset =DERMAMNIST(
                                    transform=transform,

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)

    val_dataset = DERMAMNIST_VAL(
                                    transform=transform,
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)

    test_dataset = DERMAMNIST_TEST(transform=transform)

if args.dataset == 'bloodmnist':
    input_channel=3
    num_classes=8
    args.top_bn = False
    args.epoch_decay_start = 20
    args.n_epoch = 50 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std =[0.229, 0.224, 0.225])
    ])
    train_dataset =BLOODMNIST(
                                    transform=transform,

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)

    val_dataset = BloodMNIST_VAL(
                                    transform=transform,
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)

    test_dataset = BloodMNIST_TEST(transform=transform)

if args.dataset == 'organcmnist':
    input_channel=3
    num_classes=11
    args.top_bn = False
    args.epoch_decay_start = 20
    args.n_epoch = 50 
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std =[0.229, 0.224, 0.225])
    ])
    train_dataset =ORGANCMNIST(
                                    transform=transform,

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)

    val_dataset = OrganCMNIST_VAL(
                                    transform=transform,
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)

    test_dataset = OrganCMNIST_TEST(transform=transform)

if args.dataset == 'drtid':
    input_channel=3
    num_classes=5
    args.top_bn = False
    args.epoch_decay_start = 20
    args.n_epoch = 50 
    transform_train = transforms.Compose([
        transforms.Resize((520, 520)),  # 
        transforms.RandomResizedCrop(size=512, scale=(0.95, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),  # 
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # ±10%
        transforms.RandomRotation(degrees=5),  #  ±5°
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std =[0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std =[0.229, 0.224, 0.225])
    ])
    train_dataset =DRTID(
                                    transform=transform_train,

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)

    val_dataset = DRTID_val(
                                    transform=transform,
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)

    test_dataset = DRTID_test(transform=transform)


if args.dataset == 'kaggledr':
    input_channel=3
    num_classes=5
    args.top_bn = False
    args.epoch_decay_start = 20
    args.n_epoch = 50 
    transform_train = transforms.Compose([
        transforms.Resize((520, 520)),  # 
        transforms.RandomResizedCrop(size=512, scale=(0.95, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),  # 
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # ±10%
        transforms.RandomRotation(degrees=5),  #  ±5°
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std =[0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std =[0.229, 0.224, 0.225])
    ])
    train_dataset =kaggledr(
                                    transform=transform_train,

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)

    val_dataset = kaggledr_val(
                                    transform=transform,
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)

    test_dataset = kaggledr_test(transform=transform)

if args.dataset == 'chexpert':
    input_channel=3
    num_classes=5
    args.top_bn = False
    args.epoch_decay_start = 20
    args.n_epoch = 50 
    transform_train = transforms.Compose([
        transforms.Resize((230, 230)),  # 
        transforms.RandomResizedCrop(size=224, scale=(0.95, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),  # 
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # ±10%
        transforms.RandomRotation(degrees=5),  #  ±5°
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std =[0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std =[0.229, 0.224, 0.225])
    ])
    train_dataset = chexpert(
                                    transform=transform_train,

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)

    val_dataset = chexpert_val(
                                    transform=transform,
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)

    test_dataset = chexpert_test(transform=transform)

if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

noise_or_not = train_dataset.noise_or_not

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1
        
# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)
   
save_dir = args.result_dir +'/' +args.dataset+'/coteaching/'
log_dir = args.log_dir +'/' +args.dataset+'/coteaching/'

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

if not os.path.exists(log_dir):
    os.system('mkdir -p %s' % log_dir)

model_str=args.dataset+'_coteaching_'+args.noise_type+'_'+str(args.noise_rate)+'_'+'seed' + str(args.seed)+'_'+'adjust_lr'+str(args.adjust_lr)+'epoch'+str(args.n_epoch)
log_str=args.dataset+'_coteaching_'+args.noise_type+'_'+str(args.noise_rate)+'_'+'seed' + str(args.seed)
txtfile=save_dir+"/"+model_str+".txt"
logfile = log_dir+"/"+log_str+".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))

if os.path.exists(logfile):
    os.system('mv %s %s' % (logfile, logfile+".bak-%s" % nowTime))

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(train_loader,epoch, model1, optimizer1, model2, optimizer2):
    print('Training %s...' % model_str) 
    pure_ratio_list=[]
    pure_ratio_1_list=[]
    pure_ratio_2_list=[]
    
    train_total=0
    train_correct=0 
    train_total2=0
    train_correct2=0 
    stats_1 = {"selected": 0, "true_clean": 0, "total_clean": 0}
    stats_2 = {"selected": 0, "true_clean": 0, "total_clean": 0}
    class_stats_1 = {i: {"selected": 0, "true_clean": 0, "total_clean": 0} for i in range(num_classes)}
    class_stats_2 = {i: {"selected": 0, "true_clean": 0, "total_clean": 0} for i in range(num_classes)}


    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        # if i>args.num_iter_per_epoch:
        #     break
      
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        
        labels_np = labels.cpu().numpy()
        
        # Forward + Backward + Optimize
        logits1=model1(images)
        prec1, _ = accuracy(logits1, labels, topk=(1, 5))
        train_total+=1
        train_correct+=prec1

        logits2 = model2(images)
        prec2, _ = accuracy(logits2, labels, topk=(1, 5))
        train_total2+=1
        train_correct2+=prec2
        loss_1, loss_2, sel_1, sel_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not)
        # pure_ratio_1_list.append(100*pure_ratio_1)
        # pure_ratio_2_list.append(100*pure_ratio_2)
        for j in range(len(ind)):
            cls = labels_np[j]
            is_clean = noise_or_not[ind[j]]
            stats_1["total_clean"] += is_clean
            stats_2["total_clean"] += is_clean
            class_stats_1[cls]["total_clean"] += is_clean
            class_stats_2[cls]["total_clean"] += is_clean

        for idx1  in sel_1:
            i_global = ind[idx1]
            cls = labels_np[idx1]
            is_clean = noise_or_not[i_global]
            stats_1["selected"] += 1
            stats_1["true_clean"] += is_clean
            class_stats_1[cls]["selected"] += 1
            class_stats_1[cls]["true_clean"] += is_clean

        for idx2 in sel_2:
            i_global = ind[idx2]
            cls = labels_np[idx2]
            is_clean = noise_or_not[i_global]
            stats_2["selected"] += 1
            stats_2["true_clean"] += is_clean
            class_stats_2[cls]["selected"] += 1
            class_stats_2[cls]["true_clean"] += is_clean

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f' 
                  %(epoch, args.n_epoch, i+1, len(train_dataset)//batch_size, prec1, prec2, loss_1.item(), loss_2.item() ))
# ===  ===
    def compute_metrics(stats, class_stats):
        overall_precision = stats["true_clean"] / stats["selected"] if stats["selected"] > 0 else 0
        overall_recall = stats["true_clean"] / stats["total_clean"] if stats["total_clean"] > 0 else 0
        per_class_precision = {}
        per_class_recall = {}
        for cls in range(num_classes):
            cstat = class_stats[cls]
            p = cstat["true_clean"] / cstat["selected"] if cstat["selected"] > 0 else 0
            r = cstat["true_clean"] / cstat["total_clean"] if cstat["total_clean"] > 0 else 0
            per_class_precision[cls] = p
            per_class_recall[cls] = r
        return overall_precision, overall_recall, per_class_precision, per_class_recall

    precision_1, recall_1, per_class_prec_1, per_class_rec_1 = compute_metrics(stats_1, class_stats_1)
    precision_2, recall_2, per_class_prec_2, per_class_rec_2 = compute_metrics(stats_2, class_stats_2)

    with open(logfile, "a") as f:
        f.write(f"Epoch {epoch}\n")
        f.write(f"Precision (true clean ratio): {precision_1 * 100:.2f}%   Recall (selected in all clean): {recall_1 * 100:.2f}%\n")
        f.write("=== Per-Class Statistics ===\n")
        for cls in range(num_classes):
            f.write(f"Class {cls} - Selected: {class_stats_1[cls]['selected']}, "
                    f"TrueClean: {class_stats_1[cls]['true_clean']}, "
                    f"TotalClean: {class_stats_1[cls]['total_clean']}, "
                    f"Precision: {per_class_prec_1[cls] * 100:.2f}%, "
                    f"Recall: {per_class_rec_1[cls] * 100:.2f}%\n")
        f.write("-" * 60 + "\n")

    with open(logfile, "a") as f:
        f.write(f"Epoch {epoch}\n")
        f.write(f"Precision (true clean ratio): {precision_2 * 100:.2f}%   Recall (selected in all clean): {recall_2 * 100:.2f}%\n")
        f.write("=== Per-Class Statistics ===\n")
        for cls in range(num_classes):
            f.write(f"Class {cls} - Selected: {class_stats_2[cls]['selected']}, "
                    f"TrueClean: {class_stats_2[cls]['true_clean']}, "
                    f"TotalClean: {class_stats_2[cls]['total_clean']}, "
                    f"Precision: {per_class_prec_2[cls] * 100:.2f}%, "
                    f"Recall: {per_class_rec_2[cls] * 100:.2f}%\n")
        f.write("-" * 60 + "\n")


    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)

    # return precision_1, recall_1, per_class_prec_1, per_class_rec_1, \
    #        precision_2, recall_2, per_class_prec_2, per_class_rec_2

    return train_acc1, train_acc2

# Evaluate the Model
def evaluate(test_loader, model1, model2):
    print('Evaluating...')
    model1.eval()
    model2.eval()

    correct1 = 0
    correct2 = 0
    correct_ens = 0
    total = 0

    with torch.no_grad():
        for data, labels, _ in test_loader:
            data = Variable(data).to(device)
            labels = labels.to(device)

            logits1 = model1(data)
            logits2 = model2(data)

            # 
            _, pred1 = torch.max(logits1, 1)
            _, pred2 = torch.max(logits2, 1)

            # ensemble logits 
            logits_ens = logits1 + logits2
            _, pred_ens = torch.max(logits_ens, 1)

            correct1 += pred1.eq(labels).sum().item()
            correct2 += pred2.eq(labels).sum().item()
            correct_ens += pred_ens.eq(labels).sum().item()
            total += labels.size(0)

    acc1 = 100. * correct1 / total
    acc2 = 100. * correct2 / total
    acc_mean = 100. * correct_ens / total

    print(f"\n| Eval | Net1: {acc1:.2f}%, Net2: {acc2:.2f}%, Ensemble (Mean): {acc_mean:.2f}%\n")
    return acc1, acc2, acc_mean

def main():
    
    # Data Loader (Input Pipeline)
    print(args)
    print('loading dataset...') 
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size, 
                                            num_workers=args.num_workers, 
                                            shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=args.num_workers,                                       
                                              shuffle=False)
    # Define models
    print('building model...') 
    cnn1 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    cnn1.fc = nn.Linear(cnn1.fc.in_features, num_classes)  # PathMNIST  9 
    # cnn1 = CNN(input_channel=input_channel, n_outputs=num_classes)
    cnn1.to(device)
    print(cnn1.parameters) 
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)
    
    cnn2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    cnn2.fc = nn.Linear(cnn2.fc.in_features, num_classes) 
    cnn2.to(device)
    print(cnn2.parameters) 
    optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=learning_rate)

    mean_pure_ratio1=0
    mean_pure_ratio2=0

    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc1 train_acc2  val_acc1 val_acc2 val_acc_mean test_acc1 test_acc2 test_acc_mean pure_ratio1 pure_ratio2\n')

    epoch=0
    train_acc1=0
    train_acc2=0
    val_acc1, val_acc2, val_acc_mean = 0,0,0
    test_acc1, test_acc2, test_acc_mean = 0,0,0
    # evaluate models with random weights
    val_acc1, val_acc2, val_acc_mean=evaluate(val_loader, cnn1, cnn2)
    test_acc1, test_acc2, test_acc_mean=evaluate(test_loader, cnn1, cnn2)
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%' % (epoch, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) 
                     +' ' + str(val_acc1) +' '  + str(val_acc2) +' '+ str(val_acc_mean) +' ' + str(test_acc1) + " " 
                     + str(test_acc2) + ' '+ str(test_acc_mean) + "\n")

    # training
    val_acc_list = []
    test_acc_list = []
    for epoch in range(1, args.n_epoch):
        # train models
        cnn1.train()
        if args.adjust_lr:
            adjust_learning_rate(optimizer1, epoch)
        cnn2.train()
        if args.adjust_lr:
            adjust_learning_rate(optimizer2, epoch)
        
        train_acc1, train_acc2=train(train_loader, epoch, cnn1, optimizer1, cnn2, optimizer2)
        # evaluate models
        
        val_acc1, val_acc2, val_acc_mean=evaluate(val_loader, cnn1, cnn2)
        
        test_acc1, test_acc2, test_acc_mean=evaluate(test_loader, cnn1, cnn2)
        val_acc_list.append((val_acc1 + val_acc2) / 2)
        test_acc_list.append((test_acc1 + test_acc2) / 2)
        # save results
        # mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        # mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
        print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%, Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %%' % (epoch, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  
                         + str(train_acc2) +' '+ str(val_acc1) +' '  
                         + str(val_acc2) +' '+ str(val_acc_mean) +' '  + str(test_acc1) + " " + str(test_acc2) + ' ' + str(test_acc_mean) + "\n")
            
        noise_rate_str = str(args.noise_rate).replace('.', '_')
  
        save_dir = os.path.join("/mnt/ssd1/user/co_teaching", args.dataset, args.noise_type, f"nr{noise_rate_str}")
        os.makedirs(save_dir, exist_ok=True)

        torch.save(cnn1.state_dict(), os.path.join(save_dir, f'model1_epoch{epoch}.pth'))
        torch.save(cnn2.state_dict(), os.path.join(save_dir, f'model2_epoch{epoch}.pth'))
# 1. val  test acc
    best_val_epoch = val_acc_list.index(max(val_acc_list))
    test_at_best_val = test_acc_list[best_val_epoch]
    # 2. test acc 
    best_test_acc = max(test_acc_list)

    # 3.  5  epoch  test acc 
    last_5_test_acc = sum(test_acc_list[-5:]) / len(test_acc_list[-5:])
    
    print("\n=== Summary ===")
    print(f"Best Val Epoch: {best_val_epoch} / {args.n_epoch}")
    print(f"Test Acc at Best Val: {test_at_best_val:.4f} %")
    print(f"Best Test Acc: {best_test_acc:.4f} %")
    print(f"Mean Test Acc of Last 5 Epochs: {last_5_test_acc:.4f} %")

    #  txt 
    with open(txtfile, "a") as myfile:
        myfile.write("\n=== Summary ===\n")
        myfile.write(f"Best Val Epoch: {best_val_epoch + 1} / {args.n_epoch - 1}\n")
        myfile.write(f"Test Acc at Best Val: {test_at_best_val:.4f} %\n")
        myfile.write(f"Best Test Acc: {best_test_acc:.4f} %\n")
        myfile.write(f"Mean Test Acc of Last 5 Epochs: {last_5_test_acc:.4f} %\n")



if __name__=='__main__':
    main()
