from __future__ import print_function
import os
import torch
import tools

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
from data.pathmnist import PATHMNIST,PATHMNIST_TEST,PATHMNIST_VAL
from data.dermamnist import DERMAMNIST, DERMAMNIST_VAL, DERMAMNIST_TEST
from data.drtid import DRTID_val, DRTID, DRTID_test
from data.kaggledr import kaggledr, kaggledr_val,kaggledr_test
from data.chexpert import chexpert, chexpert_val, chexpert_test
from data.bloodmnist import BLOODMNIST, BloodMNIST_TEST, BloodMNIST_VAL
from data.organcmnist import ORGANCMNIST, OrganCMNIST_VAL, OrganCMNIST_TEST

from model import CNN
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
import argparse, sys
import numpy as np
import datetime
from loss import loss_ours

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=0, help="No.")
parser.add_argument('--d', type=str, default='output', help="description")
parser.add_argument('--p', type=int, default=0, help="print")
parser.add_argument('--num_classes', type=int, default=9, help="class")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--log_dir', type = str, help = 'dir to save result txt files', default = 'logs_clean/')
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results/')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.3)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric, trid, instance]', default='symmetric')
parser.add_argument('--num_gradual', type=int, default=10, help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the paper.')
parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, or imagenet_tiny', default='pathmnist')
parser.add_argument('--n_epoch', type=int, default=50)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=200)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--epoch_decay_start', type=int, default=20)
parser.add_argument('--model_type', type=str, help='[coteaching, coteaching_plus, ours]', default='ours')
parser.add_argument('--fr_type', type=str, help='forget rate type', default='type_1')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--co_lam', type=float, help='balance', default=0.15)
parser.add_argument('--gpu', type=int, help='ind of gpu', default=0)
parser.add_argument('--channel', type=int, help='channel', default=3)
parser.add_argument('--batch', type=int, help='adjust lr', default=16)
args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
tools.set_seed(args.seed)

# #
# torch.cuda.set_device(args.gpu)

# # Seed
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = args.batch
learning_rate = args.lr

# load dataset
def load_data(args):
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
   
        
   
    if args.dataset == 'organcmnist':
        args.channel = 3
        args.num_classes = 11
        args.feature_size = 3 * 224 * 224
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

    if args.dataset=='pathmnist':
        args.channel = 3
        args.num_classes = 9
        args.feature_size = 3 * 224 * 224
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
    
    
    if args.dataset=='dermamnist':
        args.channel = 3
        args.num_classes = 7
        args.feature_size = 3 * 224 * 224
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

        
    if args.dataset=='bloodmnist':
        args.channel = 3
        args.num_classes = 8
        args.feature_size = 3 * 224 * 224
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

    if args.dataset == 'drtid':
        args.channel = 3
        args.num_classes = 5
        args.feature_size = 3 * 512 * 512
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
        args.channel = 3
        args.num_classes = 5
        args.feature_size = 3 * 512 * 512
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
        args.channel = 3
        args.num_classes = 5
        args.feature_size = 3 * 224 * 224
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
            



    return train_dataset, val_dataset, test_dataset



if args.forget_rate is None:
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate

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
        param_group['betas']=(beta1_plan[epoch], 0.999)


def gen_forget_rate(fr_type='type_1'):
    if fr_type == 'type_1':
        rate_schedule = np.ones(args.n_epoch) * forget_rate
        rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)

    # if fr_type=='type_2':
    #    rate_schedule = np.ones(args.n_epoch)*forget_rate
    #    rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)
    #    rate_schedule[args.num_gradual:] = np.linspace(forget_rate, 2*forget_rate, args.n_epoch-args.num_gradual)

    return rate_schedule


rate_schedule = gen_forget_rate(args.fr_type)

save_dir = args.result_dir +'/' +args.dataset+'/codis/'
log_dir = args.log_dir +'/' +args.dataset+'/codis/'
if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)
if not os.path.exists(log_dir):
    os.system('mkdir -p %s' % log_dir)
model_str=args.dataset+'_codis_'+args.noise_type+'_'+str(args.noise_rate)+'_'+'seed' + str(args.seed)
log_str=args.dataset+'_codis_'+args.noise_type+'_'+str(args.noise_rate)+'_'+'seed' + str(args.seed)
txtfile = save_dir + "/" + model_str + ".txt"
logfile = log_dir+"/"+log_str+".txt"
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile + ".bak-%s" % nowTime))

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
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# init_epoch = 20
# Train the Model
def train(train_loader, epoch, model1, optimizer1, model2, optimizer2, noise_or_not, args):
   
    train_total = 0
    train_correct = 0
    train_total2 = 0
    train_correct2 = 0
    num_classes = args.num_classes
    
    pure_ratio_list=[]
    pure_ratio_1_list=[]
    pure_ratio_2_list=[]
    stats_1 = {"selected": 0, "true_clean": 0, "total_clean": 0}
    stats_2 = {"selected": 0, "true_clean": 0, "total_clean": 0}
    class_stats_1 = {i: {"selected": 0, "true_clean": 0, "total_clean": 0} for i in range(num_classes)}
    class_stats_2 = {i: {"selected": 0, "true_clean": 0, "total_clean": 0} for i in range(num_classes)}
    
        
    for i, (data, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        labels = Variable(labels).to(device)
        data = Variable(data).to(device)
        labels_np = labels.cpu().numpy()
        # Forward + Backward + Optimize

        logits1 = model1(data)
        prec1, = accuracy(logits1, labels, topk=(1,))
        train_total += 1
        train_correct += prec1


        logits2 = model2(data)
        prec2, = accuracy(logits2, labels, topk=(1,))
        train_total2 += 1
        train_correct2 += prec2
        # loss_1, _, pure_ratio_1, _ = loss_ours(
        #     logits1, logits2.detach(), labels,
        #     rate_schedule[epoch], ind, noise_or_not,
        #     co_lambda=args.co_lam,
        #     device = device
        # )
        loss_1, loss_2, sel_1, sel_2 = loss_ours(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not, co_lambda=args.co_lam, device= device)
        # _, loss_2, _, pure_ratio_2 = loss_ours(
        #     logits2, logits1.detach(), labels,
        #     rate_schedule[epoch], ind, noise_or_not,
        #     co_lambda=args.co_lam,
        #     device = device
        # )
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
        # loss_1.backward(retain_graph=True)
        

        optimizer2.zero_grad()
        (loss_1 + loss_2).backward()


        optimizer1.step()
        
        
        # loss_2.backward(retain_graph=True)

        optimizer2.step()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f' 
                  %(epoch, args.n_epoch, i+1, noise_or_not.shape[0]//batch_size, prec1, prec2, loss_1.item(), loss_2.item()))

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
    
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for data, labels, _ in test_loader:
       
        data = Variable(data).to(device)
        logits1 = model1(data)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels.long()).sum()

    model2.eval()  # Change model to 'eval' mode
    correct2 = 0
    total2 = 0
    for data, labels, _ in test_loader:
        
        data = Variable(data).to(device)
        logits2 = model2(data)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels.long()).sum()

    if total1 == 0: 
        acc1 = 0
    else:
        acc1 = 100 * float(correct1) / float(total1)
        
    if total2 == 0:
        acc2 = 0
    else:
        acc2 = 100 * float(correct2) / float(total2)
    
    return acc1, acc2

def main(args):
    # Data Loader (Input Pipeline)
    print(args)

    
    train_dataset, val_dataset, test_dataset = load_data(args)


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
    
    
    noise_or_not = train_dataset.noise_or_not
   
    # Define models
    print('building model...')
    
    # clf1 = CNN(input_channel=args.channel, n_outputs=args.c)
    clf1 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    clf1.fc = nn.Linear(clf1.fc.in_features, args.num_classes)
    clf1.to(device)
    print(clf1)
    
    
    # clf2 = CNN(input_channel=args.channel, n_outputs=args.c)
    clf2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    clf2.fc = nn.Linear(clf2.fc.in_features, args.num_classes)
    clf2.to(device)
    print(clf2)
    optimizer1 = torch.optim.Adam(clf1.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.Adam(clf2.parameters(), lr=learning_rate)
    with open(txtfile, "a") as myfile:
        myfile.write('epoch train_acc1 train_acc2 val_acc1 val_acc2 test_acc1 test_acc2 \n')

    epoch = 0
    train_acc1 = 0
    train_acc2 = 0
    mean_pure_ratio1 = 0
    mean_pure_ratio2 = 0
    
    # evaluate models with random weights
    val_acc1, val_acc2 = evaluate(val_loader, clf1, clf2)
    print('Epoch [%d/%d] Val Accuracy on the %s val data: Model1 %.4f %% Model2 %.4f %%' % (
    epoch, args.n_epoch, len(val_dataset), val_acc1, val_acc2))
    
    test_acc1, test_acc2 = evaluate(test_loader, clf1, clf2)
    print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% Model2 %.4f %%' % (
    epoch, args.n_epoch, len(test_dataset), test_acc1, test_acc2))

    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(train_acc2) + ' ' + str(val_acc1) + ' ' 
                     + str(val_acc2) + ' ' + str(test_acc1) + ' ' + str(test_acc2)  + "\n")
    
    # training
    val_acc_list = []
    test_acc_list = []
    for epoch in range(1, args.n_epoch):
        # train models
        clf1.train()
        clf2.train()

        adjust_learning_rate(optimizer1, epoch)
        adjust_learning_rate(optimizer2, epoch)

        train_acc1, train_acc2 = train(train_loader, epoch, clf1, optimizer1, clf2, optimizer2, noise_or_not, args)
        
        val_acc1, val_acc2 = evaluate(val_loader, clf1, clf2)
        
        test_acc1, test_acc2 = evaluate(test_loader, clf1, clf2)
        val_acc_list.append((val_acc1 + val_acc2) / 2)
        test_acc_list.append((test_acc1 + test_acc2) / 2)
        

        
        # save results
        print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% Model2 %.4f %%' % (
        epoch, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(train_acc2) + ' ' + str(val_acc1) + ' ' + str(val_acc2) + ' ' 
                         + str(test_acc1) + ' ' + str(test_acc2) + "\n")
        noise_rate_str = str(args.noise_rate).replace('.', '_')
  
        save_dir = os.path.join("/mnt/ssd1/user/codis", args.dataset, args.noise_type, f"nr{noise_rate_str}")
        os.makedirs(save_dir, exist_ok=True)

        torch.save(clf1.state_dict(), os.path.join(save_dir, f'model1_epoch{epoch}.pth'))
        torch.save(clf2.state_dict(), os.path.join(save_dir, f'model2_epoch{epoch}.pth'))
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
    main(args)