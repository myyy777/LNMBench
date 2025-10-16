from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_drtid as dataloader
import torchvision.models as models
from data.utils import set_seed

import datetime
parser = argparse.ArgumentParser(description='PyTorch drtid Training')
parser.add_argument('--batch_size', default=16, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_mode',  default='instance')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.35, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--noise_rate', default=0.2, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed',type=int, default=0)
parser.add_argument('--gpu', default=3, type=int)
parser.add_argument('--num_classes', default=5, type=int)
parser.add_argument('--dataset', default='drtid', type=str)
parser.add_argument('--adjust_lr', type=int, help='adjust lr', default=1)
args = parser.parse_args()

# torch.cuda.set_device(args.gpuid)
# random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
set_seed(args.seed)
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
# Training
def build_classwise_thresholds(labels, num_classes, t_head=0.8, t_tail=0.4,
                               smoothing=0.0, t_min=0.2, t_max=0.95, eps=1e-8):
    # labels: LongTensor [N]>=0 
    valid = labels >= 0
    counts = torch.bincount(labels[valid], minlength=num_classes).float().clamp_(min=1)

    if smoothing > 0:
        mean = counts.mean()
        counts = (1 - smoothing) * counts + smoothing * mean

    n_min = counts.min()
    n_max = counts.max()
    if torch.isclose(n_max, n_min):
        alpha = torch.zeros_like(counts)
    else:
        alpha = (counts.log() - n_min.log()) / (n_max.log() - n_min.log() + eps)  # [0,1]

    t_class = t_tail + (t_head - t_tail) * alpha          # 
    t_class = t_class.clamp_(min=t_min, max=t_max)        # 
    return t_class  # [C]


def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = labels_x.long()
        labels_x = torch.zeros(batch_size, args.num_classes).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.to(device), inputs_x2.to(device), labels_x.to(device), w_x.to(device)
        inputs_u, inputs_u2 = inputs_u.to(device), inputs_u2.to(device)

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_classes)/args.num_classes
        prior = prior.to(device)      
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.noise_rate, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):     
        inputs = inputs.contiguous().float().to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)      
        if args.noise_mode=='instance':  # penalize confident prediction for asymmetric noise
            # penalty = conf_penalty(outputs)
            # L = loss + penalty
            L = loss      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.noise_rate, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()
def test(data, epoch, net1, net2):
    net1.eval()
    net2.eval()
    correct1 = 0
    correct2 = 0
    correct_mean = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs1 = net1(inputs)
            outputs2 = net2(inputs)

            # 
            _, predicted1 = torch.max(outputs1, 1)
            _, predicted2 = torch.max(outputs2, 1)

            #  ensemblelogits 
            outputs_mean = outputs1 + outputs2
            _, predicted_mean = torch.max(outputs_mean, 1)

            # 
            correct1 += predicted1.eq(targets).sum().item()
            correct2 += predicted2.eq(targets).sum().item()
            correct_mean += predicted_mean.eq(targets).sum().item()
            total += targets.size(0)

    acc1 = 100. * correct1 / total
    acc2 = 100. * correct2 / total
    acc_mean = 100. * correct_mean / total

    print(f"\n| Test Epoch #{epoch} | Net1: {acc1:.2f}%, Net2: {acc2:.2f}%, Ensemble (Mean): {acc_mean:.2f}%\n")
    return acc1, acc2, acc_mean


def eval_train(model, all_loss):
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset)).to(device)
    N = len(eval_loader.dataset)
    train_labels = torch.full((N,), -1, dtype=torch.long, device=device) 

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = CE(outputs, targets)

            

            #  loss  NaN / Inf
            if not torch.isfinite(loss).all():
                print(f"⚠️ 非法 loss @ batch {batch_idx}, 跳过该 batch")
                continue

            for b in range(inputs.size(0)):
                if torch.isfinite(loss[b]):
                    losses[index[b]] = loss[b]

    #  =  loss
    if losses.max() > losses.min():
        losses = (losses - losses.min()) / (losses.max() - losses.min())

    # 
    all_loss.append(losses)

    if args.noise_rate == 0.9 and len(all_loss) >= 5:
        history = torch.stack(all_loss[-5:]).mean(0)
        input_loss = history.reshape(-1, 1).cpu().numpy()
    else:
        input_loss = losses.reshape(-1, 1).cpu().numpy()

    # GMM  fallback
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]

    return prob, all_loss


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, device, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

sce_loss = SCELoss(alpha=1.0, beta=1.0, device=device, num_classes=args.num_classes)
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        labels_x = torch.max(targets_x, 1)[1]
        Lx = sce_loss(outputs_x, labels_x)
        # Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))

        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        
        probs = torch.softmax(outputs, dim=1)
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        penalty = torch.mean(torch.sum(probs * torch.log(probs), dim=1))
        return penalty

# def create_model():
#     model = ResNet18(num_classes=args.num_class)
#     model = model.cuda()
#     return model

os.makedirs('./checkpoint', exist_ok=True)
stats_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.noise_rate,args.noise_mode)+'_stats.txt','w') 
# test_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.noise_rate,args.noise_mode)+'_acc.txt','w')     


  
save_dir = args.result_dir +'/' +args.dataset+'/dividemix/'

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str=args.dataset+'_dividemix_'+args.noise_mode+'_'+str(args.noise_rate)+'_'+'seed' + str(args.seed)+'_'+'adjust_lr'+str(args.adjust_lr)

txtfile=save_dir+"/"+model_str+".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))

with open(txtfile, "a") as myfile:
    myfile.write('epoch:  val_acc1  val_acc2 val_acc_mean  test_acc1  test_acc2 test_mean \n')  

if args.dataset=='drtid':
    warm_up = 8
elif args.dataset=='cifar100':
    warm_up = 30

loader = dataloader.drtid_dataloader(args.dataset,noise_rate=args.noise_rate,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=4,\
    log=stats_log,random_seed=args.seed,num_class=args.num_classes,device=device)
print(args)
print('| Building net')
net1 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
net1.fc = nn.Linear(net1.fc.in_features, args.num_classes)  # PathMNIST  9 
net1.to(device)
net2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
net2.fc = nn.Linear(net2.fc.in_features, args.num_classes)  # PathMNIST  9 
# cudnn.benchmark = True
net2.to(device)
criterion = SemiLoss()
criterion_sce = SCELoss(alpha=0.1, beta=1.0, device=device, num_classes=args.num_classes)
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# for m in net1.modules():
#     if isinstance(m, torch.nn.BatchNorm2d):
#         m.register_buffer('num_batches_tracked', None)
# for m in net2.modules():
#     if isinstance(m, torch.nn.BatchNorm2d):
#         m.register_buffer('num_batches_tracked', None)
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='instance':
    conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks
val_acc_list = []
test_acc_list = []
for epoch in range(args.num_epochs):
 
    lr=args.lr
    # if epoch >= 150:
    #     lr /= 10      
    if args.adjust_lr:
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr       
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr          
    test_loader = loader.run('test')
    val_loader = loader.run('val')
    eval_loader = loader.run('eval_train')   
    
    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader) 
   
    else:         
        prob1,all_loss[0] =eval_train(net1,all_loss[0])   
        prob2,all_loss[1]=eval_train(net2,all_loss[1])    
        
        # t_class  = build_classwise_thresholds(train_labels, num_classes=args.num_classes, t_head=0.6, t_tail=0.4,
        #                        smoothing=0.0)
        # print(t_class)
        # t_sample = t_class[train_labels].cpu().numpy()

               
        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold)      

        if pred1.sum() == 0:
            print("Warning: No clean samples detected for model 1, applying fallback...")
            # fallback top-K 
            k = int(0.01 * len(prob1))  #  5%
            topk_idx = np.argsort(prob1)[-k:]
            pred1[:] = 0
            pred1[topk_idx] = 1

        if pred2.sum() == 0:
            print("Warning: No clean samples detected for model 2, applying fallback...")
            k = int(0.01 * len(prob2))
            topk_idx = np.argsort(prob2)[-k:]
            pred2[:] = 0
            pred2[topk_idx] = 1

        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader) # train net2   

    val_acc1, val_acc2,val_acc_mean = test(val_loader, epoch,net1,net2)
    test_acc1, test_acc2,test_acc_mean = test(test_loader, epoch,net1,net2) 
    val_acc_list.append(val_acc_mean)
    test_acc_list.append(test_acc_mean)


    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(val_acc1) +' '+ str(val_acc2) +' ' + str(val_acc_mean) +' ' +  
                     str(test_acc1) +' ' + str(test_acc2) +' ' + str(test_acc_mean)+ "\n")
        
    noise_rate_str = str(args.noise_rate).replace('.', '_')

    save_dir = os.path.join("/mnt/ssd1/user/dividemix", args.dataset, args.noise_mode, f"nr{noise_rate_str}")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(net1.state_dict(), os.path.join(save_dir, f'model1_epoch{epoch}.pth'))
    torch.save(net2.state_dict(), os.path.join(save_dir, f'model2_epoch{epoch}.pth'))

# 1. val  test acc
best_val_epoch = val_acc_list.index(max(val_acc_list))
test_at_best_val = test_acc_list[best_val_epoch]
# 2. test acc 
best_test_acc = max(test_acc_list)

# 3.  5  epoch  test acc 
last_5_test_acc = sum(test_acc_list[-5:]) / len(test_acc_list[-5:])

print("\n=== Summary ===")

print(f"Best Val Epoch: {best_val_epoch} / {args.num_epochs}")
print(f"Test Acc at Best Val: {test_at_best_val:.4f} %")
print(f"Best Test Acc: {best_test_acc:.4f} %")
print(f"Mean Test Acc of Last 5 Epochs: {last_5_test_acc:.4f} %")

#  txt 
with open(txtfile, "a") as myfile:
    myfile.write("\n=== Summary ===\n")
    myfile.write(f"Best Val Epoch: {best_val_epoch + 1} / {args.num_epochs}\n")
    myfile.write(f"Test Acc at Best Val: {test_at_best_val:.4f} %\n")
    myfile.write(f"Best Test Acc: {best_test_acc:.4f} %\n")
    myfile.write(f"Mean Test Acc of Last 5 Epochs: {last_5_test_acc:.4f} %\n")
                   


