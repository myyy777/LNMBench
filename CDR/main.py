import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

from model import LeNet

from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
# import torchvision.models as tv_models
import torch.optim as optim
import argparse, sys
import numpy as np
import datetime
import data_load
import resnet
import tools
from tools import set_seed
import torchvision.models as models
# import torch.nn as nn
import warnings
from transformer import transform_train, transform_test,transform_target
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=0, help="No.")
parser.add_argument('--d', type=str, default='output', help="description")
parser.add_argument('--p', type=int, default=0, help="print")
parser.add_argument('--c', type=int, default=10, help="class")
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results/')
parser.add_argument('--noise_rate', type=float, help='overall corruption rate, should be less than 1', default=0.4)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')
parser.add_argument('--num_gradual', type=int, default=10, help='how many epochs for linear drop rate')
parser.add_argument('--dataset', type=str, help='mnist, fmnist, cifar10, cifar100', default='cifar10')
parser.add_argument('--n_epoch', type=int, default=50)
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=350)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--model_type', type=str, help='[ce, ours]', default='cdr')
parser.add_argument('--fr_type', type=str, help='forget rate type', default='type_1')
parser.add_argument('--gpu', type=int, help='ind of gpu', default=3)
parser.add_argument('--weight_decay', type=float, help='l2', default=1e-3)
parser.add_argument('--momentum', type=int, help='momentum', default=0.9)
parser.add_argument('--batch_size', type=int, help='batch_size', default=128)
parser.add_argument('--train_len', type=int, help='the number of training data', default=54000)
parser.add_argument('--adjust_lr', type=int, help='adjust_lr', default=1)
args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(args)
# Seed
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
set_seed(args.seed)
# Hyper Parameters
learning_rate = args.lr

# load dataset
def load_data(args):
    if args.dataset == 'organcmnist':
            args.channel = 3
            args.num_classes = 11
            args.feature_size = 3 * 224 * 224
            args.n_epoch = 50
            args.batch_size = 128
            args.num_gradual = 10
            args.train_len = 12975
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std =[0.229, 0.224, 0.225])
            ])

            train_dataset = data_load.organcmnist_dataset(
                                            transform=transform,
                                            target_transform=transform_target,
                                            
                                            noise_type=args.noise_type,
                                            noise_rate=args.noise_rate,
                                            
                                            random_seed=args.seed,
                                            device=device)

            val_dataset = data_load.OrganCMNIST_VAL(
                                            transform=transform,
                                            target_transform=transform_target,
                                            
                                            noise_type=args.noise_type,
                                            noise_rate=args.noise_rate,
                                            
                                            random_seed=args.seed,
                                            device=device)

            test_dataset = data_load.organcmnist_test_dataset(
                                            transform=transform,
                                            target_transform=transform_target)

    if args.dataset == 'chexpert':
            args.channel = 3
            args.num_classes = 5
            args.feature_size = 3 * 224 * 224
            args.n_epoch = 50
            args.batch_size = 128
            args.num_gradual = 10
            args.train_len = 64208

            train_dataset = data_load.chexpert(
                                            transform=transform_train(args.dataset),
                                            target_transform=transform_target,
                                            
                                            noise_type=args.noise_type,
                                            noise_rate=args.noise_rate,
                                            
                                            random_seed=args.seed,
                                            device=device)

            val_dataset = data_load.chexpert_val(
                                            transform=transform_train(args.dataset),
                                            target_transform=transform_target,
                                            
                                            noise_type=args.noise_type,
                                            noise_rate=args.noise_rate,
                                            
                                            random_seed=args.seed,
                                            device=device)

            test_dataset = data_load.chexpert_test(
                                            transform=transform_test(args.dataset),
                                            target_transform=transform_target)


    if args.dataset == 'drtid':
        args.channel = 3
        args.num_classes = 5
        args.feature_size = 3 * 512 * 512
        args.n_epoch = 50
        args.batch_size = 16
        args.num_gradual = 10
        args.train_len = 1600

        train_dataset = data_load.DRTID(
                                        transform=transform_train(args.dataset),
                                        target_transform=transform_target,
                                        
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        
                                        random_seed=args.seed,
                                        device=device)

        val_dataset = data_load.DRTID_val(
                                        transform=transform_train(args.dataset),
                                        target_transform=transform_target,
                                        
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        
                                        random_seed=args.seed,
                                        device=device)

        test_dataset = data_load.DRTID_test(
                                        transform=transform_test(args.dataset),
                                        target_transform=transform_target)
    if args.dataset == 'kaggledr':
            args.channel = 3
            args.num_classes = 5
            args.feature_size = 3 * 512 * 512
            args.n_epoch = 50
            args.batch_size = 16
            args.num_gradual = 10
            args.train_len = 70247

            train_dataset = data_load.kaggledr(
                                            transform=transform_train(args.dataset),
                                            target_transform=transform_target,
                                            
                                            noise_type=args.noise_type,
                                            noise_rate=args.noise_rate,
                                            
                                            random_seed=args.seed,
                                            device=device)

            val_dataset = data_load.kaggledr_val(
                                            transform=transform_train(args.dataset),
                                            target_transform=transform_target,
                                            
                                            noise_type=args.noise_type,
                                            noise_rate=args.noise_rate,
                                            
                                            random_seed=args.seed,
                                            device=device)

            test_dataset = data_load.kaggledr_test(
                                            transform=transform_test(args.dataset),
                                            target_transform=transform_target)

    if args.dataset == 'dermamnist':
        args.channel = 3
        args.num_classes = 7
        args.feature_size = 3 * 224 * 224
        args.n_epoch = 50
        args.batch_size = 128
        args.num_gradual = 10
        args.train_len = 7007
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std =[0.229, 0.224, 0.225])
    ])

        train_dataset = data_load.dermamnist_dataset(
                                        transform=transform,
                                        target_transform=tools.transform_target,
                                        
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        
                                        random_seed=args.seed,
                                        device=device)

        val_dataset = data_load.DERMAMNIST_VAL(
                                        transform=transform,
                                        target_transform=tools.transform_target,
                                        
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        
                                        random_seed=args.seed,
                                        device=device)

        test_dataset = data_load.dermamnist_test_dataset(
                                        transform=transform,
                                        target_transform=tools.transform_target)

    if args.dataset == 'bloodmnist':
        args.channel = 3
        args.num_classes = 8
        args.feature_size = 3 * 224 * 224
        args.n_epoch = 50
        args.batch_size = 128
        args.num_gradual = 10
        args.train_len = 11959
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std =[0.229, 0.224, 0.225])
    ])

        train_dataset = data_load.bloodmnist_dataset(
                                        transform=transform,
                                        target_transform=tools.transform_target,
                                        
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        
                                        random_seed=args.seed,
                                        device=device)

        val_dataset = data_load.BloodMNIST_VAL(
                                        transform=transform,
                                        target_transform=tools.transform_target,
                                        
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        
                                        random_seed=args.seed,
                                        device=device)

        test_dataset = data_load.bloodmnist_test_dataset(
                                        transform=transform,
                                        target_transform=tools.transform_target)

    if args.dataset == 'pathmnist':
        args.channel = 3
        args.num_classes = 9
        args.feature_size = 3 * 224 * 224
        args.n_epoch = 50
        args.batch_size = 128
        args.num_gradual = 10
        args.train_len = 89996
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std =[0.229, 0.224, 0.225])
    ])

        train_dataset = data_load.pathmnist_dataset(
                                        transform=transform,
                                        target_transform=tools.transform_target,
                                        
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        
                                        random_seed=args.seed,
                                        device=device)

        val_dataset = data_load.PATHMNIST_VAL(
                                        transform=transform,
                                        target_transform=tools.transform_target,
                                        
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        
                                        random_seed=args.seed,
                                        device=device)

        test_dataset = data_load.pathmnist_test_dataset(
                                        transform=transform,
                                        target_transform=tools.transform_target)

    return train_dataset, val_dataset, test_dataset



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


def train_one_step(net, data, label, optimizer, criterion, nonzero_ratio, clip):
    net.train()
    pred = net(data)
    loss = criterion(pred, label)
    loss.backward()
    
    to_concat_g = []
    to_concat_v = []
    for name, param in net.named_parameters():
        if param.dim() in [2, 4]:
            to_concat_g.append(param.grad.data.view(-1))
            to_concat_v.append(param.data.view(-1))
    all_g = torch.cat(to_concat_g)
    all_v = torch.cat(to_concat_v)
    metric = torch.abs(all_g * all_v)
    num_params = all_v.size(0)
    nz = int(nonzero_ratio * num_params)
    top_values, _ = torch.topk(metric, nz)
    thresh = top_values[-1]

    for name, param in net.named_parameters():
        if param.dim() in [2, 4]:
            mask = (torch.abs(param.data * param.grad.data) >= thresh).type(torch.cuda.FloatTensor)
            mask = mask * clip
            param.grad.data = mask * param.grad.data

    optimizer.step()
    optimizer.zero_grad()
    acc = accuracy(pred, label, topk=(1,))

    return float(acc[0]), loss


def train(train_loader, epoch, model1, optimizer1, args):
    model1.train()
    train_total=0
    train_correct=0
    total_loss = 0.0
    total_samples = 0
    clip_narry = np.linspace(1-args.noise_rate, 1, num=args.num_gradual)
    clip_narry = clip_narry[::-1]
    if epoch < args.num_gradual:
        clip = clip_narry[epoch]
    else:
        clip = (1 - args.noise_rate)
    for i, (data, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        data = data.to(device)
        labels = labels.to(device)
        # Forward + Backward + Optimize
        logits1=model1(data)
        prec1,  = accuracy(logits1, labels, topk=(1, ))
        train_total+=1
        train_correct+=prec1
        # Loss transfer 

        prec1, loss = train_one_step(model1, data, labels, optimizer1, nn.CrossEntropyLoss(), clip, clip)
        total_loss += loss.item() * labels.size(0)  #  loss
        total_samples += labels.size(0) 
        avg_epoch_loss = total_loss / total_samples 
       
        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Loss1: %.4f' 
                  %(epoch+1, args.n_epoch, i+1, args.train_len//args.batch_size, prec1, loss.item()))
        
      
    train_acc1=float(train_correct)/float(train_total)
    return train_acc1, avg_epoch_loss


# Evaluate the Model
def evaluate(test_loader, model1):
    
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    with torch.no_grad():
        for data, labels, _ in test_loader:
            data = data.to(device)
            logits1 = model1(data)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels.long()).sum()

        acc1 = 100 * float(correct1) / float(total1)

    return acc1


def main(args):
    # Data Loader (Input Pipeline)
    
    save_dir = args.result_dir + '/' + args.dataset + '/%s/' % args.model_type

    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % save_dir)

    model_str = args.dataset + '_%s_' % args.model_type + args.noise_type + '_' + str(args.noise_rate) + '_' + str(args.seed)
    txtfile = save_dir + "/" + model_str + ".txt"
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    if os.path.exists(txtfile):
        os.system('mv %s %s' % (txtfile, txtfile + ".bak-%s" % nowTime))
    
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_dataset, val_dataset, test_dataset = load_data(args)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              
                                              shuffle=False)
    
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                             
                                              shuffle=False)
    
    
    
    # Define models
    print('building model...')
    
   
        #  torchvision  ResNet50
    clf1 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    clf1.fc = nn.Linear(clf1.fc.in_features, args.num_classes) 
    optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    if args.adjust_lr:
        scheduler1 = MultiStepLR(optimizer1, milestones=[20, 40], gamma=0.1)

    clf1.to(device)
    
    with open(txtfile, "a") as myfile:
        myfile.write('epoch train_acc1 loss val_acc1 test_acc1\n')

    epoch = 0
    train_acc1 = 0
    loss = 0
   
    
    # evaluate models with random weights
    val_acc1 = evaluate(val_loader, clf1)
    print('Epoch [%d/%d] Val Accuracy on the %s val data: Model1 %.4f %%' % (
    epoch + 1, args.n_epoch, len(val_dataset), val_acc1))
    
    test_acc1 = evaluate(test_loader, clf1)
    print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %%' % (
    epoch + 1, args.n_epoch, len(test_dataset), test_acc1))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ' ' + str(train_acc1)+' ' +str(loss) + ' ' + str(val_acc1) + ' ' + str(test_acc1) + "\n")
    val_acc_list = []
    test_acc_list = []  
    
    for epoch in range(0, args.n_epoch):
        if args.adjust_lr:
            scheduler1.step()
        print(optimizer1.state_dict()['param_groups'][0]['lr'])
        clf1.train()
        
        train_acc1,loss = train(train_loader, epoch, clf1, optimizer1, args)
        val_acc1 = evaluate(val_loader, clf1)
        val_acc_list.append(val_acc1)
        test_acc1 = evaluate(test_loader, clf1)
        test_acc_list.append(test_acc1)
        
        # save results
        print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% ' % (
        epoch + 1, args.n_epoch, len(test_dataset), test_acc1))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ' ' + str(train_acc1)+' ' +str(loss) + ' ' + str(val_acc1) + ' ' + str(test_acc1) + "\n")

        noise_rate_str = str(args.noise_rate).replace('.', '_')
  
        save_dir = os.path.join("/mnt/ssd1/user/CDR", args.dataset, args.noise_type, f"nr{noise_rate_str}")
        os.makedirs(save_dir, exist_ok=True)

        torch.save(clf1.state_dict(), os.path.join(save_dir, f'model_epoch{epoch}.pth'))
   
# 1. val  test acc
    best_val_epoch = val_acc_list.index(max(val_acc_list))
    test_at_best_val = test_acc_list[best_val_epoch]
    # 2. test acc 
    best_test_acc = max(test_acc_list)

    # 3.  5  epoch  test acc 
    last_5_test_acc = sum(test_acc_list[-5:]) / len(test_acc_list[-5:])
    
    print("\n=== Summary ===")
    print(f"Best Val Epoch: {best_val_epoch}")
    print(f"Test Acc at Best Val: {test_at_best_val:.4f} %")
    print(f"Best Test Acc: {best_test_acc:.4f} %")
    print(f"Mean Test Acc of Last 5 Epochs: {last_5_test_acc:.4f} %")

    #  txt 
    with open(txtfile, "a") as myfile:
        myfile.write("\n=== Summary ===\n")
        myfile.write(f"Best Val Epoch: {best_val_epoch}\n")
        myfile.write(f"Test Acc at Best Val: {test_at_best_val:.4f} %\n")
        myfile.write(f"Best Test Acc: {best_test_acc:.4f} %\n")
        myfile.write(f"Mean Test Acc of Last 5 Epochs: {last_5_test_acc:.4f} %\n")



if __name__ == '__main__':
    main(args)
    
