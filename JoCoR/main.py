# -*- coding:utf-8 -*-
import os
import torch
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
import argparse, sys
import datetime
from algorithm.jocor import JoCoR
from data.pathmnist import PATHMNIST, PATHMNIST_TEST,PATHMNIST_VAL
from data.dermamnist import DERMAMNIST, DERMAMNIST_VAL,DERMAMNIST_TEST
from data.bloodmnist import BLOODMNIST, BloodMNIST_VAL,BloodMNIST_TEST
from data.organcmnist import ORGANCMNIST,OrganCMNIST_TEST,OrganCMNIST_VAL
from data.drtid import DRTID_val, DRTID, DRTID_test
from data.kaggledr import kaggledr, kaggledr_val,kaggledr_test
from data.chexpert import chexpert, chexpert_val, chexpert_test
from data.utils import set_seed


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results/')
parser.add_argument('--log_dir', type = str, help = 'dir to save result txt files', default = 'logs_clean/')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.2)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='dermamnist')
parser.add_argument('--n_epoch', type=int, default=50)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=200)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=20)
parser.add_argument('--co_lambda', type=float, default=0.1)
parser.add_argument('--model_type', type=str, help='[mlp,cnn]', default='cnn')
parser.add_argument('--save_model', type=str, help='save model?', default="False")
parser.add_argument('--save_result', type=str, help='save result?', default="True")
parser.add_argument('--split_per', type=float, default=0.9)
parser.add_argument('--gpu', type=int, help='ind of gpu', default=2)
parser.add_argument('--adjust_lr', type=int, help='adjust lr', default=1)
parser.add_argument('--batch', type=int, help='adjust lr', default=128)

args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
# Seed
set_seed(args.seed)

# torch.manual_seed(args.seed)
# if args.gpu is not None:
#     device = torch.device('cuda:{}'.format(args.gpu))
#     torch.cuda.manual_seed(args.seed)

# else:
#     device = torch.device('cpu')
#     torch.manual_seed(args.seed)

# Hyper Parameters
batch_size = args.batch
learning_rate = args.lr

# load dataset

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
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate

save_dir = args.result_dir +'/' +args.dataset+'/jocor/'
log_dir = args.log_dir +'/' +args.dataset+'/jocor/'

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)
if not os.path.exists(log_dir):
    os.system('mkdir -p %s' % log_dir)
model_str=args.dataset+'_jocor_'+args.noise_type+'_'+str(args.noise_rate)+'_'+'seed' + str(args.seed)+'_'+'adjust_lr'+str(args.adjust_lr)
log_str=args.dataset+'_jocor_'+args.noise_type+'_'+str(args.noise_rate)+'_'+'seed' + str(args.seed)
txtfile=save_dir+"/"+model_str+".txt"
logfile = log_dir+"/"+log_str+".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))
if os.path.exists(logfile):
    os.system('mv %s %s' % (logfile, logfile+".bak-%s" % nowTime))

def main():
    # Data Loader (Input Pipeline)
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
    print(args)
    print('building model...')
    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc1 train_acc2 val_acc1 val_acc2 val_acc_mean test_acc1 test_acc2 test_acc_mean pure_ratio1 pure_ratio2\n')
    model = JoCoR(args, train_dataset, device, input_channel, num_classes)

    epoch = 0
    train_acc1 = 0
    train_acc2 = 0
    

    # evaluate models with random weights
    val_acc1, val_acc2, val_acc_mean = model.evaluate(val_loader)
    test_acc1, test_acc2, test_acc_mean = model.evaluate(test_loader)

    print(
        'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f ' % (
            epoch, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) 
                     +' ' + str(val_acc1) +' '  + str(val_acc2) +' '+ str(val_acc_mean) +' '+ str(test_acc1) + " " 
                     + str(test_acc2) +' '+ str(test_acc_mean)+ "\n")

    acc_list = []
    # training
    val_acc_list = []
    test_acc_list = []
    for epoch in range(1, args.n_epoch):
        # train models
        train_acc1, train_acc2 = model.train(train_loader, epoch,logfile)

        # evaluate models
        val_acc1, val_acc2, val_acc_mean = model.evaluate(val_loader)
        test_acc1, test_acc2, test_acc_mean = model.evaluate(test_loader)
        val_acc_list.append((val_acc1 + val_acc2) / 2)
        test_acc_list.append((test_acc1 + test_acc2) / 2)

        # save results
        # if pure_ratio_1_list is None or len(pure_ratio_1_list) == 0:
        #     print(
        #         'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f' % (
        #             epoch, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
        # else:
        #     # save results
        #     mean_pure_ratio1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
        #     mean_pure_ratio2 = sum(pure_ratio_2_list) / len(pure_ratio_2_list)
        print(
            'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%' % (
                epoch, args.n_epoch, len(test_dataset), test_acc1, test_acc2
                ))
        with open(txtfile, "a") as myfile:
                    myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) 
                                +' ' + str(val_acc1) +' '  + str(val_acc2) +' '+ str(val_acc_mean) +' ' + str(test_acc1) 
                                + " " + str(test_acc2)+ ' '  +str(test_acc_mean) + "\n")
        # noise_rate_str = str(args.noise_rate).replace('.', '_')            
        # save_dir = os.path.join("/mnt/ssd1/user/jocor", args.dataset, args.noise_type ,f"nr{noise_rate_str}")
        # os.makedirs(save_dir, exist_ok=True)
        # torch.save(model.model1.state_dict(),
        #             os.path.join(save_dir, f"model1_epoch{epoch}.pth"))
        # torch.save(model.model2.state_dict(),
        #             os.path.join(save_dir, f"model2_epoch{epoch}.pth"))
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
        myfile.write(f"Best Val Epoch: {best_val_epoch} / {args.n_epoch}\n")
        myfile.write(f"Test Acc at Best Val: {test_at_best_val:.4f} %\n")
        myfile.write(f"Best Test Acc: {best_test_acc:.4f} %\n")
        myfile.write(f"Mean Test Acc of Last 5 Epochs: {last_5_test_acc:.4f} %\n")
    #     if epoch >= 190:
    #         acc_list.extend([test_acc1, test_acc2])

    # avg_acc = sum(acc_list)/len(acc_list)
    # print(len(acc_list))
    # print("the average acc in last 10 epochs: {}".format(str(avg_acc)))


if __name__ == '__main__':
    main()
