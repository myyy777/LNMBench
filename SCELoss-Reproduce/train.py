import argparse
import torch
import time
import logging
import os
from model import SCEModel, ResNet34
from dataset import DatasetGenerator
from tqdm import tqdm
from utils.utils import AverageMeter, accuracy, count_parameters_in_MB
from train_util import TrainUtil
from loss import SCELoss
import torchvision.transforms as transforms
from pathmnist import PATHMNIST,PATHMNIST_TEST,PATHMNIST_VAL
from dermamnist import DERMAMNIST, DERMAMNIST_VAL, DERMAMNIST_TEST
from bloodmnist import BLOODMNIST, BloodMNIST_VAL, BloodMNIST_TEST
from organcmnist import ORGANCMNIST, OrganCMNIST_VAL, OrganCMNIST_TEST
from kaggledr import kaggledr,kaggledr_val,kaggledr_test
from drtid import DRTID_val, DRTID, DRTID_test
from chexpert import chexpert, chexpert_val,chexpert_test
from utils.utils import set_seed
import torchvision.models as models
import torch.nn as nn
import datetime
# ArgParse
parser = argparse.ArgumentParser(description='SCE Loss')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--l2_reg', type=float, default=5e-4)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--grad_bound', type=float, default=5.0)
parser.add_argument('--train_log_every', type=int, default=100)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--data_path', default='../../datasets', type=str)
parser.add_argument('--checkpoint_path', default='checkpoints', type=str)
parser.add_argument('--data_nums_workers', type=int, default=8)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--noise_rate', type=float, default=0.2, help='noise_rate')
parser.add_argument('--loss', type=str, default='SCE', help='SCE, CE')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha scale')
parser.add_argument('--beta', type=float, default=1.0, help='beta scale')
parser.add_argument('--version', type=str, default='SCE0.0', help='Version')
parser.add_argument('--dataset', type=str, default='dermamnist', help='pathmnist, dermamnist, drtid')
parser.add_argument('--asym', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--noise_type', type=str, default='symmetric', help='')
parser.add_argument('--split_per', type=float, default=0.9)
parser.add_argument('--gpu', type=int, help='ind of gpu', default=0)
parser.add_argument('--adjust_lr', type=int, help='adjust lr', default=1)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
args = parser.parse_args()
GLOBAL_STEP, EVAL_STEP, EVAL_BEST_ACC, EVAL_BEST_ACC_TOP5 = 0, 0, 0, 0
cell_arc = None
set_seed(args.seed)
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

save_dir = args.result_dir +'/' +args.dataset+'/sce/'

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str=args.dataset+'_sce_'+args.noise_type+'_'+str(args.noise_rate)+'_'+'seed' + str(args.seed)+'_'+'adjust_lr'+str(args.adjust_lr)

txtfile=save_dir+"/"+model_str+".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))
if args.dataset == 'organcmnist':
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
    train_dataset =chexpert(
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

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def adjust_weight_decay(model, l2_value):
    conv, fc = [], []
    for name, param in model.named_parameters():
        print(name)
        if not param.requires_grad:
            # frozen weights
            continue
        if 'module.fc1' in name:
            fc.append(param)
        else:
            conv.append(param)
    params = [{'params': conv, 'weight_decay': l2_value}, {'params': fc, 'weight_decay': 0.01}]
    print(fc)
    return params

log_dir = 'logs_train'
os.makedirs(log_dir, exist_ok=True)
log_file_name = os.path.join(
    log_dir,
    f"{args.dataset}_type{args.noise_type}_nr{args.noise_rate}_seed{args.seed}.log"
)
logger = setup_logger(name=args.version, log_file=log_file_name)
for arg in vars(args):
    logger.info("%s: %s" % (arg, getattr(args, arg)))

# if torch.cuda.is_available():
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = True
#     device = torch.device('cuda')
#     logger.info("Using CUDA!")
# else:
#     device = torch.device('cpu')


def log_display(epoch, global_step, time_elapse, **kwargs):
    display = 'epoch=' + str(epoch) + \
              '\tglobal_step=' + str(global_step)
    for key, value in kwargs.items():
        display += '\t' + str(key) + '=%.5f' % value
    display += '\ttime=%.2fit/s' % (1. / time_elapse)
    return display


def model_eval(epoch, fixed_cnn, data_loader):
    global EVAL_STEP
    fixed_cnn.eval()
    valid_loss_meters = AverageMeter()
    valid_acc_meters = AverageMeter()
    valid_acc5_meters = AverageMeter()
    ce_loss = torch.nn.CrossEntropyLoss()

    for images, labels, _ in tqdm(data_loader):
        start = time.time()
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            pred = fixed_cnn(images)
            loss = ce_loss(pred, labels)
            acc, acc5 = accuracy(pred, labels, topk=(1, 5))

        valid_loss_meters.update(loss.item())
        valid_acc_meters.update(acc.item())
        valid_acc5_meters.update(acc5.item())
        end = time.time()

        EVAL_STEP += 1
        if EVAL_STEP % args.train_log_every == 0:
            display = log_display(epoch=epoch,
                                  global_step=GLOBAL_STEP,
                                  time_elapse=end-start,
                                  loss=loss.item(),
                                  test_loss_avg=valid_loss_meters.avg,
                                  acc=acc.item(),
                                  test_acc_avg=valid_acc_meters.avg,
                                  test_acc_top5_avg=valid_acc5_meters.avg)
            logger.info(display)
    display = log_display(epoch=epoch,
                          global_step=GLOBAL_STEP,
                          time_elapse=end-start,
                          loss=loss.item(),
                          test_loss_avg=valid_loss_meters.avg,
                          acc=acc.item(),
                          test_acc_avg=valid_acc_meters.avg,
                          test_acc_top5_avg=valid_acc5_meters.avg)
    logger.info(display)
    return valid_acc_meters.avg, valid_acc5_meters.avg


def train_fixed(starting_epoch, train_loader, val_loader, test_loader,  fixed_cnn, criterion, fixed_cnn_optmizer, fixed_cnn_scheduler, utilHelper):
    global GLOBAL_STEP, reduction_arc, cell_arc, EVAL_BEST_ACC, EVAL_STEP, EVAL_BEST_ACC_TOP5
    val_acc_list = []
    test_acc_list = []
    for epoch in tqdm(range(starting_epoch, args.epoch)):
        logger.info("=" * 20 + "Training" + "=" * 20)
        fixed_cnn.train()
        train_loss_meters = AverageMeter()
        train_acc_meters = AverageMeter()
        train_acc5_meters = AverageMeter()

        for images, labels, _ in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device).long() 
            start = time.time()
            fixed_cnn.zero_grad()
            fixed_cnn_optmizer.zero_grad()
            pred = fixed_cnn(images)
            loss = criterion(pred, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(fixed_cnn.parameters(), args.grad_bound)
            fixed_cnn_optmizer.step()
            acc, acc5 = accuracy(pred, labels, topk=(1, 5))
            acc_sum = torch.sum((torch.max(pred, 1)[1] == labels).type(torch.float))
            total = pred.shape[0]
            acc = acc_sum / total

            train_loss_meters.update(loss.item())
            train_acc_meters.update(acc.item())
            train_acc5_meters.update(acc5.item())

            end = time.time()

            GLOBAL_STEP += 1
            if GLOBAL_STEP % args.train_log_every == 0:
                lr = fixed_cnn_optmizer.param_groups[0]['lr']
                display = log_display(epoch=epoch,
                                      global_step=GLOBAL_STEP,
                                      time_elapse=end-start,
                                      loss=loss.item(),
                                      loss_avg=train_loss_meters.avg,
                                      acc=acc.item(),
                                      acc_top1_avg=train_acc_meters.avg,
                                      acc_top5_avg=train_acc5_meters.avg,
                                      lr=lr,
                                      gn=grad_norm)
                logger.info(display)

        fixed_cnn_scheduler.step()
        logger.info("="*20 + "Eval" + "="*20)
        val_acc, val_acc5 = model_eval(epoch, fixed_cnn, val_loader)
        curr_acc, curr_acc5 = model_eval(epoch, fixed_cnn, test_loader)
        val_acc_list.append(val_acc)
        test_acc_list.append(curr_acc)
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  + str(val_acc) 
                        +' ' + str(curr_acc) + "\n")
        noise_rate_str = str(args.noise_rate).replace('.', '_')
        save_dir = os.path.join("/mnt/ssd1/user/sce", args.dataset, args.noise_type, f"nr{noise_rate_str}")

        os.makedirs(save_dir, exist_ok=True)

        torch.save(fixed_cnn.state_dict(), os.path.join(save_dir, f'model_epoch{epoch}.pth'))
     


        logger.info("curr_acc\t%.4f" % curr_acc)
        logger.info("BEST_ACC\t%.4f" % EVAL_BEST_ACC)
        logger.info("curr_acc_top5\t%.4f" % curr_acc5)
        logger.info("BEST_ACC_top5\t%.4f" % EVAL_BEST_ACC_TOP5)
        payload = '=' * 10 + '\n'
        payload = payload + ("curr_acc: %.4f\n best_acc: %.4f\n" % (curr_acc, EVAL_BEST_ACC))
        payload = payload + ("curr_acc_top5: %.4f\n best_acc_top5: %.4f\n" % (curr_acc5, EVAL_BEST_ACC_TOP5))
        EVAL_BEST_ACC = max(curr_acc, EVAL_BEST_ACC)
        EVAL_BEST_ACC_TOP5 = max(curr_acc5, EVAL_BEST_ACC_TOP5)
        logger.info("Model Saved!\n")
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

    return


def train():
    global GLOBAL_STEP, reduction_arc, cell_arc
    # Dataset
    # dataset = DatasetGenerator(batchSize=args.batch_size,
    #                            dataPath=args.data_path,
    #                            numOfWorkers=args.data_nums_workers,
    #                            noise_rate=args.nr,
    #                            asym=args.asym,
    #                            seed=args.seed,
    #                            dataset_type=args.dataset_type)
    # dataLoader = dataset.getDataLoader()
    with open(txtfile, "a") as myfile:
        myfile.write('epoch:  val_acc test_acc  \n')
    
    print('loading dataset...') 
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
    if args.dataset == 'pathmnist':
        num_classes = 9
        args.epoch = 50
    elif args.dataset == 'drtid':
        num_classes = 5
        args.epoch = 50
    elif args.dataset == 'dermamnist':
        num_classes = 7
        args.epoch = 50
    elif args.dataset == 'bloodmnist':
        num_classes = 8
        args.epoch = 50
    elif args.dataset == 'organcmnist':
        num_classes = 11
        args.epoch = 50
    elif args.dataset == 'kaggledr':
        num_classes = 5
        args.epoch = 50
    elif args.dataset == 'chexpert':
        num_classes = 5
        args.epoch = 50
    # elif args.dataset_type == 'cifar10':
    #     num_classes = 10
    #     args.epoch = 120
    #     fixed_cnn = SCEModel()
    else:
        raise('Unimplemented')
    print(args)
    fixed_cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    fixed_cnn.fc = nn.Linear(fixed_cnn.fc.in_features, num_classes) 
    if args.loss == 'SCE':
        criterion = SCELoss(alpha=args.alpha, beta=args.beta, device=device, num_classes=num_classes)
    elif args.loss == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        logger.info("Unknown loss")

    logger.info(criterion.__class__.__name__)
    logger.info("Number of Trainable Parameters %.4f" % count_parameters_in_MB(fixed_cnn))
    # fixed_cnn = torch.nn.DataParallel(fixed_cnn)
    fixed_cnn.to(device)

    # fixed_cnn_optmizer = torch.optim.SGD(params=adjust_weight_decay(fixed_cnn, args.l2_reg),
    #                                      lr=args.lr,
    #                                      momentum=0.9,
    #                                      nesterov=True)
    fixed_cnn_optmizer = torch.optim.SGD(params=fixed_cnn.parameters(),
                                         lr=args.lr,
                                         momentum=0.9,
                                         nesterov=True)
    fixed_cnn_scheduler = torch.optim.lr_scheduler.MultiStepLR(fixed_cnn_optmizer, milestones=[10, 20], gamma=0.1)

    utilHelper = TrainUtil(checkpoint_path=args.checkpoint_path, version=args.version)
    starting_epoch = 0
    train_fixed(starting_epoch, train_loader, val_loader, test_loader, fixed_cnn, criterion, fixed_cnn_optmizer, fixed_cnn_scheduler, utilHelper)


if __name__ == '__main__':
    train()
