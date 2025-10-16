# -*- coding:utf-8 -*-
import os, datetime, argparse, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import deque
from torchvision import transforms, models
from pathmnist import PATHMNIST, PATHMNIST_VAL, PATHMNIST_TEST
from dermamnist import DERMAMNIST, DermaMNIST_VAL, DermaMNIST_TEST
from drtid import DRTID_val, DRTID, DRTID_test
from kaggledr import kaggledr_val, kaggledr, kaggledr_test
from chexpert import chexpert, chexpert_val, chexpert_test
from bloodmnist import BLOODMNIST, BloodMNIST_VAL, BloodMNIST_TEST
from organcmnist import ORGANCMNIST, OrganCMNIST_VAL, OrganCMNIST_TEST
from utils import set_seed
# ----------------------  ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='dermamnist')             # 
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
set_seed(args.seed)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

save_dir = args.result_dir +'/' +args.dataset+'/ce/'

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str=args.dataset+'_ce_'+args.noise_type+'_'+str(args.noise_rate)+'_'+'seed' + str(args.seed)

txtfile=save_dir+"/"+model_str+".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))
# ----------------------  ----------------------

if args.dataset == 'pathmnist':
    input_channel=3
    num_classes=9
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
                                    nb_classes=num_classes,
                                    device=device)

    val_dataset = DermaMNIST_VAL(
                                    transform=transform,
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    nb_classes=num_classes,
                                    device=device)

    test_dataset = DermaMNIST_TEST(transform=transform)
if args.dataset == 'bloodmnist':
    input_channel=3
    num_classes=8
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

train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=args.batch, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=args.batch, shuffle=False, num_workers=4)

# ----------------------  ----------------------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
optimizer = torch.optim.SGD(params=model.parameters(),
                                         lr=args.lr,
                                         momentum=0.9,
                                         nesterov=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
# criterion = nn.CrossEntropyLoss()
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

# ----------------------  ----------------------

with open(txtfile, 'w') as f:
    f.write('epoch train_acc val_acc test_acc\n')

# ----------------------  ----------------------
def accuracy(loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels, *_ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            total   += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100. * correct / total

# ----------------------  ----------------------
best_val_acc = 0.0
test_at_best = 0.0
best_test_acc = 0.0
last5 = deque(maxlen=5)

for epoch in range(0, args.epochs):
    model.train()
    for imgs, labels, *_ in train_loader:
        imgs = imgs.to(device)
        labels = labels.long().to(device)
        # print("imgs device:", imgs.device)
        # print("labels type:", type(labels))
        # if isinstance(labels, torch.Tensor):
        #     print("labels device:", labels.device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()

    # 
    tr_acc = accuracy(train_loader)
    val_acc = accuracy(val_loader)
    te_acc = accuracy(test_loader)

    scheduler.step()
    # 
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_at_best = te_acc
    best_test_acc = max(best_test_acc, te_acc)
    last5.append(te_acc)

    #  log
    with open(txtfile, 'a') as f:
        f.write(f'{epoch} {tr_acc:.4f} {val_acc:.4f} {te_acc:.4f}\n')
    print(f'Epoch {epoch:03d}/{args.epochs} | train {tr_acc:.2f}%  val {val_acc:.2f}%  test {te_acc:.2f}%')

    noise_rate_str = str(args.noise_rate).replace('.', '_')
    save_dir = os.path.join("/mnt/ssd1/user/ce", args.dataset, args.noise_type, f"nr{noise_rate_str}")
    os.makedirs(save_dir, exist_ok=True)

    # torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch{epoch}.pth'))


# ----------------------  ----------------------
avg_last5 = sum(last5) / len(last5)
print('\n======== Final Report ========')
print(f'1) Val-best test acc : {test_at_best:.2f}%')
print(f'2) Best   test acc   : {best_test_acc:.2f}%')
print(f'3) Last-5 test avg   : {avg_last5:.2f}%')

#  txt
with open(txtfile, 'a') as f:
    f.write('\n# Final Report\n')
    f.write(f'val_best_test_acc {test_at_best:.3f}\n')
    f.write(f'test_max_acc      {best_test_acc:.3f}\n')
    f.write(f'test_last5_avg    {avg_last5:.3f}\n')
