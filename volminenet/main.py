import tools
import data_load
import argparse
from models import *
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer import transform_train, transform_test,transform_target
from torch.optim.lr_scheduler import MultiStepLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from data_load import Chexpert,Chexpert_test,Chexpert_val,DRTID,DRTID_test,DRTID_val,Kaggledr,Kaggledr_test,Kaggledr_val
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='initial learning rate', default=0.01)
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'pathmnist')
parser.add_argument('--n_epoch', type=int, default=50)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--noise_type', type=str, default='symmetric')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=1e-4)
parser.add_argument('--lam', type = float, default =0.0001)
parser.add_argument('--gpu', type=int, help='ind of gpu', default=1)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--matrix_dir', type=str, help='dir to save estimated matrix', default='matrix')
args = parser.parse_args()
np.set_printoptions(precision=4,suppress=True)
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
tools.set_seed(args.seed)




if args.dataset == 'pathmnist':
    args.n_epoch = 50

    args.num_classes = 9
    milestones = [15,30]

    train_data = data_load.PATHMNIST(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                 

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)
    val_data = data_load.PATHMNIST_VAL(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                  
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)
    test_data = data_load.PATHMNIST_TEST(target_transform = transform_target,transform=transform_test(args.dataset))
    

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    trans = sig_t(device, args.num_classes)
    optimizer_trans = optim.SGD(trans.parameters(), lr=args.lr, weight_decay=0, momentum=0.9)

if args.dataset == 'dermamnist':
    args.n_epoch = 50

    args.num_classes = 7
    milestones = [15,30]

    train_data = data_load.DERMAMNIST(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                 

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)
    val_data = data_load.DERMAMNIST_VAL(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                  
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)
    test_data = data_load.DERMAMNIST_TEST(target_transform = transform_target,transform=transform_test(args.dataset))
    

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    trans = sig_t(device, args.num_classes)
    optimizer_trans = optim.SGD(trans.parameters(), lr=args.lr, weight_decay=0, momentum=0.9)

if args.dataset == 'bloodmnist':
    args.n_epoch = 50

    args.num_classes = 8
    milestones = [15,30]

    train_data = data_load.BLOODMNIST(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                 

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)
    val_data = data_load.BLOODMNIST_VAL(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                  
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)
    test_data = data_load.BLOODMNIST_TEST(target_transform = transform_target,transform=transform_test(args.dataset))
    

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    trans = sig_t(device, args.num_classes)
    optimizer_trans = optim.SGD(trans.parameters(), lr=args.lr, weight_decay=0, momentum=0.9)


if args.dataset == 'drtid':
    args.n_epoch = 50

    args.num_classes = 5
    milestones = [15,30]


    train_data =DRTID(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                 

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)

    val_data = DRTID_val(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                  
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)

    test_data = DRTID_test(target_transform = transform_target,transform=transform_test(args.dataset))

  
    
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    trans = sig_t(device, args.num_classes)
    optimizer_trans = optim.SGD(trans.parameters(), lr=args.lr, weight_decay=0, momentum=0.9)
    
if args.dataset == 'kaggledr':
    args.n_epoch = 50
    
    args.num_classes = 5
    milestones = [15,30]
    

    train_data =Kaggledr(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                 

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)

    val_data = Kaggledr_val(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                  
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)

    test_data = Kaggledr_test(target_transform = transform_target,transform=transform_test(args.dataset))

  
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    trans = sig_t(device, args.num_classes)
    optimizer_trans = optim.SGD(trans.parameters(), lr=args.lr, weight_decay=0, momentum=0.9)

if args.dataset == 'chexpert':
    args.n_epoch = 50

    args.num_classes = 5
    milestones = [15,30]
 

    train_data =Chexpert(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                 

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)

    val_data = Chexpert_val(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                  
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)

    test_data = Chexpert_test(target_transform = transform_target,transform=transform_test(args.dataset))

  
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    trans = sig_t(device, args.num_classes)
    optimizer_trans = optim.SGD(trans.parameters(), lr=args.lr, weight_decay=0, momentum=0.9)
# save_dir, model_dir, matrix_dir, logs = create_dir(args)

# print(args, file=logs, flush=True)
if args.dataset == 'organcmnist':
    args.n_epoch = 50

    args.num_classes = 11
    milestones = [15,30]

    train_data = data_load.ORGANCMNIST(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                 

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)
    val_data = data_load.ORGANCMNIST_VAL(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                  
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)
    test_data = data_load.ORGANCMNIST_TEST(target_transform = transform_target,transform=transform_test(args.dataset))
    

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    trans = sig_t(device, args.num_classes)
    optimizer_trans = optim.SGD(trans.parameters(), lr=args.lr, weight_decay=0, momentum=0.9)


#optimizer and StepLR
optimizer_es = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
scheduler1 = MultiStepLR(optimizer_es, milestones=milestones, gamma=0.1)
scheduler2 = MultiStepLR(optimizer_trans, milestones=milestones, gamma=0.1)


#data_loader
train_loader = DataLoader(dataset=train_data, 
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=4,
                          drop_last=True)

val_loader = DataLoader(dataset=val_data,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=4
                        )

test_loader = DataLoader(dataset=test_data,
                         batch_size=args.batch_size,
                         num_workers=4,
                         shuffle=False)


loss_func_ce = F.nll_loss


import datetime
save_dir = args.result_dir +'/' +args.dataset
if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str=args.dataset+'_vol_'+args.noise_type+'_'+str(args.noise_rate)+'_'+'seed' + str(args.seed)

txtfile=save_dir+"/"+model_str+".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))

with open(txtfile, "a") as myfile:

    myfile.write('epoch: train_acc val_acc test_acc es_error  \n')

matrix_dir = args.matrix_dir + '/' + args.dataset + '/'+ args.noise_type + '/' + 'noise_rate_%s'%(args.noise_rate)

if not os.path.exists(matrix_dir):
    os.system('mkdir -p %s'%(matrix_dir))
#cuda
if torch.cuda.is_available:
    model = model.to(device)
    trans = trans.to(device)



val_loss_list = []
val_acc_list = []
test_acc_list = []

# print(train_data.t)


t = trans()
est_T = t.detach().cpu().numpy()
print(est_T)


# estimate_error = tools.error(est_T, train_data.t)

# print('Estimation Error: {:.2f}'.format(estimate_error))

def main():


    for epoch in range(args.n_epoch):

        print('epoch {}'.format(epoch ))
        model.train()
        trans.train()

        train_loss = 0.
        train_vol_loss =0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
        eval_loss = 0.
        eval_acc = 0.

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer_es.zero_grad()
            optimizer_trans.zero_grad()


            clean = model(batch_x)

            t = trans()

            out = torch.mm(clean, t)

            # vol_loss = t.slogdet().logabsdet
            sign, logabsdet = torch.linalg.slogdet(t)
            vol_loss = logabsdet

            # ce_loss = loss_func_ce(out.log(), batch_y.long())
            ce_loss = F.cross_entropy(out, batch_y.long())
            loss = ce_loss + args.lam * vol_loss

            train_loss += loss.item()
            train_vol_loss += vol_loss.item()

            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()


            loss.backward()
            optimizer_es.step()
            optimizer_trans.step()

        print('Train Loss: {:.6f}, Vol_loss: {:.6f}  Acc: {:.6f}'.format(train_loss / (len(train_data))*args.batch_size, train_vol_loss / (len(train_data))*args.batch_size, train_acc / (len(train_data))))

        scheduler1.step()
        scheduler2.step()

        with torch.no_grad():
            model.eval()
            trans.eval()
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                clean = model(batch_x)
                t = trans()

                out = torch.mm(clean, t)
                # loss = loss_func_ce(out.log(), batch_y.long())
                loss = F.cross_entropy(out, batch_y.long())
                val_loss += loss.item()
                pred = torch.max(out, 1)[1]
                val_correct = (pred == batch_y).sum()
                val_acc += val_correct.item()

                
        print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_data))*args.batch_size, val_acc / (len(val_data))))

        with torch.no_grad():
            model.eval()
            trans.eval()

            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                clean = model(batch_x)

                loss = F.cross_entropy(clean, batch_y.long())
                eval_loss += loss.item()
                pred = torch.max(clean, 1)[1]
                eval_correct = (pred == batch_y).sum()
                eval_acc += eval_correct.item()

            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)) * args.batch_size,
                                                          eval_acc / (len(test_data))))


            est_T = t.detach().cpu().numpy()
            # estimate_error = tools.error(est_T, train_data.t)

            matrix_path = matrix_dir + '/' + 'matrix_epoch_%d.npy' % (epoch+1)
            np.save(matrix_path, est_T)

            # print('Estimation Error: {:.2f}'.format(estimate_error))
            print(est_T)

        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch))+':' + str(train_acc / (len(train_data))) +' '  
                            +str(val_acc / (len(val_data)))+' '  + str(eval_acc / (len(test_data))) + "\n" )
            

        val_acc_list.append(val_acc / (len(val_data)))
        test_acc_list.append(eval_acc / (len(test_data)))

        noise_rate_str = str(args.noise_rate).replace('.', '_')
  
        save_dir = os.path.join("/mnt/ssd1/user/vol", args.dataset, args.noise_type, f"nr{noise_rate_str}")
        os.makedirs(save_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch{epoch}.pth'))

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

   


  


if __name__=='__main__':
    main()
