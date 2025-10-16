import os
import torch
import tools
import numpy as np
import data_load
import argparse, sys
import Lenet, Resnet
import torch.nn as nn
import torch.optim as optim
from loss import reweight_loss, reweighting_revision_loss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from transformer import transform_train, transform_test,transform_target
import torchvision.models as models
import torch.nn as nn
import datetime
from data_load import  PATHMNIST,PATHMNIST_TEST,PATHMNIST_VAL,DRTID,DRTID_test,DRTID_val,DERMAMNIST,DERMAMNIST_TEST,DERMAMNIST_VAL,ORGANCMNIST,ORGANCMNIST_TEST,ORGANCMNIST_VAL, BLOODMNIST, BLOODMNIST_TEST, BLOODMNIST_VAL
from data_load import Chexpert,Chexpert_test,Chexpert_val,DRTID,DRTID_test,DRTID_val,Kaggledr,Kaggledr_test,Kaggledr_val
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='initial learning rate', default=0.01)
parser.add_argument('--lr_revision', type=float, help='revision training learning rate', default=5e-7)
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=1e-4)
parser.add_argument('--model_dir', type=str, help='dir to save model files', default='model')
parser.add_argument('--prob_dir', type=str, help='dir to save output probability files', default='prob' )
parser.add_argument('--matrix_dir', type=str, help='dir to save estimated matrix', default='matrix')
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'dermamnist')
parser.add_argument('--n_epoch', type=int, default=50)
parser.add_argument('--n_epoch_revision', type=int, default=50)
parser.add_argument('--n_epoch_estimate', type=int, default=5)
parser.add_argument('--num_classes', type=int, default=8)
parser.add_argument('--percentile', type=int, default=97)
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--gpu', type=int, help='ind of gpu', default=2)
parser.add_argument('--adjust_lr', type=int, help='adjust lr', default=1)

args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
tools.set_seed(args.seed)
#seed
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)


#mnist, cifar10, cifar100

    
    

   

if args.dataset == 'pathmnist':
    args.n_epoch = 50
    args.n_epoch_estimate = 5
    args.num_classes = 9
    args.n_epoch_revision = 50

    train_dataset =PATHMNIST(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                 

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)

    val_dataset = PATHMNIST_VAL(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                  
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)

    test_dataset = PATHMNIST_TEST(target_transform = transform_target,transform=transform_test(args.dataset))

 

    estimate_state = True
    model = Resnet.ResNetWithRevision(num_classes=args.num_classes).to(device) 


if args.dataset == 'dermamnist':
    args.n_epoch = 50
    args.n_epoch_estimate = 5
    args.num_classes = 7
    args.n_epoch_revision = 50

    train_dataset =DERMAMNIST(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                 

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)

    val_dataset = DERMAMNIST_VAL(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                  
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)

    test_dataset = DERMAMNIST_TEST(target_transform = transform_target,transform=transform_test(args.dataset))

 

    estimate_state = True
    model = Resnet.ResNetWithRevision(num_classes=args.num_classes).to(device) 

if args.dataset == 'bloodmnist':
    args.n_epoch = 50
    args.n_epoch_estimate = 5
    args.num_classes = 8
    args.n_epoch_revision = 50

    train_dataset =BLOODMNIST(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                 

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)

    val_dataset = BLOODMNIST_VAL(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                  
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)

    test_dataset = BLOODMNIST_TEST(target_transform = transform_target,transform=transform_test(args.dataset))

 

    estimate_state = True
    model = Resnet.ResNetWithRevision(num_classes=args.num_classes).to(device) 

if args.dataset == 'organcmnist':
    args.n_epoch = 50
    args.n_epoch_estimate = 5
    args.num_classes = 11
    args.n_epoch_revision = 50

    train_dataset =ORGANCMNIST(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                 

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)

    val_dataset = ORGANCMNIST_VAL(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                  
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)

    test_dataset = ORGANCMNIST_TEST(target_transform = transform_target,transform=transform_test(args.dataset))

 

    estimate_state = True
    model = Resnet.ResNetWithRevision(num_classes=args.num_classes).to(device) 


if args.dataset == 'drtid':
    args.n_epoch = 50
    args.n_epoch_estimate = 5
    args.num_classes = 5
    args.n_epoch_revision = 50

    train_dataset =DRTID(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                 

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)

    val_dataset = DRTID_val(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                  
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)

    test_dataset = DRTID_test(target_transform = transform_target,transform=transform_test(args.dataset))

  
    estimate_state = True
    model = Resnet.ResNetWithRevision(num_classes=args.num_classes).to(device)
    
if args.dataset == 'kaggledr':
    args.n_epoch = 50
    args.n_epoch_estimate = 5
    args.num_classes = 5
    args.n_epoch_revision = 50

    train_dataset =Kaggledr(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                 

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)

    val_dataset = Kaggledr_val(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                  
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)

    test_dataset = Kaggledr_test(target_transform = transform_target,transform=transform_test(args.dataset))

  
    estimate_state = True
    model = Resnet.ResNetWithRevision(num_classes=args.num_classes).to(device)

if args.dataset == 'chexpert':
    args.n_epoch = 50
    args.n_epoch_estimate = 5
    args.num_classes = 5
    args.n_epoch_revision = 50

    train_dataset =Chexpert(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                 

                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,

                                    random_seed=args.seed,
                                    device=device)

    val_dataset = Chexpert_val(
                                    transform=transform_train(args.dataset),
                                    target_transform = transform_target,
                                  
                                    
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                
                                    random_seed=args.seed,
                                    device=device)

    test_dataset = Chexpert_test(target_transform = transform_target,transform=transform_test(args.dataset))

  
    estimate_state = True
    model = Resnet.ResNetWithRevision(num_classes=args.num_classes).to(device)

#optimizer and StepLR
optimizer_es = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
optimizer_revision = optim.Adam(model.parameters(), lr=args.lr_revision, weight_decay=args.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

      
    
#data_loader
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=4,
                          drop_last=True)

estimate_loader = DataLoader(dataset=train_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4
                             )

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=4
                       )

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=args.batch_size,
                         num_workers=4,
                         shuffle=False,)

#loss
loss_func_ce = nn.CrossEntropyLoss()
loss_func_reweight = reweight_loss()
loss_func_revision = reweighting_revision_loss()

#cuda
if torch.cuda.is_available:
    model = model.to(device)
    loss_func_ce = loss_func_ce.to(device)
    loss_func_reweight = loss_func_reweight.to(device)
    loss_func_revision = loss_func_revision.to(device)
    
#mkdir
model_save_dir = args.model_dir + '/' + args.dataset + '/' + args.noise_type + '/' + 'noise_rate_%s'%(args.noise_rate) 

if not os.path.exists(model_save_dir):
    os.system('mkdir -p %s'%(model_save_dir))

prob_save_dir = args.prob_dir + '/' + args.dataset + '/' + args.noise_type + '/' + 'noise_rate_%s'%(args.noise_rate)

if not os.path.exists(prob_save_dir):
    os.system('mkdir -p %s'%(prob_save_dir))

matrix_save_dir = args.matrix_dir + '/' + args.dataset + '/'+ args.noise_type + '/' + 'noise_rate_%s'%(args.noise_rate)

if not os.path.exists(matrix_save_dir):
    os.system('mkdir -p %s'%(matrix_save_dir))
save_dir = args.result_dir +'/' +args.dataset+'/T_Revision/'

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str=args.dataset+'_trevision_'+args.noise_type+'_'+str(args.noise_rate)+'_'+'seed' + str(args.seed)+'_'+'adjust_lr'+str(args.adjust_lr)

txtfile=save_dir+"/"+model_str+".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))
#estimate transition matrix
index_num = int(len(train_dataset) / args.batch_size)
A = torch.zeros((args.n_epoch_estimate, len(train_dataset), args.num_classes))   
val_acc_list = []
total_index =  index_num + 1

#main function
def main():

    print(args)
    print('Estimate transition matirx......Waiting......')
    
    for epoch in range(args.n_epoch_estimate):
      
        print('epoch {}'.format(epoch))
        model.train()
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
     
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer_es.zero_grad()
            out = model(batch_x, revision=False)
            loss = loss_func_ce(out, batch_y)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            loss.backward()
            optimizer_es.step()
        
        torch.save(model.state_dict(), model_save_dir + '/'+ 'epoch_%d.pth'%(epoch+1))
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_dataset))*args.batch_size, train_acc / (len(train_dataset))))
        
        with torch.no_grad():
            model.eval()
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                out = model(batch_x, revision=False)
                loss = loss_func_ce(out, batch_y)
                val_loss += loss.item()
                pred = torch.max(out, 1)[1]
                val_correct = (pred == batch_y).sum()
                val_acc += val_correct.item()
                
        print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_dataset))*args.batch_size, val_acc / (len(val_dataset)))) 
        val_acc_list.append(val_acc / (len(val_dataset)))
        
        with torch.no_grad():
            model.eval()
            for index,(batch_x,batch_y) in enumerate(estimate_loader):
                 batch_x = batch_x.to(device)
                 out = model(batch_x, revision=False)
                 out = F.softmax(out,dim=1)
                 out = out.cpu()
                 if index <= index_num:
                    A[epoch][index*args.batch_size:(index+1)*args.batch_size, :] = out 
                 else:
                     A[epoch][index_num*args.batch_size, len(train_dataset), :] = out 
       
    val_acc_array = np.array(val_acc_list)
    model_index = np.argmax(val_acc_array)
    
    A_save_dir = prob_save_dir + '/' + 'prob.npy'
    np.save(A_save_dir, A)           
    prob_ = np.load(A_save_dir)
    
    transition_matrix_ = tools.fit(prob_[model_index, :, :], args.num_classes, estimate_state)
    transition_matrix = tools.norm(transition_matrix_)
    
    matrix_path = matrix_save_dir + '/' + 'transition_matrix.npy'
    np.save(matrix_path, transition_matrix)
    T = torch.from_numpy(transition_matrix).float().to(device)


    
    True_T = tools.transition_matrix_generate(noise_rate=args.noise_rate, num_classes=args.num_classes)
    estimate_error = tools.error(T.cpu().numpy(), True_T)
    print('The estimation error is %s'%(estimate_error))
    # initial parameters
        
    estimate_model_path = model_save_dir + '/' + 'epoch_%s.pth'%(model_index+1)
    estimate_model_path = torch.load(estimate_model_path)
    model.load_state_dict(estimate_model_path)
    
    print('Estimate finish.....Training......')
    val_acc_list_r = []
    with open(txtfile, "a") as myfile:
        myfile.write('Estimate finish.....Training......\n')
        myfile.write('epoch: train_acc val_acc test_acc \n')
    for epoch in range(args.n_epoch):
        print('epoch {}'.format(epoch))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
        eval_loss = 0.
        eval_acc = 0.
        # scheduler.step()
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_y = batch_y.long()
            optimizer.zero_grad()
            out = model(batch_x, revision=False)
            prob = F.softmax(out, dim=1)
            prob = prob.t()
            loss = loss_func_reweight(out, T, batch_y)
            out_forward = torch.matmul(T.t(), prob)
            out_forward = out_forward.t()
            train_loss += loss.item()
            pred = torch.max(out_forward, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        with torch.no_grad(): 
            model.eval()
            for batch_x,batch_y in val_loader:
                model.eval()
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                out = model(batch_x, revision=False)
                prob = F.softmax(out, dim=1)
                prob = prob.t()
                loss = loss_func_reweight(out, T, batch_y)
                out_forward = torch.matmul(T.t(), prob)
                out_forward = out_forward.t()
                val_loss += loss.item()
                pred = torch.max(out_forward, 1)[1]
                val_correct = (pred == batch_y).sum()
                val_acc += val_correct.item()
                
        torch.save(model.state_dict(), model_save_dir + '/'+ 'epoch_r%d.pth'%(epoch))
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_dataset))*args.batch_size, train_acc / (len(train_dataset))))
        val_acc_list_r.append(val_acc / (len(val_dataset)))
        
        with torch.no_grad():
            model.eval()
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                out = model(batch_x, revision=False)
                loss = loss_func_ce(out, batch_y)
                eval_loss += loss.item()
                pred = torch.max(out, 1)[1]
                eval_correct = (pred == batch_y).sum()
                eval_acc += eval_correct.item()
                
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_dataset))*args.batch_size, eval_acc / (len(test_dataset))))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch))+':' + str(train_acc / (len(train_dataset))) +' '  
                         +str(val_acc / (len(val_dataset)))+' '  + str(eval_acc / (len(test_dataset))) + "\n" )
            
    val_acc_array_r = np.array(val_acc_list_r)
    reweight_model_index = np.argmax(val_acc_array_r)
    
    reweight_model_path = model_save_dir + '/' + 'epoch_r%s.pth'%(reweight_model_index+1)
    reweight_model_path = torch.load(reweight_model_path)
    model.load_state_dict(reweight_model_path)
    nn.init.constant_(model.T_revision.weight, 0.0)
    
    print('Revision......')
    with open(txtfile, "a") as myfile:
        myfile.write('Revision......\n')
        myfile.write('epoch: train_acc val_acc test_acc \n')
    val_acc_list_f = []
    test_acc_list_f = []
    for epoch in range(args.n_epoch_revision):
       
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
        eval_loss = 0.
        eval_acc = 0.
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer_revision.zero_grad()  
            out, correction = model(batch_x, revision=True)
            prob = F.softmax(out, dim=1)
            prob = prob.t()
            loss = loss_func_revision(out, T, correction, batch_y)
            out_forward = torch.matmul((T+correction).t(), prob)
            out_forward = out_forward.t()
            train_loss += loss.item()
            pred = torch.max(out_forward, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            loss.backward()
            optimizer_revision.step()
            
        with torch.no_grad(): 
            model.eval()        
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                out, correction = model(batch_x, revision=True)
                prob = F.softmax(out, dim=1)
                prob = prob.t()
                loss = loss_func_revision(out, T, correction, batch_y)
                out_forward = torch.matmul((T+correction).t(), prob)
                out_forward = out_forward.t()
                val_loss += loss.item()
                pred = torch.max(out_forward, 1)[1]
                val_correct = (pred == batch_y).sum()
                val_acc += val_correct.item()
                 
        estimate_error = tools.error(True_T, (T+correction).cpu().detach().numpy()) 
        matrix_path = matrix_save_dir + '/' + f'transition_matrix_r{epoch}.npy'
        np.save(matrix_path, (T+correction).cpu().detach().numpy())
        print('Estimate error: {:.6f}'.format(estimate_error))
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_dataset))*args.batch_size, train_acc / (len(train_dataset))))
        print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(val_dataset))*args.batch_size, val_acc / (len(val_dataset))))
    
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                out, _ = model(batch_x, revision=True)
                loss = loss_func_ce(out, batch_y)
                eval_loss += loss.item()
                pred = torch.max(out, 1)[1]
                eval_correct = (pred == batch_y).sum()
                eval_acc += eval_correct.item()
                
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_dataset))*args.batch_size, eval_acc / (len(test_dataset))))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch))+':' + str(train_acc / (len(train_dataset))) +' '  
                         +str(val_acc / (len(val_dataset)))+' '  + str(eval_acc / (len(test_dataset)))+"\n"  )
        val_acc_list_f.append(val_acc)
        test_acc_list_f.append(eval_acc)
        noise_rate_str = str(args.noise_rate).replace('.', '_')
  
        save_dir = os.path.join("/mnt/ssd1/user/trevision", args.dataset, args.noise_type, f"nr{noise_rate_str}")
        os.makedirs(save_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch{epoch}.pth'))
        # 1. val  test acc
    best_val_epoch = val_acc_list_f.index(max(val_acc_list_f))
    test_at_best_val = test_acc_list_f[best_val_epoch]
    # 2. test acc 
    best_test_acc = max(test_acc_list_f)

    # 3.  5  epoch  test acc 
    last_5_test_acc = sum(test_acc_list_f[-5:]) / len(test_acc_list_f[-5:])
    
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
