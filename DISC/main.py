import argparse
from utils import load_config, get_log_name, save_results, \
                  get_test_acc, print_config
from datasets import cifar_dataloader
import algorithms
import numpy as np

import nni
import time
import torchvision.models as models
from transformer import transform_train, transform_test,transform_target
import data_load
from tools import set_seed
import torch
from torch.utils.data import Dataset,DataLoader
import datetime
import os
from data_load import TransformFixMatch_PathMNIST, TransformFixMatch_DRTID, TransformFixMatch_chexpert

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    '-c',
                    type=str,
                    default='/home/user/label_noise/DISC/configs/DISC_organcmnist.py',
                    help='The path of config file.')
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--dataset', type=str, default='bloodmnist',)
parser.add_argument('--root', type=str, default='/data/yfli/CIFAR10')
parser.add_argument('--save_path', type=str, default='./log/')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--noise_type', type=str, default='instance')
parser.add_argument('--noise_rate', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_classes', type=int, default=11)
parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()


def main():
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    tuner_params = nni.get_next_parameter()
    config = load_config(args.config, _print=False)
    config.update(tuner_params)
    config['dataset'] = args.dataset
    config['root'] = args.root
    config['gpu'] = args.gpu
    config['noise_type'] = args.noise_type
    config['noise_rate'] = args.noise_rate
    config['seed'] = args.seed
    # config['num_classes'] = args.num_classes
    config['momentum'] = args.momentum
    clean_label = None
    print_config(config)
    set_seed(config['seed'])
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")

    if config['dataset'] == 'pathmnist':

        train_data = data_load.pathmnist_dataset(transform=TransformFixMatch_PathMNIST(), target_transform=transform_target,
                                            noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, device = device)
        val_data = data_load.pathmnist_val_dataset(transform=transform_test(args.dataset), target_transform=transform_target,
                                            noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, device = device)
        test_data = data_load.pathmnist_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
        # model = ResNet34(args.num_classes)
        
        trainloader = DataLoader(dataset=train_data, 
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=4)
        valloader = DataLoader(dataset=val_data, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            
                            num_workers=4)
    
        testloader = DataLoader(dataset=test_data,
                        batch_size=args.batch_size,
                        num_workers=4,
                        shuffle=False
                        )
        clean_label = train_data.t



    if config['dataset'] == 'dermamnist':

        train_data = data_load.dermamnist_dataset(transform=TransformFixMatch_PathMNIST(), target_transform=transform_target,
                                            noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, device = device)
        val_data = data_load.dermamnist_val_dataset(transform=transform_test(args.dataset), target_transform=transform_target,
                                            noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, device = device)
        test_data = data_load.dermamnist_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
        # model = ResNet34(args.num_classes)
        
        trainloader = DataLoader(dataset=train_data, 
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=4)
        valloader = DataLoader(dataset=val_data, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            
                            num_workers=4)
    
        testloader = DataLoader(dataset=test_data,
                        batch_size=args.batch_size,
                        num_workers=4,
                        shuffle=False
                        )
        clean_label = train_data.t

    if config['dataset'] == 'bloodmnist':

            train_data = data_load.bloodmnist_dataset(transform=TransformFixMatch_PathMNIST(), target_transform=transform_target,
                                                noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, device = device)
            val_data = data_load.bloodmnist_val_dataset(transform=transform_test(args.dataset), target_transform=transform_target,
                                                noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, device = device)
            test_data = data_load.bloodmnist_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
            # model = ResNet34(args.num_classes)
            
            trainloader = DataLoader(dataset=train_data, 
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=4)
            valloader = DataLoader(dataset=val_data, 
                                batch_size=args.batch_size,
                                shuffle=False,
                                
                                num_workers=4)
        
            testloader = DataLoader(dataset=test_data,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=False
                            )
            clean_label = train_data.t

    if config['dataset'] == 'organcmnist':

        train_data = data_load.organcmnist_dataset(transform=TransformFixMatch_PathMNIST(), target_transform=transform_target,
                                            noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, device = device)
        val_data = data_load.organcmnist_val_dataset(transform=transform_test(args.dataset), target_transform=transform_target,
                                            noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, device = device)
        test_data = data_load.organcmnist_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
        # model = ResNet34(args.num_classes)
        
        trainloader = DataLoader(dataset=train_data, 
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=4)
        valloader = DataLoader(dataset=val_data, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            
                            num_workers=4)
    
        testloader = DataLoader(dataset=test_data,
                        batch_size=args.batch_size,
                        num_workers=4,
                        shuffle=False
                        )
        clean_label = train_data.t




    if config['dataset'] == 'drtid':

        train_data = data_load.drtid_dataset(transform=TransformFixMatch_DRTID(), target_transform=transform_target,
                                            noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, device = device)
        val_data = data_load.drtid_val_dataset(transform=transform_test(args.dataset), target_transform=transform_target,
                                            noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, device = device)
        test_data = data_load.drtid_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
        # model = ResNet34(args.num_classes)
        
        trainloader = DataLoader(dataset=train_data, 
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=4)
        valloader = DataLoader(dataset=val_data, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            
                            num_workers=4)
    
        testloader = DataLoader(dataset=test_data,
                        batch_size=args.batch_size,
                        num_workers=4,
                        shuffle=False
                        )
        clean_label = train_data.t


    if config['dataset'] == 'kaggledr':

        train_data = data_load.kaggledr(transform=TransformFixMatch_DRTID(), target_transform=transform_target,
                                            noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, device = device)
        val_data = data_load.kaggledr_val(transform=transform_test(args.dataset), target_transform=transform_target,
                                            noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, device = device)
        test_data = data_load.kaggledr_test(transform=transform_test(args.dataset), target_transform=transform_target)
        # model = ResNet34(args.num_classes)
        
        trainloader = DataLoader(dataset=train_data, 
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=4)
        valloader = DataLoader(dataset=val_data, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            
                            num_workers=4)
    
        testloader = DataLoader(dataset=test_data,
                        batch_size=args.batch_size,
                        num_workers=4,
                        shuffle=False
                        )
        clean_label = train_data.t

    if config['dataset'] == 'chexpert':

        train_data = data_load.chexpert(transform=TransformFixMatch_chexpert(), target_transform=transform_target,
                                            noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, device = device)
        val_data = data_load.chexpert_val(transform=transform_test(args.dataset), target_transform=transform_target,
                                            noise_rate=args.noise_rate, random_seed=args.seed, noise_type=args.noise_type, device = device)
        test_data = data_load.chexpert_test(transform=transform_test(args.dataset), target_transform=transform_target)
        # model = ResNet34(args.num_classes)
        
        trainloader = DataLoader(dataset=train_data, 
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=4)
        valloader = DataLoader(dataset=val_data, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            
                            num_workers=4)
    
        testloader = DataLoader(dataset=test_data,
                        batch_size=args.batch_size,
                        num_workers=4,
                        shuffle=False
                        )
        clean_label = train_data.t

    if config['algorithm'] == 'DISC':
        model = algorithms.DISC(config,
                                input_channel=config['input_channel'],
                                num_classes=config['num_classes'], clean_label=clean_label)
        train_mode = 'train_index'
        
        
    else:
        model = algorithms.__dict__[config['algorithm']](
            config,
            input_channel=config['input_channel'],
            num_classes=config['num_classes'])
        train_mode = 'train_single'
        if config['algorithm'] == 'StandardCETest':
            train_mode = 'train_index'

    if 'cifar' in config['dataset']:
        dataloaders = cifar_dataloader(cifar_type=config['dataset'],
                                    #    root=config['root'],
                                       batch_size=config['batch_size'],
                                       num_workers=config['num_workers'],
                                       noise_type=config['noise_type'],
                                       percent=config['percent'])
        trainloader, testloader = dataloaders.run(
            mode=train_mode), dataloaders.run(mode='test')

    num_test_images = len(testloader.dataset)
    save_dir = args.result_dir +'/' +args.dataset

    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % save_dir)

    model_str=args.dataset+'_DISC_'+args.noise_type+'_'+str(args.noise_rate)+'_'+'seed' + str(args.seed)

    txtfile=save_dir+"/"+model_str+".txt"
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    if os.path.exists(txtfile):
        os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))

    start_epoch = 0
    epoch = 0
    with open(txtfile, "a") as myfile:
        myfile.write('epoch:  val_acc test_acc \n')
    # evaluate models with random weights
    if 'webvision' in config['dataset']:
        test_acc = model.evaluate(testloader)
        print(
            'Epoch [%d/%d] Test Accuracy on the %s test images: top1: %.4f, top5: %.4f'
            % (epoch, config['epochs'], num_test_images, test_acc[0],
               test_acc[1]))
    else:
        val_acc = get_test_acc(model.evaluate(valloader))
        test_acc = get_test_acc(model.evaluate(testloader))
        print('Epoch [%d/%d] Test Accuracy on the %s test images: %.4f' %
              (epoch, config['epochs'], num_test_images, test_acc))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  +  str(val_acc) +' ' + str(test_acc)+"\n")

    acc_list, acc_all_list = [], []
    best_acc, best_epoch = 0.0, 0
    
    # loading training labels
    if config['algorithm'] == 'DISC' or config['algorithm'] == 'StandardCETest':
        
        model.get_labels(trainloader)
        model.weak_labels = model.labels.detach().clone()
        print('The labels are loaded!!!')

    since = time.time()
 
   
    val_acc_list = []
    test_acc_list = []
    for epoch in range(start_epoch, config['epochs']):
        # train
        model.train(trainloader, epoch)


        # evaluate
        val_acc = get_test_acc(model.evaluate(valloader))
        test_acc = get_test_acc(model.evaluate(testloader))
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)

        if best_acc < test_acc:
            best_acc, best_epoch = test_acc, epoch

        print(
            'Epoch [%d/%d] Test Accuracy on the %s test images: %.4f %%' %
            (epoch + 1, config['epochs'], num_test_images, test_acc))

        if epoch >= config['epochs'] - 10:
            acc_list.extend([test_acc])

        acc_all_list.extend([test_acc])
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  +  str(val_acc) +' ' + str(test_acc)+"\n")

        noise_rate_str = str(args.noise_rate).replace('.', '_')
        save_dir = os.path.join("/mnt/ssd1/user/disc", args.dataset, args.noise_type, f"nr{noise_rate_str}")
        os.makedirs(save_dir, exist_ok=True)

        torch.save(model.model_scratch.state_dict(), os.path.join(save_dir, f'model_epoch{epoch}.pth'))
        # torch.save(cnn2.state_dict(), os.path.join(save_dir, f'model2_epoch{epoch}.pth'))
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
        
    time_elapsed = time.time() - since
    total_min = time_elapsed // 60
    hour = total_min // 60
    min = total_min % 60
    sec = time_elapsed % 60

    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        hour, min, sec))

    if config['save_result']:
        config['algorithm'] = config['algorithm'] + args.model_name
        acc_np = np.array(acc_list)
        nni.report_final_result(acc_np.mean())
        jsonfile = get_log_name(config, path=args.save_path)
        np.save(jsonfile.replace('.json', '.npy'), np.array(acc_all_list))
        save_results(config=config,
                     last_ten=acc_np,
                     best_acc=best_acc,
                     best_epoch=best_epoch,
                     jsonfile=jsonfile)


if __name__ == '__main__':
    main()
