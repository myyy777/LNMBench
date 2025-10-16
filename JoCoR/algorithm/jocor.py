# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from model.cnn import MLPNet,CNN
import numpy as np
from common.utils import accuracy

from algorithm.loss import loss_jocor
import torchvision.models as models

class JoCoR:
    def __init__(self, args, train_dataset, device, input_channel, num_classes):

        # Hyper Parameters
        self.batch_size = 128
        learning_rate = args.lr
        self.num_classes = num_classes

        if args.forget_rate is None:
            if args.noise_type == "asymmetric":
                forget_rate = args.noise_rate / 2
            else:
                forget_rate = args.noise_rate
        else:
            forget_rate = args.forget_rate

        self.noise_or_not = train_dataset.noise_or_not

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [learning_rate] * args.n_epoch
        self.beta1_plan = [mom1] * args.n_epoch

        for i in range(args.epoch_decay_start, args.n_epoch):
            self.alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
            self.beta1_plan[i] = mom2

        # define drop rate schedule
        self.rate_schedule = np.ones(args.n_epoch) * forget_rate
        self.rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)

        self.device = device
        self.num_iter_per_epoch = args.num_iter_per_epoch
        self.print_freq = args.print_freq
        self.co_lambda = args.co_lambda
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset

        # if args.model_type == "cnn":
        #     self.model1 = CNN(input_channel=input_channel, n_outputs=num_classes)
        #     self.model2 = CNN(input_channel=input_channel, n_outputs=num_classes)
        # elif args.model_type == "mlp":
        #     self.model1 = MLPNet()
        #     self.model2 = MLPNet()
        self.model1 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model1.fc = nn.Linear(self.model1.fc.in_features, num_classes) 
        self.model1.to(device)
        print(self.model1.parameters)
        self.model2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model2.fc = nn.Linear(self.model2.fc.in_features, num_classes) 
        self.model2.to(device)
        # self.model2.to(device)
        print(self.model2.parameters)

        self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()),
                                          lr=learning_rate)

        self.loss_fn = loss_jocor


        self.adjust_lr = args.adjust_lr

    # Evaluate the Model
    # def evaluate(self, test_loader):
    #     print('Evaluating ...')
    #     self.model1.eval()  # Change model to 'eval' mode.
    #     self.model2.eval()  # Change model to 'eval' mode

    #     correct1 = 0
    #     total1 = 0
    #     for images, labels, _ in test_loader:
    #         images = Variable(images).to(self.device)
    #         logits1 = self.model1(images)
    #         outputs1 = F.softmax(logits1, dim=1)
    #         _, pred1 = torch.max(outputs1.data, 1)
    #         total1 += labels.size(0)
    #         correct1 += (pred1.cpu() == labels).sum()

    #     correct2 = 0
    #     total2 = 0
    #     for images, labels, _ in test_loader:
    #         images = Variable(images).to(self.device)
    #         logits2 = self.model2(images)
    #         outputs2 = F.softmax(logits2, dim=1)
    #         _, pred2 = torch.max(outputs2.data, 1)
    #         total2 += labels.size(0)
    #         correct2 += (pred2.cpu() == labels).sum()

    #     acc1 = 100 * float(correct1) / float(total1)
    #     acc2 = 100 * float(correct2) / float(total2)
    #     return acc1, acc2
    def evaluate(self, test_loader):
        print('Evaluating ...')
        self.model1.eval()
        self.model2.eval()

        correct1 = 0
        correct2 = 0
        correct_ens = 0
        total = 0

        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits1 = self.model1(images)
                logits2 = self.model2(images)

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

    # Train the Model
    def train(self, train_loader, epoch,logfile):
        print('Training ...')
        self.model1.train()  # Change model to 'train' mode.
        self.model2.train()  # Change model to 'train' mode

        if self.adjust_lr == 1:
            self.adjust_learning_rate(self.optimizer, epoch)

        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0
        pure_ratio_1_list = []
        pure_ratio_2_list = []
        stats_1 = {"selected": 0, "true_clean": 0, "total_clean": 0}
        stats_2 = {"selected": 0, "true_clean": 0, "total_clean": 0}
        class_stats_1 = {i: {"selected": 0, "true_clean": 0, "total_clean": 0} for i in range(self.num_classes)}
        class_stats_2 = {i: {"selected": 0, "true_clean": 0, "total_clean": 0} for i in range(self.num_classes)}

        for i, (images, labels, indexes) in enumerate(train_loader):
            ind = indexes.cpu().numpy().transpose()
            # if i > self.num_iter_per_epoch:
            #     break

            images = Variable(images).to(self.device)
            labels = Variable(labels).to(self.device)

            labels_np = labels.cpu().numpy()

            # Forward + Backward + Optimize
            logits1 = self.model1(images)
            prec1 = accuracy(logits1, labels, topk=(1,))
            train_total += 1
            train_correct += prec1

            logits2 = self.model2(images)
            prec2 = accuracy(logits2, labels, topk=(1,))
            train_total2 += 1
            train_correct2 += prec2

            loss_1, loss_2, sel_1 = self.loss_fn(logits1, logits2, labels, self.rate_schedule[epoch],
                                                                 ind, self.noise_or_not, self.co_lambda)
            for j in range(len(ind)):
                        cls = labels_np[j]
                        is_clean = self.noise_or_not[ind[j]]
                        stats_1["total_clean"] += is_clean
                        stats_2["total_clean"] += is_clean
                        class_stats_1[cls]["total_clean"] += is_clean
                        class_stats_2[cls]["total_clean"] += is_clean

            for idx1  in sel_1:
                i_global = ind[idx1]
                cls = labels_np[idx1]
                is_clean = self.noise_or_not[i_global]
                stats_1["selected"] += 1
                stats_1["true_clean"] += is_clean
                class_stats_1[cls]["selected"] += 1
                class_stats_1[cls]["true_clean"] += is_clean

           


            self.optimizer.zero_grad()
            loss_1.backward()
            self.optimizer.step()



            if (i + 1) % self.print_freq == 0:
                print(
                    'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f'
                    % (epoch, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, prec2,
                       loss_1.data.item(), loss_2.data.item()))


        def compute_metrics(stats, class_stats):
            overall_precision = stats["true_clean"] / stats["selected"] if stats["selected"] > 0 else 0
            overall_recall = stats["true_clean"] / stats["total_clean"] if stats["total_clean"] > 0 else 0
            per_class_precision = {}
            per_class_recall = {}
            for cls in range(self.num_classes):
                cstat = class_stats[cls]
                p = cstat["true_clean"] / cstat["selected"] if cstat["selected"] > 0 else 0
                r = cstat["true_clean"] / cstat["total_clean"] if cstat["total_clean"] > 0 else 0
                per_class_precision[cls] = p
                per_class_recall[cls] = r
            return overall_precision, overall_recall, per_class_precision, per_class_recall

        precision_1, recall_1, per_class_prec_1, per_class_rec_1 = compute_metrics(stats_1, class_stats_1)

        with open(logfile, "a") as f:
            f.write(f"Epoch {epoch}\n")
            f.write(f"Precision (true clean ratio): {precision_1 * 100:.2f}%   Recall (selected in all clean): {recall_1 * 100:.2f}%\n")
            f.write("=== Per-Class Statistics ===\n")
            for cls in range(self.num_classes):
                f.write(f"Class {cls} - Selected: {class_stats_1[cls]['selected']}, "
                        f"TrueClean: {class_stats_1[cls]['true_clean']}, "
                        f"TotalClean: {class_stats_1[cls]['total_clean']}, "
                        f"Precision: {per_class_prec_1[cls] * 100:.2f}%, "
                        f"Recall: {per_class_rec_1[cls] * 100:.2f}%\n")
            f.write("-" * 60 + "\n")

        train_acc1 = float(train_correct) / float(train_total)
        train_acc2 = float(train_correct2) / float(train_total2)
        
        return train_acc1, train_acc2

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
