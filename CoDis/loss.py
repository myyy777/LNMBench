from __future__ import print_function
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from numpy.testing import assert_array_almost_equal
import warnings

warnings.filterwarnings('ignore')

def kl_loss_compute(pred, soft_targets, reduce=True):


    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1), reduction='none')

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)
    

def js_loss_compute(pred, soft_targets, reduce=True):
    
    pred_softmax = F.softmax(pred, dim=1)
    targets_softmax = F.softmax(soft_targets, dim=1)
    mean = (pred_softmax + targets_softmax) / 2

    epsilon = 1e-8  #  log(0)
    mean = torch.clamp(mean, min=epsilon)
    mean = mean / mean.sum(dim=1, keepdim=True)  

    kl_1 = F.kl_div(F.log_softmax(pred, dim=1), mean, reduction='none')
    kl_2 = F.kl_div(F.log_softmax(soft_targets, dim=1), mean, reduction='none')
    js = (kl_1 + kl_2) / 2 
    
    if reduce:
        return torch.mean(torch.sum(js, dim=1))
    else:
        return torch.sum(js, 1)

def loss_ours(y_1, y_2, t, forget_rate, ind, noise_or_not, device, co_lambda=0.1):
    """
    双模型协同训练损失 + 噪声遗忘
    
    y_1, y_2: 两个模型的 logits，shape (B, C)
    t: 真实标签，shape (B,)
    forget_rate: 当前 epoch 的遗忘率 (0~1)
    ind: 原始数据集中该 batch 样本的全局索引，用于查询 noise_or_not
    noise_or_not: 训练集中每个样本是否干净的布尔数组
    """

    # 1) per-sample
    ce_1 = F.cross_entropy(y_1, t, reduction='none')  # (B,)
    ce_2 = F.cross_entropy(y_2, t, reduction='none')  # (B,)

    # 2) JS per-sample
    js_per_sample = js_loss_compute(y_1, y_2, reduce=False)  # (B,)

    # 3) 
    loss_1 = ce_1 - co_lambda * js_per_sample
    loss_2 = ce_2 - co_lambda * js_per_sample

    # 4)  loss “”
    ind_1_sorted = torch.argsort(loss_1.detach()).to(device)
    ind_2_sorted = torch.argsort(loss_2.detach()).to(device)

    # 5) “” 1 
    # remember_rate = 1.0 - forget_rate
    # batch_size = ind_1_sorted.size(0)
    # num_remember = max(1, int(remember_rate * batch_size))

    remember_rate = 1 - forget_rate
    batch_size = ind_1_sorted.size(0)
    num_remember = int(remember_rate * batch_size)

    # 6)  Tensor 
    ind_1_update = ind_1_sorted[:num_remember]  # LongTensor on device
    ind_2_update = ind_2_sorted[:num_remember]

    # 7)  numpy 
    idx1 = ind[ind_1_update.cpu().numpy()]
    idx2 = ind[ind_2_update.cpu().numpy()]

    # pure_ratio_1 = np.sum(noise_or_not[idx1]) / float(num_remember)
    # pure_ratio_2 = np.sum(noise_or_not[idx2]) / float(num_remember)
    ind_1_update = torch.tensor(ind_1_update, dtype=torch.long).to(y_1.device)
    ind_2_update = torch.tensor(ind_2_update, dtype=torch.long).to(y_2.device)

    # 8) 
    selected_loss_1 = loss_1[ind_2_update]  #  loss  y_1 (1) 
    selected_loss_2 = loss_2[ind_1_update]  #  loss  y_2 (2) 

    if torch.isnan(ce_1).any():
        print("❌ [NaN] 交叉熵 ce_1 出现 NaN")
    if torch.isnan(ce_2).any():
        print("❌ [NaN] 交叉熵 ce_2 出现 NaN")
    if torch.isnan(js_per_sample).any():
        print("❌ [NaN] JS 散度 出现 NaN")
    if torch.isnan(loss_1).any():
        print("❌ [NaN] loss_1 出现 NaN")
    if torch.isnan(loss_2).any():
        print("❌ [NaN] loss_2 出现 NaN")
    if torch.isnan(selected_loss_1).any():
        print("❌ [NaN] selected_loss_1 出现 NaN")
    if torch.isnan(selected_loss_2).any():
        print("❌ [NaN] selected_loss_2 出现 NaN")

    # 9) 
    return selected_loss_1.mean(), selected_loss_2.mean(), ind_1_update, ind_2_update



# def loss_ours(y_1, y_2, t, forget_rate, ind, noise_or_not, device, co_lambda=0.1):
#     # 1)  CE
#     loss_1 = F.cross_entropy(y_1, t, reduction='none')
#     loss_2 = F.cross_entropy(y_2, t, reduction='none')

#     # 2) 
#     ind_1_sorted = torch.argsort(loss_1.detach(), descending=False).to(device)
#     ind_2_sorted = torch.argsort(loss_2.detach(), descending=False).to(device)

#     # 3) 
#     remember_rate = 1 - forget_rate
#     num_remember = max(1, int(remember_rate * ind_1_sorted.size(0)))

#     # 4)  Tensor 
#     ind_1_update = ind_1_sorted[:num_remember]
#     ind_2_update = ind_2_sorted[:num_remember]

#     # 5)  pure_ratio numpy 
#     idx1_cpu = ind_1_update.cpu().numpy()
#     idx2_cpu = ind_2_update.cpu().numpy()
#     pure_ratio_1 = np.sum(noise_or_not[ind[idx1_cpu]]) / float(num_remember)
#     pure_ratio_2 = np.sum(noise_or_not[ind[idx2_cpu]]) / float(num_remember)

#     # 6) 
#     loss_1_update = loss_1[ind_2_update]
#     loss_2_update = loss_2[ind_1_update]

#     # 7) 
#     return loss_1_update.mean(), loss_2_update.mean(), pure_ratio_1, pure_ratio_2



# def loss_ours(y_1, y_2, t, forget_rate, ind, noise_or_not, device,co_lambda=0.1):

#     loss_1 = F.cross_entropy(y_1, t, reduction='none') - co_lambda * js_loss_compute(y_1, y_2,reduce=False)
#    
#     # ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
#     ind_1_sorted = torch.argsort(loss_1.detach()).to(device)

#     loss_1_sorted = loss_1[ind_1_sorted]

#     loss_2 = F.cross_entropy(y_2, t, reduction='none') - co_lambda * js_loss_compute(y_1, y_2,reduce=False)
#     
#     # ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
#     ind_2_sorted = torch.argsort(loss_2.detach()).to(device)

#     loss_2_sorted = loss_2[ind_2_sorted]

#     remember_rate = 1 - forget_rate
#     num_remember = int(remember_rate * len(loss_1_sorted))

#     ind_1_update=ind_1_sorted[:num_remember].cpu()
#     ind_2_update=ind_2_sorted[:num_remember].cpu()
#     if len(ind_1_update) == 0:
#         ind_1_update = ind_1_sorted.cpu().numpy()
#         ind_2_update = ind_2_sorted.cpu().numpy()
#         num_remember = ind_1_update.shape[0]

#     pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted.cpu()[:num_remember]]])/float(num_remember)
#     pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted.cpu()[:num_remember]]])/float(num_remember)

#     loss_1_update = loss_1[ind_2_update]
#     loss_2_update = loss_2[ind_1_update]
    
    
#     return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2


# def loss_ours(y_1, y_2, t, forget_rate, ind, noise_or_not, device,co_lambda=0.1):

#     # --- Debug Point 1: Check inputs and intermediate loss calculation ---
#     # This helps catch NaNs/Infs originating from model outputs or initial loss calculations
    
#     # Calculate Cross-Entropy Loss for y_1 and y_2 against true labels t
#     ce_loss_1 = F.cross_entropy(y_1, t, reduction='none')
#     ce_loss_2 = F.cross_entropy(y_2, t, reduction='none')

#     # Calculate JS Divergence between y_1 and y_2
#     js_div = js_loss_compute(y_1, y_2, reduce=False)
    
#     # Calculate the full loss for each model
#     loss_1 = ce_loss_1 - co_lambda * js_div
#     loss_2 = ce_loss_2 - co_lambda * js_div # Note: The original code used js_loss_compute(y_1, y_2) for both, which is typical for CoDis.

#     print(f"  CE Loss 1 (mean): {ce_loss_1.mean().item():.6f}, Min: {ce_loss_1.min().item():.6f}, Max: {ce_loss_1.max().item():.6f}")
#     print(f"  CE Loss 2 (mean): {ce_loss_2.mean().item():.6f}, Min: {ce_loss_2.min().item():.6f}, Max: {ce_loss_2.max().item():.6f}")
#     print(f"  JS Divergence (mean): {js_div.mean().item():.6f}, Min: {js_div.min().item():.6f}, Max: {js_div.max().item():.6f}")
#     print(f"  co_lambda * JS: {(co_lambda * js_div).mean().item():.6f}")
#     print(f"  Loss 1 (before sorting, mean): {loss_1.mean().item():.6f}, Min: {loss_1.min().item():.6f}, Max: {loss_1.max().item():.6f}")
#     print(f"  Loss 2 (before sorting, mean): {loss_2.mean().item():.6f}, Min: {loss_2.min().item():.6f}, Max: {loss_2.max().item():.6f}")

#     if torch.isnan(loss_1).any() or torch.isinf(loss_1).any():
#         print("  CRITICAL WARNING: loss_1 contains NaN/Inf BEFORE sorting!")
#     if torch.isnan(loss_2).any() or torch.isinf(loss_2).any():
#         print("  CRITICAL WARNING: loss_2 contains NaN/Inf BEFORE sorting!")

#     # Sort losses to select samples
#     ind_1_sorted = torch.argsort(loss_1.detach()).to(device)
#     loss_1_sorted = loss_1[ind_1_sorted]

#     ind_2_sorted = torch.argsort(loss_2.detach()).to(device)
#     loss_2_sorted = loss_2[ind_2_sorted]

#     remember_rate = 1 - forget_rate
#     current_batch_size = len(loss_1_sorted) # This is 128 for your batch size

#     # --- Debug Point 2: Check num_remember calculation ---
#     # This is the most common source of NaN from division by zero
#     calculated_num_remember_float = remember_rate * current_batch_size
#     num_remember = int(calculated_num_remember_float)

#     print(f"\n  Forget Rate: {forget_rate:.8f}")
#     print(f"  Remember Rate: {remember_rate:.8f}")
#     print(f"  Current Batch Size: {current_batch_size}")
#     print(f"  Calculated num_remember (float): {calculated_num_remember_float:.8f}")
#     print(f"  Num Remember (after int truncation): {num_remember}")

#     ind_1_update=ind_1_sorted[:num_remember].cpu()
#     ind_2_update=ind_2_sorted[:num_remember].cpu()

#     # --- Debug Point 3: Safeguard against num_remember == 0 ---
#     # The original fallback doesn't prevent num_remember from being 0 if len(ind_1_update) was already 0
#     if num_remember == 0:
#         # This warning means that even with a batch size of 128, the remember_rate was so small
#         # (less than 1/128 ~ 0.0078) that it resulted in 0 samples to remember.
#         warnings.warn("CRITICAL: num_remember is 0 after calculation. Returning placeholder loss to prevent NaN.")
#         # Return a tiny, non-zero loss to prevent division by zero and allow training to continue (with a warning)
#         return torch.tensor(1e-6, device=device), torch.tensor(1e-6, device=device), 0.0, 0.0

#     # Original fallback block (mostly redundant if num_remember is already 0, but keep for original logic)
#     if len(ind_1_update) == 0: # This check is only true if num_remember was 0
#         ind_1_update = ind_1_sorted.cpu().numpy() # This will still be an empty numpy array if num_remember was 0
#         ind_2_update = ind_2_sorted.cpu().numpy()
#         num_remember = ind_1_update.shape[0] # This will still be 0

#     pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted.cpu()[:num_remember]]])/float(num_remember)
#     pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted.cpu()[:num_remember]]])/float(num_remember)

#     loss_1_update = loss_1[ind_2_update]
#     loss_2_update = loss_2[ind_1_update]

#     # --- Debug Point 4: Final loss values before return ---
#     print(f"\n  Final Loss 1 Update Sum: {torch.sum(loss_1_update).item():.6f}")
#     print(f"  Final Loss 2 Update Sum: {torch.sum(loss_2_update).item():.6f}")
#     print(f"  Pure Ratio 1: {pure_ratio_1:.6f}")
#     print(f"  Pure Ratio 2: {pure_ratio_2:.6f}")
    
#     return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2
