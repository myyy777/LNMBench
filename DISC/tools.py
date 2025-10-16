import numpy as np
import noise_utils, pdb
import random
import torch
def norm(T):
    row_sum = np.sum(T, 1)
    T_norm = T / row_sum
    return T_norm

def error(T, T_true):
    error = np.sum(np.abs(T-T_true)) / np.sum(np.abs(T_true))
    return error

def transition_matrix_generate(noise_rate=0.5, num_classes=10):
    P = np.ones((num_classes, num_classes))
    n = noise_rate
    P = (n / (num_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, num_classes-1):
            P[i, i] = 1. - n
        P[num_classes-1, num_classes-1] = 1. - n
    return P


def fit(X, num_classes, filter_outlier=False):
    # number of classes
    c = num_classes
    T = np.empty((c, c))
    eta_corr = X
    for i in np.arange(c):
        if not filter_outlier:
            idx_best = np.argmax(eta_corr[:, i])
        else:
            eta_thresh = np.percentile(eta_corr[:, i], 97,interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)
        for j in np.arange(c):
            T[i, j] = eta_corr[idx_best, j]
    return T


# flip clean labels to noisy labels
# train set and val set split
def dataset_split(train_images, train_labels, noise_rate=0.5, split_per=0.9, random_seed=0, num_classes=10, noise_type='instance',device=None):
    # clean_train_labels = train_labels[:, np.newaxis]
    train_labels_clean = np.copy(train_labels)
    if noise_type == 'symmetric':
        noisy_labels, real_noise_rate, transition_matrix = noise_utils.noisify_multiclass_symmetric(train_labels,
                                                noise=noise_rate, random_state= random_seed, nb_classes=num_classes)
        #print(noisy_labels.shape)
        #rint(type(noisy_labels))

    elif noise_type == 'flip':
        noisy_labels, real_noise_rate, transition_matrix = noise_utils.noisify_pairflip(train_labels,
                                                    noise=noise_rate, random_state=random_seed, nb_classes=num_classes)
    elif noise_type == 'asymmetric':
        noisy_labels, real_noise_rate, transition_matrix = noise_utils.noisify_multiclass_asymmetric(train_labels,
                                                    noise=noise_rate, random_state=random_seed, nb_classes=num_classes)

    elif noise_type == 'instance':
        noisy_labels = noise_utils.noisify_instance(noise_rate, train_images, train_labels,num_classes,  random_seed, device)

        noisy_labels = np.array(noisy_labels)
        noisy_labels = noisy_labels.squeeze()

        # return train_images, noisy_labels, train_labels
        #return  clean_data,clean_lables,None
    total = len(noisy_labels)
    indices = np.arange(total)
    np.random.shuffle(indices)
    split = int(split_per * total)
    train_idx = indices[:split]
    val_idx = indices[split:]
    train_set = train_images[train_idx]
    val_set = train_images[val_idx]
    train_noisy_labels = noisy_labels[train_idx]
    val_noisy_labels = noisy_labels[val_idx]
    train_clean_labels = train_labels_clean[train_idx] 

    noisy_labels = noisy_labels.squeeze()
    train_images = train_set
    noisy_labels = train_noisy_labels
    train_labels = train_clean_labels

    return train_images,  noisy_labels, train_labels, val_set,  val_noisy_labels  #transition_matrix
    #return train_set, val_set, train_labels, val_labels, transition_matrix
    


def set_seed(seed):
    # Python 
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    # PyTorch GPU
    torch.cuda.manual_seed(seed)


    #  CuDNN 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False