import os, torch
import os.path
import copy, pdb
import hashlib
import errno
import numpy as np
from numpy.testing import assert_array_almost_equal
import torch.nn.functional as F
import time
from torchvision import models, transforms
from scipy import stats
from tqdm import tqdm
from math import inf
def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files

# basic function
def multiclass_noisify(y, P, random_state=1):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
#    print (np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        # flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        flipped = flipper.multinomial(1, P[i, :], 1)[0]

        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=1, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print (P)

    return y_train, actual_noise,P

def noisify_multiclass_asymmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.eye(nb_classes)

    if noise > 0.0:
        P = np.random.uniform(low=0.1, high=1., size=(nb_classes, nb_classes))
        for i in range(nb_classes):
            P[i, i] = 1 - noise
            sum = P[i].sum() - P[i, i]
            for j in range(nb_classes):
                if i != j:
                    P[i, j] = P[i, j] / sum * noise


        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy

    return y_train, actual_noise,P

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy

    return y_train, actual_noise,P

def noisify(nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=1):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate, t = noisify_pairflip(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate, t = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)

    if noise_type == 'asymmetric':
        train_noisy_labels, actual_noise_rate, t = noisify_multiclass_asymmetric(train_labels, noise_rate,
                                                                             random_state=random_state, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate



def noisify_instance(
    n,               # 
    train_images,    # ndarray or Tensor, shape=(N,C,H,W) or (N,H,W,C)
    train_labels,    # ndarray or Tensor, shape=(N,)
    num_classes,     # 
    seed,            # 
    device,          # torch.device
):
    # 0) 
    cache_dir="/home/user/label_noise/instance_noise"# 
    os.makedirs(cache_dir, exist_ok=True)
    nr = str(n).replace('.', '_')
    cache_file = os.path.join(
        cache_dir,
        f"noisy_nr{nr}_seed{seed}.npy"
    )
    # 1) 
    if os.path.exists(cache_file):
        print("Loading noisy labels from cache:", cache_file)
        return np.load(cache_file)

    # 2) 
    print("Cache not found. Generating noisy labels...")
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    norm_std    = 0.1
    flip_dist   = stats.truncnorm((0-n)/norm_std, (1-n)/norm_std, loc=n, scale=norm_std)
    flip_rate   = flip_dist.rvs(len(train_labels))

    # prepare clean labels tensor
    labels = train_labels
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    labels = labels.to(device).view(-1).long()

    # load model
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(
        "/home/user/label_noise/resnet50_pathmnist_test.pth",
        map_location=device
    ))
    model.to(device).eval()

    # build P distributions
    P_list = []
    with torch.no_grad():
        for idx, (img, y) in enumerate(zip(train_images, train_labels)):
            # make x [1,C,H,W]
            if isinstance(img, np.ndarray):
                x = torch.from_numpy(img).float()  # maybe HWC or CHW
                if x.ndim == 3 and x.shape[2] in (1,3):
                    x = x.permute(2,0,1)
            else:
                x = img.clone().float()
                if x.ndim == 3 and x.shape[2] in (1,3):
                    x = x.permute(2,0,1)
            x = x.unsqueeze(0).to(device)

            y = int(y)
            logits = model(x).squeeze(0)        # (num_classes,)
            logits[y] = -inf
            base = F.softmax(logits, dim=0)

            p = flip_rate[idx] * base
            p[y] += 1.0 - flip_rate[idx]
            P_list.append(p)

    P = torch.stack(P_list, dim=0).cpu().numpy()
    noisy = np.array([np.random.choice(num_classes, p=P[i]) for i in range(len(labels))])

    # 3) 
    np.save(cache_file, noisy)
    print("Saved noisy labels to cache:", cache_file)

    # 4) 
    clean_labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    actual_noise_mask = (noisy != clean_labels_np)
    actual_noise_rate = np.mean(actual_noise_mask)

    print(f"[INFO] Actual overall noise rate: {actual_noise_rate:.4f}")
    print(f"[INFO] Number of flipped samples: {actual_noise_mask.sum()} / {len(labels)}")

    # 5) 
    per_class_noise = []
    for c in range(num_classes):
        idx = (clean_labels_np == c)
        if np.sum(idx) == 0:
            per_class_noise.append(0.0)
            continue
        flipped = np.sum(noisy[idx] != clean_labels_np[idx])
        per_class_noise.append(flipped / np.sum(idx))
    print(f"[INFO] Per-class flip rate: {np.round(per_class_noise, 4)}")

    return noisy


# def noisify_instance(
#     n,               # 
#     train_images,    # Tensor  ndarray, shape=(N, C, H, W)
#     train_labels,    # Tensor  ndarray, shape=(N,)
#     num_classes,     # 
#           #  std
#     seed,          #        # 
#     device
# ):
#     # 1)  & 
#     norm_std=0.1
#     model_path = "/home/user/label_noise/resnet50_pathmnist_test.pth"
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = device
#     np.random.seed(int(seed))
#     torch.manual_seed(int(seed))
#     torch.cuda.manual_seed(int(seed))

#     # 2)  flip_rate
#     flip_dist = stats.truncnorm(
#         (0 - n) / norm_std,
#         (1 - n) / norm_std,
#         loc=n,
#         scale=norm_std
#     )
#     flip_rate = flip_dist.rvs(len(train_labels))

#     # 3)  clean 
#     labels = train_labels
#     if isinstance(labels, np.ndarray):
#         labels = torch.from_numpy(labels)
#     labels = labels.to(device).view(-1).long()

#     # 4) 
#     model = models.resnet50(weights=None)
#     model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device).eval()

#     # 5)  model logits  P
#     P_list = []
#     with torch.no_grad():
#         # for idx, (img, y) in enumerate(tqdm(zip(train_images, train_labels),
#         #                                       total=len(train_labels),
#         #                                       desc="Building P")):
#         for idx, (img, y) in enumerate(zip(train_images, train_labels)):
#             # img  ndarray Tensor
#             if isinstance(img, np.ndarray):
#                 x = torch.from_numpy(img).to(device).unsqueeze(0).float()
#                 x = x.permute(0, 3, 1, 2).contiguous() 
#             else:
#                 x = img.to(device).unsqueeze(0)
#             y = int(y)

#             logits = model(x)               # (1, num_classes)
#             A = logits.squeeze(0).clone()   # (num_classes,)

#             #  softmax
#             A[y] = -inf
#             base = F.softmax(A, dim=0)      # sum over j≠y == 1

#             #  flip_rate 
#             p = flip_rate[idx] * base
#             p[y] += 1.0 - flip_rate[idx]

#             P_list.append(p)

#     P = torch.stack(P_list, dim=0).cpu().numpy()  # (N, num_classes)

#     # 6) 
#     new_label = [np.random.choice(num_classes, p=P[i]) for i in range(len(labels))]

#     # 7) 
#     orig = labels.cpu().numpy()
#     noisy = np.array(new_label)
#     print(f'Empirical noise rate: {(noisy != orig).mean():.4f}')

#     record = [[0]*num_classes for _ in range(num_classes)]
#     for a, b in zip(orig, noisy):
#         record[a][b] += 1

#     print('****************************************')
#     print('following is flip percentage:')
#     for i in range(num_classes):
#         tot = sum(record[i])
#         for j in range(num_classes):
#             print(f"{record[i][j]/tot: .2f}", end='\t')
#         print()

#     return noisy





# def noisify_instance(train_data,train_labels,noise_rate,feature_size):
#     if max(train_labels)>10:
#         num_class = 100
#     else:
#         num_class = 10
#     np.random.seed(0)

#     q_ = np.random.normal(loc=noise_rate,scale=0.1,size=1000000)
#     q = []
#     for pro in q_:
#         if 0 < pro < 1:
#             q.append(pro)
#         if len(q)==80000:
#             break

#     w = np.random.normal(loc=0, scale=1, size=(feature_size, num_class))
#     #w = np.random.normal(loc=0,scale=1,size=(32*32*3,num_class))

#     noisy_labels = []
#     for i, sample in enumerate(train_data):
#         sample = sample.flatten()
#         p_all = np.matmul(sample,w)


#         p_all[train_labels[i]] = -1000000


#         p_all = q[i]* F.softmax(torch.tensor(p_all),dim=0).numpy()
#         #print(p_all)

#         p_all[train_labels[i]] = 1 - q[i]
#         noisy_labels.append(np.random.choice(np.arange(num_class),p=p_all/sum(p_all)))

#     over_all_noise_rate = 1 - float(torch.tensor(train_labels).eq(torch.tensor(noisy_labels)).sum())/train_data.shape[0]
#     ind = torch.tensor(train_labels).eq(torch.tensor(noisy_labels))
#     clean_data = train_data[ind]
#     clean_lables = train_labels[ind]

#     return noisy_labels,over_all_noise_rate, clean_data,clean_lables

def create_dir(args):
    save_dir = args.save_dir + '/' + args.dataset + '/' + '%s' % (args.noise_type) + '/' + 'noise_rate_%s' % (
        args.noise_rate) #+ '/' + 'lam=%3f' % (args.lam) + '/' + 'sigma=%3f' % (args.sigma)
    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % (save_dir))

    '''model_dir = save_dir + '/models'

    if not os.path.exists(model_dir):
        os.system('mkdir -p %s' % (model_dir))

    matrix_dir = save_dir + '/matrix'

    if not os.path.exists(matrix_dir):
        os.system('mkdir -p %s' % (matrix_dir))'''
    txt = save_dir + '/lam=%.3f' % (args.lam) + '__' + 'sigma=%.3f' % (args.sigma) + '.txt'
    logs = open(txt, 'w')

    print("save_dir: ",save_dir)
    return  save_dir,  logs




