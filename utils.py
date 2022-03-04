import torch
print(torch.__version__)
print(torch.cuda.is_available())

from matplotlib import pyplot as plt

import numpy as np
import torch.nn as nn
from torch import cuda
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
import os
from random import shuffle
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment as lsa
import networkx as nx
import pickle
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

colors = [[1., 1., 1., 1.],
         [0., 0., 0., 1.],
         [1., .1, .1, .8],
         [0., .8, .3, .8]]

my_cm = LinearSegmentedColormap.from_list('Accuracy', colors)



device = torch.device('cuda') if cuda.is_available() else torch.device('cpu')


def keep_topk(mat, k=3):
    v, idx = torch.topk(mat, k, dim=-1)
    n = idx.size(-2)
    new_mat = torch.zeros_like(mat)
    for i in range(idx.size(0)):
        for j in range(k):
            new_mat[i, 0, range(n), idx[i, 0, :, j]] = mat[i, 0, range(n), idx[i, 0, :, j]]
    return new_mat.float()


def blossom(mat):
    A = mat.clone()
    n = A.size(-1)
    mask = torch.eye(n, device=device) * 2
    big_a = torch.zeros((2*n, 2*n), device=device)
    big_a[:n, :n] = A
    big_a[n:, n:] = A
    big_a[:n, n:] = A*mask
    big_a[n:, :n] = A*mask
    G = nx.convert_matrix.from_numpy_matrix(np.array(big_a.cpu().data))
    pairings = nx.matching.max_weight_matching(G)
    y_out = torch.zeros_like(A)
    for (i, j) in pairings:
        if i>n and j>n:
            continue
        y_out[i%n, j%n] = 1
        y_out[j%n, i%n] = 1
    return y_out

def binarize(y_hat, threshold=0, use_blossom=False):
    """
    First term is necessary to ignore paddings for example
    y_hat in shape [B, 1, N, N]
    """
    
    if not use_blossom:
        return (y_hat>threshold).float()*(y_hat == y_hat.max(dim=-1, keepdim=True)[0]).float()
    else:
        # threshold_mask = (y_hat>.005).float()
        # y_hat = y_hat * threshold_mask
        y_hat = keep_topk(y_hat, 3)
        new_y_hat = torch.zeros_like(y_hat[:, 0], device=device)
        for i, sample in enumerate(y_hat[:, 0]):
            out = blossom(sample.squeeze())
            new_y_hat[i] = out
        return new_y_hat.unsqueeze(1) # [B, 1, N, N]

def binarize_3d(y_hat):
    return (y_hat>0).float()*(y_hat[:, 1] >= y_hat[:, 0]).float()


def create_pk_mask(target):
    """
    Returning a mask with the same size as target
    1 where the row or column has PK
    """
    n = target.size(-1)
    mask = torch.zeros_like(target, device=device)
    for t in range(target.size(0)):
        pk = [False for _ in range(n)]
        pk_idx = []
        pairing = torch.argmax(target[t, 0], -1)
        for i in range(n):
            if i in pk_idx:
                continue
            is_pk = False
            j = pairing[i]
            if i < j:
                for k in range(i, j):
                    if pairing[k]<i or pairing[k]>j:
                        pk[i] = True
                        pk[k] = True
                        pk[j] = True
                        pk_idx.append(i)
                        pk_idx.append(j)
                        pk_idx.append(k)
                        continue
            
    
        idx = list(set(pk_idx))
        mask[t, 0, idx] = 1
        mask[t, 0, :, idx] = 1
    return mask

def has_pk(pairings):
    """
    pairings is the pairing list as a torch.Tensor
    """
    for idx in range(len(pairings)):
        i, j = idx, pairings[idx].item()
        if i>j:
            i, j = j, i
        if i==j:
            continue
        if torch.max(pairings[i:j]) > j:
            return True
        if torch.min(pairings[i:j]) < i:
            return True
    return False

def f1_pk(y_hat, y):
    """
    Return pk for [TN, FN, FP, TP]
    """
    pk = np.array([0, 0, 0, 0])
    for y1, y2 in zip(y_hat, y):
        y1, y2 = y1.squeeze(), y2.squeeze()
        i = has_pk(y1.argmax(-1))
        j = has_pk(y2.argmax(-1))
        pk[i*2+j] += 1
    return pk


def precision(y_hat, y, sampling_f=binarize, shift_allowed=False, 
              is_2d=True, consider_unpairings=True, reduction=True, only_pk=False):
    """
    Precision of y_hat w.r.t y which is the TP / (TP+FP)
    This function also convert the y_hat to a binary one
    y_hat and y are in shape [B, 1, N, N]
    """
    if consider_unpairings:
        mask2 = 1.
    else:
        mask2 = 1 - torch.eye(y_hat.size(-1), device=device) # [N, N]
    if is_2d:
        if sampling_f:
            y_hat = sampling_f(y_hat.squeeze())
#         y_hat = y_hat.view(y.squeeze().shape)
        
        mask = (y!=-1).float() # [B, 1, N, N]
        y_hat = y_hat * mask # [B, 1, N, N]

        if shift_allowed:
            kernel = torch.ones((1, 1, 3, 3), device=device)
            kernel[0, 0, [0, 0, 2, 2], [0, 2, 0, 2]] = 0
            y = (F.conv2d(y, kernel, padding=1)>0).float()

        if only_pk:
            pk_mask = create_pk_mask(y)
        else:
            pk_mask = 1.
        tp = torch.sum(y_hat * y * mask * mask2 * pk_mask, dim=(1, 2, 3))
        prec = (tp / (torch.sum(y_hat * mask * mask2 * pk_mask, dim=(1, 2, 3)) + 1e-15))
        if reduction:
            return torch.mean(prec).item()
        return prec
    else:
        mask = (y>=0)
        tp = torch.sum((y_hat == y) * mask).item()
        return tp / torch.sum(mask).item()


def recall(y_hat, y, sampling_f=binarize, shift_allowed=False, 
           is_2d=True, consider_unpairings=True, reduction=True, only_pk=False):
    """
    Recall of y_hat w.r.t y which is the TP / (TP+FN)
    This function also convert the y_hat to a binary one
    y_hat and y are in shape [B, 1, N, N]
    """
    if consider_unpairings:
        mask2 = 1.
    else:
        mask2 = 1 - torch.eye(y_hat.size(-1), device=device) # [N, N]
        
    if is_2d:
        if sampling_f:
            y_hat = sampling_f(y_hat.squeeze())

        if shift_allowed:
            kernel = torch.ones((1, 1, 3, 3), device=device)
            kernel[0, 0, [0, 0, 2, 2], [0, 2, 0, 2]] = 0
            y_hat = (F.conv2d(y_hat, kernel, padding=1)>0).float()

            
        mask = (y!=-1).float() # [B, 1, N, N]
        pk_mask = 1.
        if only_pk:
            pk_mask = create_pk_mask(y)
        tp = torch.sum(y_hat * y * mask * mask2 * pk_mask, dim=(1, 2, 3))
        recall = (tp / (torch.sum(y * mask * mask2 * pk_mask, dim=(1, 2, 3)) + 1e-15))
        if reduction:
            return torch.mean(recall).item()
        return recall
    else:
        mask = (y>=0)
        tp = torch.sum((y_hat == y) * mask).item()
        return tp / torch.sum(mask).item()
    

def calculate_accuracy(y_hat, y, binarize=binarize):
    """
    Accuracy, ****** better to not use it!!! ******
    """
#     return precision(y, y_hat)
    y = y.float()
    binary_y = binarize(y_hat)
    return torch.sum(binary_y == y.squeeze()).item()  / float(y.size(0)*y.size(-1)*y.size(-2))


def evaluate_exact(pred_a, true_a):
    """
    reference: https://github.com/ml4bio/e2efold/blob/5d8d59377f787695357d780506a2eadf32dead9c/e2efold/common/utils.py
    ICLR2020 paper
    """
    tp_map = torch.sign(torch.Tensor(pred_a)*torch.Tensor(true_a))
    tp = tp_map.sum()
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    fp = pred_p - tp
    fn = true_p - tp
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1_score = 2*tp/(2*tp + fp + fn)
    return precision, recall, f1_score


def calculate_f1(y_hat, y, sampling_f=None, shift_allowed=False, 
                 is_2d=True, consider_unpairings=True, reduction=True, prec_rec=False, only_pk=False):
    """
    F1-score which is 2*precision*recall / (precision + recall)
    That 1e-10 is for numerical issues (devided by zero)
    y_hat and y have shapes [B, 1, N, N]
    """
    y = y.float()
    if sampling_f:
        y_hat = sampling_f(y_hat) # [B, 1, N, N]
    
    prec = precision(y_hat, y, None, shift_allowed=shift_allowed,
                 is_2d=is_2d, consider_unpairings=consider_unpairings, 
                 reduction=reduction, only_pk=only_pk)
    
    rec = recall(y_hat, y, None, shift_allowed=shift_allowed,
                 is_2d=is_2d, consider_unpairings=consider_unpairings, 
                 reduction=reduction, only_pk=only_pk)
    
    if prec_rec:
        return 2 * prec * rec / (prec + rec + 1e-15), prec, rec
    return 2 * prec * rec / (prec + rec + 1e-15)



def plot3(y_out1, y_out2, y, img_file):
    fig, ax = plt.subplots(1, 3)
    fig.tight_layout()
    y_out1 = y_out1.cpu().data
    y_out2 = y_out2.cpu().data
    y = y.cpu().data
    ax[0].cla()
    ax[0].axis('off')
    ax[0].matshow(y_out1, cmap=plt.cm.viridis)

    ax[1].cla()
    ax[1].axis('off')
    ax[1].matshow(y_out2, cmap=plt.cm.viridis)

    ax[2].cla()
    ax[2].axis('off')
    ax[2].matshow(y, cmap=plt.cm.viridis)
    
    fig.savefig(img_file, transparent=True, dpi=int(y_out2.size(-1)*3))
    plt.pause(.001)


def plot(y_out, y, img_file):
    fig, ax = plt.subplots(1, 3)
    fig.tight_layout()
    y_out = y_out.cpu().data
    y = y.cpu().data
    ax[0].cla()
    ax[0].axis('off')
    ax[0].matshow(y_out*2+y, cmap=my_cm)

    ax[1].cla()
    ax[1].axis('off')
    ax[1].matshow(y_out, cmap=plt.cm.viridis)

    ax[2].cla()
    ax[2].axis('off')
    ax[2].matshow(y, cmap=plt.cm.viridis)
    
    fig.savefig(img_file, transparent=True, dpi=int(y_out.size(-1)*3))
    plt.pause(.001)


########################################################
########################################################
## Reading / Loading / Creating the dataset
bases = ['A', 'U', 'G', 'C']

## Dataset
def read_csv_dataset(csv_file, 
                     max_length_limit=200,
                     min_length_limit=1,
                     n_first_samples=1000000,
                     seq_start_point=0,
                     do_shuffle=True,
                     seq_to_family_file='datasets/seq_to_family.pkl',
                     family_set=None):
    sequences = []
    pairings = []
    families = []
    print(csv_file)
#     s2f_dict = {}
    if seq_to_family_file:
        seq_to_family = pickle.load(open(seq_to_family_file, 'rb'))
    else:
        seq_to_family = {i:0 for i in range(100000)}
    with open(csv_file, 'r') as f_csv:
        samples = f_csv.read().split('\n')
        if do_shuffle:
            shuffle(samples)
        for sample in samples:
            if not sample:
                continue
            elements = sample.split(',')
            if min_length_limit<len(elements)<max_length_limit:
                seq = elements[seq_start_point].upper()
                if len(seq) != len(elements)-1-seq_start_point:
                    print('Something is wrong with this sequence:', seq)
                    continue
                
                cnt = 0
                for b in bases:
                    cnt += seq.count(b)
                    
                # Handling some exceptions like N in the seq
#                 if cnt != len(seq):
#                     continue
                if not seq or seq in sequences:
                    continue
                try:
                    pairing = [int(e) for e in elements[seq_start_point+1:]]
                except:
                    print(elements)
                if np.max(pairing) >= len(pairing):
                    continue
                if seq_to_family_file:
                    family = seq_to_family[seq]
                else:
                    # family = elements[0].split('_')[1]
                    family = 'f'
                if (not family_set) or family in family_set:
                    families.append(family)
                    sequences.append(seq)
                    pairings.append(pairing)
#                 s2f_dict[seq] = families[-1]
            if len(sequences) == n_first_samples:
                break
    return sequences, pairings, families


def split_train_test(X, Y, train_size_ratio=.8, onehot=True):
    n_samples = len(X)

    tr = int(train_size_ratio*n_samples)
    X_train = X[:tr]
    Y_train = Y[:tr]
    X_test = X[tr:]
    Y_test = Y[tr:]
    # X_train = [create_matrix(s, onehot=onehot) for s in X_train]
    # X_test = [create_matrix(s, onehot=onehot) for s in X_test]
    # Y_train = [pairing_to_matrix(s) for s in Y_train]
    # Y_test = [pairing_to_matrix(s) for s in Y_test]

    train_samples = list(zip(X_train, Y_train))
    test_samples = list(zip(X_test, Y_test))
    del X
    del Y
    return train_samples, test_samples


def split_train_test_torch(X, Y, train_size_ratio=.8):
    n_samples = X.shape[0]

    tr = int(train_size_ratio*n_samples)
    X_train = torch.tensor(X[:tr]).float()
    Y_train = torch.tensor(Y[:tr]).float()
    X_test = torch.tensor(X[tr:]).float()
    Y_test = torch.tensor(Y[tr:]).float()
    del X
    del Y
    return X_train, X_test, Y_train, Y_test


def onehot_to_seq(possibilities):
    seq = [-1] * possibilities.shape[1]
    z, x, y = np.where(possibilities==1)
    for i, j, k in zip(x, y, z):
        seq[i] = bases[int(k / 4)]
    return ''.join(seq)


def create_matrix(sequence, onehot=True, min_dist=3):
    """
    Create input matrix which is 16xNxN or 1xNxN according to the onehot value
    At the moment works faster than the previous one (matrix multiplication vs normal loop)
    
    We have 16 different pairing types w.r.t [A, U, G, C]
        0, 5, 10, 15 are self_loops (unpair) --> 1
        1, 4, 6, 9, 11, 14 are pairings --> 6
        others are invalid --> 1
        = 8 modes (channels)
    """
    n = len(sequence)
    invalid = []
    seq = []
    for i, s in enumerate(sequence):
        if s not in bases:
            invalid.append(i)
            seq.append(0)
        else:
            seq.append(bases.index(s))

    seq = torch.tensor(seq, device=device)
    if onehot:
        mat = torch.zeros((17, n, n), device=device)
    else:
        mat = torch.zeros((1, n, n), device=device)


    q2 = seq.repeat(n, 1)
    q1 = q2.transpose(1, 0)    
    t = torch.stack(((torch.abs(q1-q2)==1).long(), torch.eye(n, device=device).long()))
    mask = torch.max(t, 0)[0]
    flat_mat = ((q1*4+q2+1) * mask)
    
    for i in range(1, min_dist+1):
        flat_mat[range(n-i), range(i, n)] = 0
        flat_mat[range(i, n), range(n-i)] = 0
    
#     flat_mat[invalid] = 0
#     flat_mat[:, invalid] = 0
    flat_mat = flat_mat.unsqueeze(0)
    
    if onehot:
        idx2 = torch.arange(n).repeat(n, 1)
        idx1 = idx2.transpose(1, 0).reshape(-1)
        idx2 = idx2.reshape(-1)
        mat[flat_mat.reshape(-1), idx1, idx2] = 1
#         mat[q2.reshape(-1), idx1, idx2] = 1
#         mat[4+q1.reshape(-1), idx1, idx2] = 1
        mat = mat[1:]
        mat8 = mat[[1, 4, 6, 9, 11, 14]]
        # mat8 = mat[[1, 6, 11]]
        # mat8 = mat8 + mat8.transpose(-1, -2)

        mat8 = torch.cat((mat8, torch.sum(mat[[0, 5, 10, 15]], 0).unsqueeze(0)), 0)
        mat8 = torch.cat((mat8, 1-torch.sum(mat8, 0).unsqueeze(0)), 0)
        return mat8
    return flat_mat


def pairing_to_matrix(pairing):
    """
    Convert the pairing list to the pairing matrix
    """
    n = len(pairing)
    y = torch.zeros((1, n, n), device=device)
    y[0, range(n), pairing] = 1
    return y

def matrix_to_onehot(mat):
    """
    mat is a matrix with size (1, n, n)
        with values between 0 and 16
    return a matrix with size (16, n, n)

    """
    n = mat.shape[-1]
    scale = 1
    if np.max(mat) == 1:
        scale = 16
    onehot = np.zeros((16, n, n), devcie=device)
    for i in range(n):
        for j in range(n):
            if mat[0, i, j]:
                onehot[int(mat[0, i, j])*scale-1, i, j] = 1
    return onehot


def pad_matrix(mat, final_size, insert_mode='m', padding_value=-1):
    """
    mat has size (k, n, n)
    Create a matrix with size (k, final_size, final_size)
    put the mat in it according to insert_mode:
        m: put mat at the center
        lt: left top
        r: random
    """
    n = mat.shape[-1]
    if final_size < n:
        print('Final size should be greater or equal than the current size!')
    final_mat = torch.ones((mat.shape[0], final_size, final_size), device=device)
    final_mat = final_mat * padding_value
    if insert_mode == 'm':
        i = final_size//2 - n//2
        final_mat[:, i:i+n, i:i+n] = mat
    return final_mat

    
def pairing_to_dot(pairing):
    mapping = ['.', ')', ']', '}', '{', '[', '(']
    n = len(pairing)
    dot = ['.'] * n
    dot_bracked = np.array([0] * n)
    for i in range(n):
        if dot_bracked[i] != 0:
            continue
        j = pairing[i]
        if i==j:
            dot_bracked[i] = 0

        c = dot_bracked[min(i, j):max(i, j)+1]
        level = (np.sum(c)>0) * np.max(c) + int(i!=j)

        dot_bracked[min(i, j)] = -level
        dot_bracked[max(i, j)] = level

    return ''.join([mapping[c] for c in dot_bracked])




