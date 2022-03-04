#!/usr/bin/env python
# coding: utf-8


import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
from torch import cuda
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from random import shuffle
import pickle
from time import time
# from torch._six import int_classes as _int_classes
from os import system

from data import *
from models import *
from utils import *

device = torch.device('cuda') if cuda.is_available() else torch.device('cpu')
print('Device is', device)

try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    pass

plt.ioff()




###################
###################
### HyperParameters
learning_rate = 5e-3
batch_size = 16
n_epochs = 100
do_save_model = True
using_exist_dataset = False

time_file = 'time_file'
res_address = 'res/%i-%s-%s.pdf'
pkl_address = 'res/%i-%s-%s.pkl'
seq_address = 'res/%i-%s-%s.txt'
models_path = 'models/%i.mdl'
log_file = 'log'
last_batch_test = 'lbt_cnnfold'

using_exist_model = True
start_i = 1
use_two_models = False # Change it to True if you want to use 2 models (one for shorter than 600 and another for else)

# if use_two_models is False, only model_address matters
model_address = 'models/cnnfold600.mdl' # CNNFold-600 (for shorters)
model_address2 = 'models/cnnfold.mdl' # CNNFold

# Just a text in our log file to know what are we running
init_text = "Test | Running CNNFold-mix"

# --------- IMPORTANT PARAMS FOR DIFFERENT TESTING PURPOSES ----------
consider_unpairings = False # Some works consider unpairings in the accuracy
use_blossom = False # BLOSSOM for the post-processing. If it's False, argmax would be used.
shift_allowed = False # for that (S) in the results. True means predicting a pairing with a neighbor considers as correct
dataset = 'align' 
# Choose a dataset from this list:
# * 'align': RNAStrAlign
# * 'align_pk': only pseudoknotted structures in RNAStrAlign
# * 'archive': ArchiveII
# * 'bprna': BpRNA with TR0 and TS0
# * 'bprna_new': the new version of BpRNA with more RNA families
# --------- IMPORTANT PARAMS FOR DIFFERENT TESTING PURPOSES ----------

only_test = True # No training and only testing
if only_test:
    n_epochs = start_i + 1

n_samples = 1000000
prediction_repeat = 2
dynamic_batching = True
train_partition = False
test_partition = False

only_last_output = False # Doesn't matter for testing
only_pk = False 
train_min_len = 0
train_max_len = 6000
test_min_len = 0
test_max_len = 6000
is_2d = True

topk_k = 5


if dataset == 'align':
    seq_to_family_file = 'datasets/seq_to_family.pkl'
    train_dataset = 'datasets/align_train.csv'
    test_dataset = 'datasets/align_test.csv'
    start_point_in_csv = 0
elif dataset == 'align_pk':
    seq_to_family_file = 'datasets/seq_to_family.pkl'
    train_dataset = 'datasets/align_train.csv'
    test_dataset = 'datasets/align_test_pk.csv'
    start_point_in_csv = 0
elif dataset == 'archive':
    seq_to_family_file = 'datasets/archive_seq_to_family.pkl'
    train_dataset = 'datasets/archiveii.csv'
    test_dataset = 'datasets/archiveii.csv'
    start_point_in_csv = 1
elif dataset == 'bprna':
    seq_to_family_file = None
    train_dataset = 'datasets/tr0.csv'
    test_dataset = 'datasets/ts0.csv'
    start_point_in_csv = 1
elif dataset == 'bprna_new':
    seq_to_family_file = None
    train_dataset = 'datasets/bprna_new.csv'
    test_dataset = 'datasets/bprna_new.csv'
    start_point_in_csv = 1
elif dataset == 'TestSetA':
    seq_to_family_file = None
    train_dataset = 'datasets/TrainSetA.csv'
    test_dataset = 'datasets/TestSetA.csv'
    start_point_in_csv = 1
elif dataset == 'TestSetB':
    seq_to_family_file = None
    train_dataset = 'datasets/TrainSetA.csv'
    test_dataset = 'datasets/TestSetB.csv'
    start_point_in_csv = 1
elif dataset == 'archive_mx':
    seq_to_family_file = None
    train_dataset = 'datasets/rnastralign_mx.csv'
    test_dataset = 'datasets/archiveII_mx.csv'
    start_point_in_csv = 1
elif dataset == 'sample':
	seq_to_family_file = 'datasets/seq_to_family.pkl'
	train_dataset = 'datasets/sample_file.csv'
	test_dataset = 'datasets/sample_file.csv'
	start_point_in_csv = 0
# elif dataset == 'bprna_new_tr0':
#     seq_to_family_file = None
#     train_dataset = 'datasets/tr0.csv'
#     test_dataset = 'datasets/ts0.csv'
#     start_point_in_csv = 1


model = ResNet()
model2 = ResNet()


system('mkdir %s' % (res_address.split('/')[0]))
system('mkdir %s' % (models_path.split('/')[0]))

train_size_ratio = .8
onehot_inputs = True


##################################
##################################
### Creating / Loading the dataset
if not using_exist_dataset:
    print('Creating the dataset ...')
    all_families = ['f']
    if dataset == 'align':
        all_families = ['16S_rRNA_database', 
                         '5S_rRNA_database',
                         'group_I_intron_database',
                         'RNaseP_database',
                         'SRP_database',
                         'telomerase_database',
                         'tmRNA_database',
                         'tRNA_database'] # Stralign
    elif dataset == 'archive':
        all_families = ['16s', '23s', '5s', 'grp1', 'grp2', 'RNaseP', 'srp', 'telomerase', 'tmRNA', 'tRNA'] # Archive
    elif dataset == 'bprna':
        all_families = ['RNP', 'RFAM', 'tmRNA', 'SRP', 'CRW', 'SPR']

    train_families = [
                        # '16S_rRNA_database', 
                         #'5S_rRNA_database',
                         'group_I_intron_database',
                         'RNaseP_database',
                         'SRP_database',
                         'telomerase_database',
                         'tmRNA_database',
                         # 'tRNA_database'
                         ] # Change it if you want to train on some specific families (o.w. set it to None)
    train_families = None # set it to None to considers all families in training

    

    sequences, structures, families = read_csv_dataset(train_dataset, 
                                         min_length_limit=train_min_len,
                                         max_length_limit=train_max_len,
                                         n_first_samples=n_samples,
                                         seq_start_point=start_point_in_csv,
                                         do_shuffle=True,
                                         seq_to_family_file=seq_to_family_file,
                                         family_set=train_families)
    if train_partition:
        sequences, structures = partition(sequences, structures, 500, 400)
    train_samples = list(zip(sequences, structures, families))

    sequences, structures, families = read_csv_dataset(test_dataset, 
                                         min_length_limit=test_min_len,
                                         max_length_limit=test_max_len,
                                         n_first_samples=n_samples,
                                         seq_start_point=start_point_in_csv,
                                         do_shuffle=True,
                                         seq_to_family_file=seq_to_family_file,
                                         family_set=train_families)
    if test_partition:
        sequences, structures = partition(sequences, structures, 500, 400)
    test_samples = list(zip(sequences, structures, families))


    pickle.dump(train_samples, open('train_list', 'wb+'))
    pickle.dump(test_samples, open('test_list', 'wb+'))
else:
    train_samples = pickle.load(open('train_list', 'rb'))
    test_samples = pickle.load(open('test_list', 'rb'))
    
        
if dynamic_batching: # Should be False (was for some testing for creating batches with different sequence lengths)
	# I afraid setting it to True would break something (at least for the CNN)
    r_sampler = SequentialSampler(train_samples)
    b_sampler = BatchSampler(r_sampler, max_size=2000, max_batch_size=16, max_interval=100, max_length=train_max_len)
    train_loader = DataLoader(train_samples,
                            shuffle=False,
                            batch_sampler=b_sampler,
                            collate_fn=lambda b:collate(b, onehot_inputs))

    r_sampler2 = SequentialSampler(test_samples)
    b_sampler2 = BatchSampler(r_sampler2, max_size=2000, max_batch_size=1, max_interval=100, max_length=test_max_len)
    test_loader = DataLoader(test_samples,
                            shuffle=False,
                            batch_sampler=b_sampler2,
                            collate_fn=lambda b:collate(b, onehot_inputs))
else:
    train_loader = DataLoader(train_samples,
                            shuffle=True,
                            batch_size=1,
                            collate_fn=lambda b:collate(b, onehot_inputs))

    test_loader = DataLoader(test_samples,
                            shuffle=True,
                            batch_size=1,
                            collate_fn=lambda b:collate(b, onehot_inputs))


print('#train: %i | #test: %i' % (len(train_samples), len(test_samples)))
lengths = np.array([len(seq[0]) for seq in train_samples])
plt.hist(lengths)

plt.figure()
lengths = np.array([len(seq[0]) for seq in test_samples])
plt.hist(lengths)




cnn1 = model.to(device)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print('#params', count_parameters(cnn1))

cnn2 = model2.to(device)
if using_exist_model:
    cnn1 = torch.load(model_address, map_location=device)
    cnn2 = torch.load(model_address2, map_location=device)
cnn1.train()
cnn2.train()


optimizer = optim.Adam(cnn1.parameters(), lr=learning_rate)
threshold = .0


running_loss = 0
training_loss = []
time_list = [[] for i in range(3000)]
moving_loss = 0

with open(log_file, 'a+') as f:
    f.write('============= START ===============\n')
    f.write('%s\n' % init_text)

for epoch in range(start_i, n_epochs, 1):
    cnn1.train()
    total_loss = 0
    
    train_accuracy = 0
    loss_hat_prob = torch.zeros(1)

    acc = []
    precs, recs = [], []
    lengths = []
    cnt = 0
    sum_elems = 0
    losses = []

    for i, (x, y, f) in enumerate(train_loader):
        if only_test:
            acc.append(0)
            break
        print('Training ...')
        cnt += y.size(0)

        optimizer.zero_grad()

        cnn1.train()
        cnn2.train()

        x = x.to(device) # [B, 8, N, N]
        y = y.to(device) # [B, 8, N, N]
        
        y_hat = cnn1(x, test=only_last_output, repeat=prediction_repeat) # [B*x, 1, N, N]
        
        if not only_last_output:
            repeat = y_hat.size(0)//y.size(0)
            y = y.repeat(repeat, 1, 1, 1) # repeat for the loss | [B*x, 1, N, N]
        
        
        mask = (y!=-1).float() # [B*x, 1, N, N]
        y_hat = y_hat * mask # [B*x, 1, N, N]

        if not is_2d: # For NLLLoss --> argmax
            y_sum = mask.max(-1)[0] 
            y_new = (y.argmax(-1).float()+1) * y_sum - 1
            y_hat = y_hat.squeeze().transpose(-1, -2)
        
            loss_hat_prob = F.nll_loss(y_hat, y_new.long(), ignore_index=-1)
            # losses = losses + loss_hat_prob
        else: # For MSELoss
            y_new = y
            loss_hat_prob = F.mse_loss(y_hat, y_new)
            # loss_hat_prob = F.cross_entropy(y_hat.transpose(1, 2), y_new.transpose(1, 2).argmax(1))
            # loss_hat_prob = F.binary_cross_entropy(y_hat, y_new)
        
        y_new = y_new[-x.size(0):] # [B, 1, N, N]
        y_hat = y_hat[-x.size(0):] # [B, 1, N, N]
        
        if moving_loss == 0:
            moving_loss = loss_hat_prob.item()
        else:
            moving_loss = .95 * moving_loss + .05 * loss_hat_prob.item()

        loss_hat_prob.backward()
        optimizer.step()

        if not is_2d:
            y_hat = y_hat.argmax(-2).float()

        # ######################
        # ######################
        # If there is no softmax in the model, uncomment this
        # e.g. while testing with cross_entropy loss
        # y_hat = F.softmax(y_hat, -2)
        # ######################
        ######################

        y_hat = binarize(y_hat, threshold)
        f1, prec, rec = calculate_f1(y_hat, y_new, None, is_2d=is_2d, 
            consider_unpairings=consider_unpairings, reduction=False, shift_allowed=shift_allowed, prec_rec=True)
        f1 = list(np.array(f1.cpu()))
        prec = list(np.array(prec.cpu()))
        rec = list(np.array(rec.cpu()))
        acc.extend(f1)
        precs.extend(prec)
        recs.extend(rec)

        if not i%500:
            seq = str(y.size(-1))
            sample_idx = 0 # epoch%y_hat.size(0)
            n = y_hat[sample_idx].size(-1)
            if not is_2d:
                out = torch.zeros((n, n))
                out[range(n), y_hat[sample_idx].long().cpu().data] = 1
                out = out * mask[sample_idx]
            else:
                out = y_hat[sample_idx, 0] * mask[sample_idx, 0] # [N, N]
                y = y[sample_idx, 0] * mask[sample_idx, 0]

            print('[%i, %i]\tMoving Loss: %.6f | Moving F1: %.3f \t[This: %.3f]' % 
                        (cnt, x.size(-1), moving_loss, np.mean(acc), acc[sample_idx-y_new.size(0)]))
            with open(log_file, 'a+') as f:
                f.write('[%i, %i]\tMoving Loss: %.6f | Moving F1: %.3f \t[This: %.3f]\n' % 
                        (cnt, x.size(-1), moving_loss, np.mean(acc), acc[sample_idx-y_new.size(0)]))

        total_loss += loss_hat_prob.item()
    
    train_accuracy = np.mean(acc)

    if do_save_model:
        torch.save(cnn1, models_path % epoch)
    training_loss.append(loss_hat_prob.item())

    precision_list, recall_list, f1_list = [], [], []
    acc = []
    precs, recs = [], []
    acc_families = {f:[[], []] for f in families}
    families = []
    pk_res = np.array([0, 0, 0, 0])
    with open(time_file, 'w+') as f:
        pass # Just to make the previous time_file empty
    with torch.no_grad():

        for i, (x, y, family) in enumerate(test_loader):
            t1 = time()
            # cnn1.eval()
            # cnn2.eval()

            x = x.to(device) # [B, 1, N, N]
            y = y.to(device) # [B, 1, N, N]

            if y.size(-1) <= 600 or not use_two_models:
                y_hat_test = cnn1(x, test=True, repeat=prediction_repeat) # [B, 1, N, N]
            else:
                y_hat_test = cnn2(x, test=True, repeat=prediction_repeat) # [B, 1, N, N]

            mask = (y!=-1).float() # [B, 1, N, N]
            y_hat_test = y_hat_test * mask # [B, 1, N, N]
      
            # ######################
            # ######################
            # y_hat_test = F.softmax(y_hat_test, -2)
            # ######################
            # ######################

            
            # yht_thresh = (y_hat_test > 1./x.size(-1)).float()
            t2 = time()
            y_hat_test = binarize(y_hat_test, threshold, use_blossom=use_blossom)
            t3 = time()

            pk_res += f1_pk(y_hat_test, y)

            with open(time_file, 'a+') as f:
                f.write('%i\t%.8f\t%.8f\n' % (y_hat_test.size(-1), t2-t1, t3-t2))
            # yht_thresh = keep_topk(y_hat_test, k=topk_k)
            # yht_thresh = (yht_thresh * yht_thresh>0).float()             
            
            time_list[y.size(-1)].append(t2-t1)

            f1, prec, rec = calculate_f1(y_hat_test, y, None, is_2d=is_2d, 
                consider_unpairings=consider_unpairings, reduction=False, shift_allowed=shift_allowed, prec_rec=True, only_pk=only_pk)
            # if only_pk:
            #     f1 = recall(y_hat_test, y, None, reduction=False, shift_allowed=shift_allowed,
            #         consider_unpairings=consider_unpairings, is_2d=is_2d, only_pk=only_pk)
            f1 = list(np.array(f1.cpu()))
            prec = list(np.array(prec.cpu()))
            rec = list(np.array(rec.cpu()))

            acc.extend(f1)
            precs.extend(prec)
            recs.extend(rec)
            # f1 = calculate_f1(y_hat_test, y, None, is_2d=is_2d, 
            #     consider_unpairings=consider_unpairings, reduction=False, shift_allowed=shift_allowed, prec_rec=True)

            for k in range(y.size(0)):
                precision, recall, f1_score = evaluate_exact(y_hat_test[k].cpu(), y[k].cpu())
                precision_list.append(precision.item())
                recall_list.append(recall.item())
                f1_list.append(f1_score.item())

            # f1 = list(np.array(f1.cpu()))
            # acc.extend(f1)
            for val, fam in zip(f1, family):
                acc_families[fam][0].append(val)
                acc_families[fam][1].append(y.size(-1))
            lengths.extend([y.size(-1)]*y.size(0))
            families.extend(family)

            if i%100==0:
                L = str(y.size(-1))
                # print('PK', pk_res)
                idx = ['A', 'U', 'U', 'G', 'G', 'C']

                sample_idx = epoch%y_hat_test.size(0)

                if not is_2d:
                    n = y_hat_test.size(-1)
                    out = torch.zeros((n, n))
                    out[range(n), y_hat_test[sample_idx].long().cpu().data] = 1
                    out = out * mask[sample_idx]
                else:
                    out = y_hat_test[sample_idx, 0] * mask[sample_idx, 0] # [N, N]
                    y = y[sample_idx, 0] * mask[sample_idx, 0] 
                # tmp = torch.min(torch.argmax(x[sample_idx], 0), 1)[0]
                # seq = [idx[int(i)] for i in tmp]
                with open(log_file, 'a+') as f:
                    f.write('PK: %s\n' % str(pk_res))
                    f.write('%i\t%f\t[%f]\n' % (x.size(-1), acc[sample_idx-x.size(0)], np.mean(acc)))
                # with open(seq_address % (epoch, 'TEST-%i' % (i), L), 'a+') as f:
                #     f.write(''.join(seq))
                pickle.dump(np.array(torch.argmax(out, -1).cpu()), open(pkl_address % (epoch, 'TEST-%i' % (i), L), 'wb+'))
                # plot(out, y, res_address % (epoch, 'TEST-%i' % (i), L))


        with open('time_list', 'w+') as f:
            # print(str(time_list))
            for i in range(3000):
                for t in time_list[i]:
                    f.write(str(t))
                f.write('\n')


        # time_list = [np.mean(np.array(i)) for i in time_list]
        # pickle.dump()

        test_acc = np.mean(acc)
        w_test_acc = np.sum(np.array(acc)*np.array(lengths)) / np.sum(lengths)
        test_acc_families = {}
        for family in acc_families:
            if family in acc_families:
                a, w = acc_families[family]
                acc_n = np.mean(a)
                acc_w = np.sum(np.array(a)*np.array(w)/np.sum(w))
                test_acc_families[family] = [acc_n, acc_w]

        print('Epoch %i [Train %f | Test %f | Weighted Test %f | Training loss %f]\n' % (epoch, 
                                                                train_accuracy,
                                                                test_acc,
                                                                w_test_acc,
                                                                total_loss))

        family_acc_template = 'Family-based F1 (family: F1 | Weighted-F1):\n'
        for family in all_families:
            if family in test_acc_families:
                family_acc_template += '%25s:\t %f\n' % (family, test_acc_families[family][0])
        with open(last_batch_test, 'w+') as f:
            for a, l, fam in zip(acc, lengths, families):
                f.write('%f, %i, %s\n' % (a, l, fam))
        with open(log_file, 'a+') as f:
            # f.write('PK %s\n' % str(pk_res))
            f.write('Epoch %i [Train %f | Test %f | Weighted Test %f | Training loss %f]\n' % (epoch, 
                                                                train_accuracy,
                                                                test_acc,
                                                                w_test_acc,
                                                                total_loss))
            f.write('prec %f | rec %f\n' % (np.mean(precs), np.mean(recs)))
            f.write(family_acc_template)


# Just added this comment so that the last line of the code isn't the last line anymore :)
