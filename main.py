import config
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse
from PCNN_MAXIUM import PCNN_MAXIUM
import os
import pickle

import torch
from sklearn.preprocessing import MultiLabelBinarizer
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Policy
from tree import Tree
os.environ['CUDA_VISIBLE_DEVICES'] = '0'




def calc_sl_loss(probs, update=True):
    y_true = Variable(torch.from_numpy(conf.batch_label)).cuda().float()
 #   y_true = y_true.long()
    loss = criterion(probs, y_true)
    if update:
        tree.n_update += 1
        base_model_optimizer.zero_grad()
        loss.backward()
        base_model.optimizer.step()
    return loss


def forward_step_sl():#详细介绍

    # TODO can reuse logits
    if conf.global_ratio > 0:
        logits, probs = policy.base_model()#[64,53] 每个flat关系的概率
        global_loss = calc_sl_loss(probs, update=False)#一个tensor单值 在论文中就是flat_loss
    else:
        probs = None
        global_loss = 0
    
    if conf.flat_probs_only:
        policy.sl_loss = global_loss
        return global_loss, probs

    policy.bag_vec = logits
    bag_ids = conf.bag_ids
    cur_batch_size = len(bag_ids) #一般都是64
    cur_class_batch = np.zeros(cur_batch_size, dtype=int)
    for t in range(conf.n_steps_sl):#3
        #print("cur_class_batch", cur_class_batch)     
        next_classes_batch = tree.p2c_batch(cur_class_batch)#[batch,上一阶段标签的子标签]，可以看成第n层及他之前的标签
        if len(next_classes_batch[0]) == 0:
            print(t, cur_class_batch)
            break
        #print("next_classes_batch", next_classes_batch, len(next_classes_batch))
        next_classes_batch_true, indices, next_class_batch_true, bag_ids = tree.get_next(cur_class_batch, next_classes_batch, bag_ids)
        #print("next_classes_batch_true", next_classes_batch_true, type(next_classes_batch_true))
        #print("indices", indices, len(indices))
        #print("next_class_batch_true", next_class_batch_true, type(next_class_batch_true))
        policy.step_sl(conf, cur_class_batch, next_classes_batch, next_classes_batch_true)#加入local_loss
        cur_class_batch = next_class_batch_true
        policy.duplicate_bag_vec(indices)
    policy.sl_loss /= conf.n_steps# n_steps=17 不知道这里为什么取17
    policy.sl_loss = (1 - conf.global_ratio) * policy.sl_loss + conf.global_ratio * global_loss
    return global_loss, probs
def train_one_step():
        policy.base_model.embedding.word = torch.from_numpy(conf.batch_word).cuda()
        policy.base_model.embedding.pos1 = torch.from_numpy(conf.batch_pos1).cuda()
        policy.base_model.embedding.pos2 = torch.from_numpy(conf.batch_pos2).cuda()
        policy.base_model.encoder.mask = torch.from_numpy(conf.batch_mask).cuda()
        policy.base_model.selector.scope = conf.batch_scope
        policy.base_model.selector.attention_query = torch.from_numpy(conf.batch_attention_query).cuda()
        policy.base_model.selector.label = torch.from_numpy(conf.batch_label).cuda()
        policy.base_model.classifier.label = torch.from_numpy(conf.batch_label).cuda()
def test_one_step():
        policy.base_model.embedding.word = torch.from_numpy(conf.batch_word).cuda()
        policy.base_model.embedding.pos1 = torch.from_numpy(conf.batch_pos1).cuda()
        policy.base_model.embedding.pos2 = torch.from_numpy(conf.batch_pos2).cuda()
        policy.base_model.encoder.mask = torch.from_numpy(conf.batch_mask).cuda()
        policy.base_model.selector.scope = conf.batch_scope

def train_step_sl():
    # if conf.load_model:
    #     policy.load_state_dict(torch.load("./checkpoint/epoch_" + str(1)))
    policy.train()
    best_auc = 0.0
    best_p = None
    best_r = None
    best_epoch = 0
    for epoch in range(1, conf.max_epoch + 1):
        print('Epoch ' + str(epoch) + ' starts...')
        loss_total = 0
        tree.cur_epoch = epoch
        np.random.shuffle(conf.train_order)

        conf.acc_NA.clear()
        conf.acc_not_NA.clear()
        conf.acc_total.clear()

        for batch_num in range(conf.train_batches):
            conf.get_train_batch(batch_num)
            train_one_step()
            global_loss, flat_probs = forward_step_sl()
            policy_optimizer.zero_grad()
            policy.sl_loss.backward()
            policy_optimizer.step()
            tree.n_update += 1
            loss_total += policy.sl_loss
            policy.sl_loss = 0

            time_str = datetime.datetime.now().isoformat()
            conf.cal_train_one_step(flat_probs)
            sys.stdout.write("epoch %d step %d time %s | loss: %f, NA accuracy: %f, not NA accuracy: %f, total accuracy: %f\r" % (epoch, batch_num, time_str, global_loss, conf.acc_NA.get(), conf.acc_not_NA.get(), conf.acc_total.get()))    
            sys.stdout.flush()

        if epoch % conf.save_epoch == 0:
            print('Epoch ' + str(epoch) + ' has finished')
            print("Test model....")
            conf.testModel = policy.base_model
            auc, pr_x, pr_y = conf.test_one_epoch()
            print("auc:", auc)
            if auc > best_auc:
                best_auc = auc
                best_p = pr_x
                best_r = pr_y
                best_epoch = epoch
            print('Saving model...')
            torch.save(policy.state_dict(), "./checkpoint/epoch_" + str(epoch))
            #test_step_sl()
    print("Finish training")
    print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
    print("Storing best result...")
    

def test_step_sl():
    print('test starts')
    if conf.load_model:
        policy.load_state_dict(torch.load("./checkpoint/epoch_" + str(3)))
    policy.eval()
    test_result = []
    for batch_num in range(conf.test_batches):
        conf.get_test_batch(batch_num)
        test_one_step()
        logits, probs = policy.base_model()
        policy.bag_vec = logits
        bag_ids = conf.bag_ids
        cur_batch_size = len(bag_ids)
        cur_class_batch = np.zeros(cur_batch_size, dtype=int)

        for t in range(conf.n_steps_sl):#3
            print("step", t)
            print("cur_class_batch", cur_class_batch, len(cur_class_batch))
            next_classes_batch = tree.p2c_batch(cur_class_batch)#[batch,上一阶段标签的子标签]，可以看成第n层及他之前的标签
            if len(next_classes_batch[0]) == 0:
                print(t, cur_class_batch)
                break  
            print("next_classes_batch", next_classes_batch[0], len(next_classes_batch[0]))
            probs = policy.step_sl(conf, cur_class_batch, next_classes_batch, None, sigmoid = True)
            print("probs", probs[0], len(probs[0]))
            indices, next_class_batch_pred, bag_ids, next_class_batch_logits = tree.get_next_by_probs(conf, cur_class_batch, next_classes_batch, bag_ids, probs, save_prob=False)

            print("indices", indices ,len(indices))
            print("next_class_batch_pred", next_class_batch_pred, len(next_class_batch_pred))
            print("bag_ids", bag_ids)
            cur_class_batch = next_class_batch_pred
            policy.duplicate_bag_vec(indices)


            prob = list(stack_output.data.cpu().numpy())
            if t == 2:
                print(next_class_batch_true,len(prob[0]), len(prob))
                print(prob, len([0]),len(prob))
                print(bag_ids, len(bag_ids))


    return 




conf = config.Config()
conf.load_train_data()
conf.load_test_data()

tree = Tree()
base_model = PCNN_MAXIUM(conf)
policy = Policy(conf, tree.n_class, base_model)
policy.cuda()



policy_optimizer = torch.optim.Adam(policy.parameters(), lr=conf.policy_lr, weight_decay=conf.policy_weight_decay)
#base_model_optimizer = torch.optim.SGD(policy.base_model.parameters(), lr = conf.base_model_lr, weight_decay = conf.base_model_weight_decay)
criterion = torch.nn.BCEWithLogitsLoss() #输入logits 和 label 输出的是loss


#train_step_sl()

test_step_sl()



