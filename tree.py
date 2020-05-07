import logging
import pickle
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

import json


class Tree:
    def __init__(self, is_train = True):
        #necessary

        with open("./data/p2c_id.json", "r") as file:
            self.p2c_idx_temp = json.load(file)
        if is_train:
            with open("./data/train_hierarchical_bag_label.json", "r") as file:
                self.hierarchical_bag_label_temp = json.load(file) 
        self.p2c_idx = {}
        for i in self.p2c_idx_temp:
            self.p2c_idx[int(i)] = self.p2c_idx_temp[i]

        self.hierarchical_bag_label = {}    
        for i in self.hierarchical_bag_label_temp:
            self.hierarchical_bag_label[int(i)] = self.hierarchical_bag_label_temp[i]
        # print(self.p2c_idx)
        # print(self.hierarchical_bag_label)
        self.next_true_bin, self.next_true = self.generate_next_true()
        self.p2c_idx_np = self.pad_p2c_idx()
        self.n_class = len(self.hierarchical_bag_label)
        self.n_update = 0
        self.cur_epoch = 0
    def pad_p2c_idx(self): #构造p2c_idx_np
        col = max([len(c) for c in self.p2c_idx.values()]) #col是含有子节点最多的个数。 可以直接+1，因为这里的allow_up没用了
        res = np.zeros((len(self.p2c_idx), col), dtype=int) #[104,25]
        for row_i in range(len(self.p2c_idx)):
            res[row_i, :len(self.p2c_idx[row_i])] = self.p2c_idx[row_i]
        return res


    def p2c_batch(self, ids):#输入标签id，返回标签id的子标签+标签id
        # ids is virtual
        res = self.p2c_idx_np[ids]
        # remove columns full of zeros
        b = res[:, ~np.all(res == 0, axis=0)]
        return res[:, ~np.all(res == 0, axis=0)]

    def generate_next_true(self):#构造
        next_true_bin = defaultdict(lambda: defaultdict(list))
        next_true = defaultdict(lambda: defaultdict(list))
        for did in range(len(self.hierarchical_bag_label)):
            class_idx_set = set(self.hierarchical_bag_label[did]) #是当前文档标签的集合
            class_idx_set.add(0)
            for c in class_idx_set:
                for idx, next_c in enumerate(self.p2c_idx[c]):
                    if next_c in class_idx_set:
                        next_true_bin[did][c].append(1)#key是当前文档id，value是一个dict, {"原来就有的标签"：[0，1，0]} 原来的标签的子标签是否在已经在当前标签中
                        next_true[did][c].append(next_c)#key是当前文档id，value是一个dict, {"原来就有的标签"：[34，55]} 如果已经在当前标签中，添加标签的子标签
                    else:
                        next_true_bin[did][c].append(0)
                # if lowest label 我觉得这个可以不用 (在我们的项目中，标签比较均衡，留着好像也没事) 
                if len(next_true[did][c]) == 0:
                    next_true[did][c].append(c)#如果没有子标签在当前文档标签中，就添加当前标签。
        return next_true_bin, next_true#构造
        
    def get_next(self, cur_class_batch, next_classes_batch, bag_ids):#详细介绍
        assert len(cur_class_batch) == len(bag_ids)
        next_classes_batch_true_bin = np.zeros(next_classes_batch.shape)
        indices = []
        next_class_batch_true = []
        for ct, (c, did) in enumerate(zip(cur_class_batch, bag_ids)):
            nt = self.next_true_bin[did][c] 
            if len(self.next_true[did][c]) == 0:
                exit(-1)   
            next_classes_batch_true_bin[ct][:len(nt)] = nt
            for idx in self.next_true[did][c]:
                indices.append(ct) 
                next_class_batch_true.append(idx)
        bag_ids = [bag_ids[idx] for idx in indices]
        return next_classes_batch_true_bin, indices, np.array(next_class_batch_true), bag_ids


    def get_next_by_probs(self, conf, cur_class_batch, next_classes_batch, bag_ids, probs, save_prob):
        assert len(cur_class_batch) == len(bag_ids) == len(bag_ids) == len(probs)
        #print("probs:",len(probs),probs)
        indices = []
        next_class_batch_pred = []
        if save_prob:
            thres = 0
        else:
            thres = 0.5
        preds = (probs > thres).int().data.cpu().numpy()
        for ct, (c, next_classes, did, pred, p) in enumerate(
                zip(cur_class_batch, next_classes_batch, bag_ids, preds, probs)):
            # need allow_stay=True, filter last one (cur) to avoid duplication
            next_pred = np.nonzero(pred)[0]
            if not conf.multi_label:
                idx_above_thres = np.argsort(p.data.cpu().numpy()[next_pred])
                for idx in idx_above_thres[::-1]:
                    if next_classes[next_pred[idx]] != c:
                        next_pred = [next_pred[idx]]
                        break
            else:
                if len(next_pred) != 0 and next_classes[next_pred[-1]] == c:
                    next_pred = next_pred[:-1]
            # if no next > threshold, stay at current class
            if len(next_pred) == 0:
                p_selected = []
                next_pred = [c]
            else:
                p_selected = p.data.cpu().numpy()[next_pred]
                next_pred = next_classes[next_pred]
            # indices remember where one is from; idx is virtual class idx
            for idx in next_pred:
                indices.append(ct)
                next_class_batch_pred.append(idx)
            if save_prob:
                for idx, p_ in zip(next_pred, p_selected):
                    if idx in self.id2prob[did]:
                        self.logger.warning(f'[{did}][{idx}] already existed!')
                    self.id2prob[did][idx] = p_
        bag_ids = [bag_ids[idx] for idx in indices]
        return indices, np.array(next_class_batch_pred), bag_ids

# tree = Tree()
# cur_class_batch = [1,1,1,1,1,1,2]
# next_classes_batch = tree.p2c_batch(cur_class_batch)
# print(next_classes_batch)


