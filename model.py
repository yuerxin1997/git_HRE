import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, conf, n_class, base_model):#in_dim应该是hidden_size *3+class_embedding_size
        super(Policy, self).__init__()
        #self.args = args
        self.conf = conf
        self.L = 0.02
        self.baseline_reward = 0
        self.entropy_l = []
        self.beta = conf.beta #2
        self.beta_decay_rate = .9
        self.n_update = 0
        self.class_embed = nn.Embedding(n_class, self.conf.class_embed_size) #n_class 105 这个embedding不知道有没有参与训练？
        self.class_embed_bias = nn.Embedding(n_class, 1)

        #这里是对标签嵌入做一些操作 没看懂。
        stdv = 1. / np.sqrt(self.class_embed.weight.size(1))
        self.class_embed.weight.data.uniform_(-stdv, stdv)
        self.class_embed_bias.weight.data.uniform_(-stdv, stdv)
        




        self.saved_log_probs = []
        self.rewards = []
        self.rewards_greedy = []
        self.bag_vec = None
        self.base_model = base_model

        self.criterion = torch.nn.BCEWithLogitsLoss()#二分类交叉熵函数，输入logits和label，返回loss
        self.sl_loss = 0


        in_dim = self.conf.class_embed_size + self.conf.hidden_size * 3 # i 我们的indim = indim_ori(690) + class_embed_size(50) = 740

        self.l1 = nn.Linear(in_dim, self.conf.l1_size)# ours: [740,300]
        self.l2 = nn.Linear(self.conf.l1_size, self.conf.class_embed_size) #[300,50]
        # elif self.args.use_l1:
        #     self.l1 = nn.Linear(in_dim, args.class_embed_size)

    def update_baseline(self, target):#更新baseline_reward,好像不咋用了
        # a moving average baseline, not used anymore
        self.baseline_reward = self.L * target + (1 - self.L) * self.baseline_reward
    def finish_episode(self):#完成一个episode后，把参数更新了
        self.sl_loss = 0
        self.n_update += 1
        if self.n_update % self.conf.update_beta_every == 0: #update_beta_every=500
            self.beta *= self.beta_decay_rate #应该是越往后训练，更新的越少了

        self.entropy_l = []
        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.rewards_greedy[:]

    def forward(self, cur_class_batch, next_classes_batch):#详细介绍
        #输入这个批次的现在的标签，以及能产生的所有标签，state_embed在一个batch内是一样的
        #输出就是给出能产生所有标签的概率
        cur_class_embed = self.class_embed(cur_class_batch)  # (batch, 50)
        #print(cur_class_batch)
        next_classes_embed = self.class_embed(next_classes_batch)  # (batch, max_choices, 50)
        #print("-"*20,"class_embed_bias:",self.class_embed_bias)
        nb = self.class_embed_bias(next_classes_batch).squeeze(-1)#原来是[batch,1]后来压缩成[[batch]
        #print("-"*20,"nb:",nb.size())

        states_embed = self.bag_vec
        #print("states_embed", states_embed, states_embed.size())
        #print("*"*20,states_embed.size,states_embed.size())
        states_embed = torch.cat((states_embed, cur_class_embed), 1)
        if self.conf.use_l2: #true
            h1 = F.relu(self.l1(states_embed))
            h2 = F.relu(self.l2(h1))
        # else:
        #     h2 = F.relu(self.l1(states_embed))
        h2 = h2.unsqueeze(-1)  # (batch, 50, 1)
        probs = torch.bmm(next_classes_embed, h2).squeeze(-1) + nb #[batch,max_choices]
        #torch.bmmm是batch matrix multiply: torch a.size()=[batch,a,b] b.size()=[batch,c,d] torch.bmm(a,b)=[batch,a,ds]
        return probs

    def duplicate_bag_vec(self, indices):#输入bathch中的id，返回bag_vec
        assert self.bag_vec is not None
        assert len(indices) > 0
        self.bag_vec = self.bag_vec[indices]

    def duplicate_reward(self, indices):#好像没用到
        assert len(indices) > 0
        self.saved_log_probs[-1] = [[probs[i] for i in indices] for probs in self.saved_log_probs[-1]]
        self.rewards[-1] = [[R[i] for i in indices] for R in self.rewards[-1]]




    def generate_logits(self, conf, cur_class_batch, next_classes_batch):#输入[batch,tokens],[batch,cur_class_batch],[batch,max_choices],返回[batch,max_choices_logits]
        # print("cur_class_batch")
        # print(cur_class_batch,len(cur_class_batch))
        cur_class_batch = Variable(torch.from_numpy(cur_class_batch)).cuda()
        # print("next_classes_batch")
        # print(next_classes_batch, len(next_classes_batch))
        next_classes_batch = Variable(torch.from_numpy(next_classes_batch)).cuda() #[batch,max_choices]
        logits = self(cur_class_batch, next_classes_batch)
        # mask padding relations
        logits = (next_classes_batch == 0).float() * -99999 + (next_classes_batch != 0).float() * logits
        return logits

    def step_sl(self, conf, cur_class_batch, next_classes_batch, next_classes_batch_true, sigmoid=True):
        logits = self.generate_logits(conf, cur_class_batch, next_classes_batch)
        if not sigmoid:
            return logits
        if next_classes_batch_true is not None:
            y_true = Variable(torch.from_numpy(next_classes_batch_true)).cuda().float()
            self.sl_loss += self.criterion(logits, y_true)
        return torch.sigmoid(logits)

    # def step(self, mini_batch, cur_class_batch, next_classes_batch, test=False, flat_probs=None):
    #     logits = self.generate_logits(mini_batch, cur_class_batch, next_classes_batch)
    #     if self.conf.softmax:
    #         probs = F.softmax(logits, dim=-1)
    #     else:
    #         probs = F.sigmoid(logits)
    #     if not test:
    #         # + epsilon to avoid log(0)
    #         #增加的
    #         temp = torch.mean(torch.log(probs + 1e-32) * probs)
    #         self.entropy_l.append(temp.view(1))
    #     next_classes_batch = Variable(torch.from_numpy(next_classes_batch)).cuda()
    #     probs = probs + (next_classes_batch != 0).float() * 1e-16
    #     m = Categorical(probs)
    #     if test or self.args.sample_mode == 'choose_max':
    #         action = torch.max(probs, 1)[1]
    #     # elif self.args.sample_mode == 'random':
    #     #     if random.random() < 1.2:
    #     #         if self.args.gpu:
    #     #             action = Variable(torch.zeros(probs.size()[0]).long().random_(0, probs.size()[1])).cuda()
    #     #         else:
    #     #             action = Variable(torch.zeros(probs.size()[0]).long().random_(0, probs.size()[1]))
    #     #     else:
    #     #         action = m.sample()
    #     else:
    #         action = m.sample() #随机采样actions
    #     return action, m





