import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from networks.embedding import *
from networks.encoder import *
from networks.selector import *
from networks.classifier import *

class PCNN_MAXIUM(nn.Module):
	def __init__(self, config):	
		super(PCNN_MAXIUM, self).__init__()
		self.config = config
		self.embedding = Embedding(config)
		self.encoder = PCNN(config)
		self.selector = Maxium(config, config.hidden_size * 3)
		self.classifier = Classifier(config)
	def forward(self):
		embedding = self.embedding()
		sen_embedding = self.encoder(embedding)
		logits, prob = self.selector(sen_embedding)

		# if is_prob:
		# 	return prob
		return logits, prob

	def test(self):
		embedding = self.embedding()
		sen_embedding = self.encoder(embedding)
		return self.selector.test(sen_embedding)
