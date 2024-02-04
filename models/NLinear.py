import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.SVQ_block import VectorQuantize
class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.vq = VectorQuantize(dim = configs.enc_in, codebook_size = configs.seq_len, decay = 0.8, commitment_weight = 1., orthogonal_reg_weight=0.8, heads = 4,
            separate_codebook_per_head = True, ema_update = False, learnable_codebook = True)
        
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        quantized, indices, commit_loss1 = self.vq(x)
        x = self.Linear(quantized.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x, commit_loss1 # [Batch, Output length, Channel]