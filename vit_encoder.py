# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:55:45 2023

@author: Paras
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Head(nn.Module):
    def __init__(self,num_heads,weight_dimension):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn((weight_dimension*num_heads,weight_dimension))).to(device)
        self.w2 = nn.Parameter(torch.randn((weight_dimension*num_heads,weight_dimension))).to(device)
        self.w3 = nn.Parameter(torch.randn((weight_dimension*num_heads,weight_dimension))).to(device)


    def forward(self,x):

        self.Q = x @ self.w1
        self.K = x @ self.w2
        self.V = x @ self.w3

        lnq = nn.LayerNorm(self.Q.size()[1:]).to(device)
        lnk = nn.LayerNorm(self.K.size()[1:]).to(device)
        lnv = nn.LayerNorm(self.V.size()[1:]).to(device)


        self.Q = lnq(self.Q)
        self.K = lnk(self.K)
        self.V = lnv(self.V)
        self.K = torch.transpose(self.K, -2, -1)

        out = self.Q @ self.K
        out = out/np.sqrt(self.Q.shape[1])
        out = F.softmax(out,dim=-1)
        out = out @ self.V
        return out

class MHA(nn.Module):
    def __init__(self,num_heads,weight_dimension):
        super().__init__()
        self.num_heads = num_heads
        self.weight_dimension = weight_dimension
        self.heads = nn.ModuleList()
        for i in range(self.num_heads):
            head = Head(self.num_heads,self.weight_dimension)
            self.heads.append(head)

    def forward(self,x):

        out_multihead = nn.ModuleList()
        out_multihead = torch.cat([self.heads[i](x) for i in range(self.num_heads)], dim=-1)
        return out_multihead

class VitEncoder(nn.Module):

    def __init__(self,img_size,patch_size,embedding_dim,n_heads,hidden_dims_mlp,n_classes,batch_size):

        super().__init__()
        self.patch_size = patch_size
        self.n_heads = n_heads
        self.hidden_dims_mlp = hidden_dims_mlp
        self.img_size = img_size
        self.n_classes = n_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.num_patches = (img_size//patch_size)**2

        self.weight_dimension = self.embedding_dim//self.n_heads

        self.mha = MHA(self.n_heads,self.weight_dimension).to(device)

        self.mlp_inside_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim*(self.num_patches+1), self.hidden_dims_mlp),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dims_mlp, self.embedding_dim*(self.num_patches+1)),
            nn.GELU(),
            nn.Dropout(0.1)
            )
        
    def forward(self,images):

        ln = nn.LayerNorm(images.size()[1:]).to(self.device)
        out = ln(images)
        layer_norm1 = out.clone()
        out = self.mha(out)
        out = out + layer_norm1
        skip = out.clone()
        out = out.to(self.device)
        ln = nn.LayerNorm(out.size()[1:]).to(self.device)
        out = ln(out)
        out = self.mlp_inside_encoder(out.reshape(out.shape[0],out.shape[1]*out.shape[2]))
        #print(out.shape)
        out = skip + out.reshape(self.batch_size,layer_norm1.shape[1],self.embedding_dim)
        return out

'''
out = out[:,-1,:]
out = self.mlp_classification(out)
'''