# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:15:31 2023

@author: Paras
"""
from torch import nn

class VitBlocks(nn.Module):

    def __init__(self,encoder_layer,num_layers):
        super().__init__()
        self.encoders = nn.Sequential()
        for i in range(num_layers):
            self.encoders.append(encoder_layer)

    def forward(self,x):
        out = self.encoders(x)
        return out


#* unpacking operator
