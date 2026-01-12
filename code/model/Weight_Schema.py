import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class Weight_Schema(nn.Module):
    def __init__(self,hid_dim,bias=True):
        super(Weight_Schema, self).__init__()
        self.hid_dim = hid_dim
        self.weight = Parameter(torch.FloatTensor(hid_dim, hid_dim))
        self.non_linear = nn.Tanh()
        if bias:
            self.bias = Parameter(torch.FloatTensor(hid_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self, ):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self,h,Adj):
        adj_sum = sum(Adj)
        output = torch.spmm(adj_sum, torch.spmm(h, self.weight))
        if self.bias is not None:
            F.tanh(output + self.bias)
        if self.bias is not None:
            F.tanh(output + self.bias)
        else:
            F.elu(output)
        return output


