import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class AggAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128,name="0"):
        super(AggAttention, self).__init__()
        self.name = name
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        print(self.name)
        print(beta)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)

class GCN(nn.Module):         #GCN
    def __init__(self, hid_dim,bias=True):
        super(GCN, self).__init__()
        self.weight = Parameter(torch.FloatTensor(hid_dim,hid_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(hid_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.non_linear = nn.Tanh()

    def reset_parameters(self, ):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        output = torch.spmm(adj, self.non_linear(torch.spmm(inputs, +self.weight)))
        if self.bias is not None:
            F.tanh(output + self.bias)
        if self.bias is not None:
            F.tanh(output + self.bias)
        else:
            F.elu(output)
        return output

class Classes_Relation_AGG(nn.Module):
    '''
    对同类型之间的节点进行聚合，同类型节点之间存在不同的关系，首先进行关系内聚合，然后进行关系间聚合
    '''
    def __init__(self,hid_dim,feat_drop,relation_num):
        super(Classes_Relation_AGG, self).__init__()
        self.len = relation_num     #相同类型间边关系数量
        #self.GCN = nn.ModuleList([GCN(hid_dim) for _ in range(relation_num)] )    #不同关系用不同GCN 所以有多个GCN层
        self.GCN2 = GCN(hid_dim)     #不同关系用不同GCN 所以有多个GCN层
        self.inter_agg = AggAttention(hid_dim,name="类型关系聚合")      #注意力层，对不同关系间使用注意力机制进行聚合
    def forward(self,feature,same_type_adj):
        '''
        :param feature:目标节点类型特征
        :param same_type_adj:目标节点类型间的关系
        :return:
        '''
        # h = [self.GCN[i](feature,same_type_adj[i]) for i in range(self.len)]
        # return self.inter_agg(torch.stack(h,dim=1))
        adj_sum = sum(same_type_adj)        #两个矩阵变成了一个矩阵    商品于商品之间的所有关系构成一个矩阵
        return self.GCN2(feature,adj_sum)


