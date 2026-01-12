import torch
import torch.nn as nn
import torch.nn.functional as F
from .Classes_Relation_AGG import AggAttention,GCN
import numpy as np

class Dif_Classes_AGG(nn.Module):
    '''
    1.先将自身特征投影到特定空间
    2.再将特定关系的特征投影到特定空间
    3.对特定关系进行注意力聚合，得到多个特定关系的矩阵特征
    4.对不同关系矩阵特征使用注意力进行聚合
    '''
    def __init__(self,hid_dim,relation_num,drop):
        '''
        :param relation_num: 目标类型-其他类型的关系数量
        '''
        super(Dif_Classes_AGG, self).__init__()
        self.non_linear = nn.Tanh()
        self.relation_num = relation_num
        self.project_target = nn.Linear(hid_dim,hid_dim)    #先将自身类型投影到特定空间
        self.project_difRelation = nn.ModuleList([nn.Linear(hid_dim,hid_dim) for _ in range(relation_num)])     #对不同类型的边邻居节点投影到不同的空间
        self.intra_att = nn.ModuleList([GCN(hid_dim) for _ in range(relation_num)])          #将邻居节点按注意力机制聚合，获得不同关系的关系特征矩阵
        self.inter = AggAttention(hid_dim,name="关系聚合")             #将不同关系特征矩阵按注意力机制进行聚合
        self.dropout = nn.Dropout(p=0.5)

    def forward(self,features,adj):
        '''
        :param features: 所有类型节点特征，features[0]为目标节点类型，features[1]为其他类型节点特征
        :param adj:列表内为目标类型-其他类型 不同关系的邻接矩阵
        '''
        target_feat = self.project_target(features[0])    #得到目标节点的特征映射
        relation_feat = [self.non_linear(self.project_difRelation[i](features[0])) for i in range(self.relation_num)]     #得到各个关系的特征映射
        # 将目标节点特征、特定关系特征、特定关系邻接矩阵 传入不同的注意力机制中，得到关系特征矩阵列表
        relation_matrix = [self.intra_att[i](relation_feat[i],adj[i])  for i in range(self.relation_num)]
        relation_agg = self.dropout(self.inter(torch.stack(relation_matrix,dim=1)))
        return relation_agg +target_feat




