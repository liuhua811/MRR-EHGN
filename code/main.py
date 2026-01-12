import argparse
import torch
import numpy as np
import random
import time
import warnings
import matplotlib.pyplot as plt
import psutil
import os
from sklearn.metrics import roc_curve, auc
from tools import plot_auc_curve

from sklearn.manifold import TSNE
from tools import evaluate_results_nc
from pytorchtools import EarlyStopping
from load_data import load_company_data
from model.model import Model

warnings.filterwarnings('ignore')


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# --------------------------------------------------------
# 主函数
# --------------------------------------------------------
def main(args):

    # 读取数据
    ADJ, features, labels, num_classes, train_idx, val_idx, test_idx = load_company_data()
    input_dim = [i.shape[1] for i in features]

    # 类型数量
    same_type_num = len(ADJ[0])
    relation_num = len(ADJ[1])

    # 定义模型
    model = Model(
        input_dim,
        args["hidden_units"],
        num_classes,
        args["feat_drop"],
        same_type_num,
        relation_num
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=args['patience'],
        verbose=True,
        save_path=f'checkpoint/checkpointTest_{args["dataset"]}.pt'
    )

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args['lr'],
        weight_decay=args['weight_decay']
    )

    # --------------------------------------------------------
    # 训练
    # --------------------------------------------------------
    for epoch in range(args['num_epochs']):
        start_time = time.time()

        model.train()
        logits, h_list = model(features, ADJ)
        loss = loss_fcn(logits[train_idx], labels[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证与测试损失
        model.eval()
        logits, h_list = model(features, ADJ)
        val_loss = loss_fcn(logits[val_idx], labels[val_idx])
        test_loss = loss_fcn(logits[test_idx], labels[test_idx])

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Train Loss {loss.item():.4f} | "
            f"Val Loss {val_loss.item():.4f} | "
            f"Test Loss {test_loss.item():.4f}"
        )

        # Early stopping
        early_stopping(val_loss.item(), model)
        if early_stopping.early_stop:
            print("Early stopping!")
            break


    print("\nTesting...")
    model.load_state_dict(
        torch.load(f'checkpoint/checkpointTest_{args["dataset"]}.pt')
    )
    model.eval()

    logits, h_list = model(features, ADJ)

    # 评估指标（Macro-F1 / Micro-F1 / AUC 等）
    evaluate_results_nc(
        h_list.detach().cpu().numpy(),
        labels.cpu().numpy(),
        int(labels.max()) + 1,
        train_idx,
        test_idx
    )

    # 绘制 ROC-AUC 曲线


    auc_value = plot_auc_curve(
        logits,
        labels,
        test_idx,
        save_path='auc_curve.png'  # 可选：保存图片
    )

    print(f"Test AUC: {auc_value:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='基于 GAT 的 HAN 企业破产预测模型')

    parser.add_argument('--dataset', default='CIKM2019', help='数据集')
    parser.add_argument('--prefix', default='data/', help='数据路径')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_heads', type=list, default=[8], help='多头注意力数')
    parser.add_argument('--hidden_units', type=int, default=64, help='隐藏维度')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout')
    parser.add_argument('--feat_drop', type=float, default=0.2, help='特征 dropout')
    parser.add_argument('--sample_rate', type=list, default=[7, 1], help='采样率')
    parser.add_argument('--num_epochs', type=int, default=1000, help='最大迭代次数')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='L2 正则')
    parser.add_argument('--patience', type=int, default=30, help='EarlyStop patience')
    parser.add_argument('--device', type=str, default='cpu', help='cpu 或 cuda:0')
    parser.add_argument('--repeat', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args().__dict__

    set_random_seed(args['seed'])
    print(args)

    main(args)
