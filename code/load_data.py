import numpy as np
import scipy
import pickle
import torch
import dgl
import torch.nn.functional as F
import torch as th
import scipy.sparse as sp











def load_company_data(prefix=r'D:\桌面文件\中小企业破产预测'):

    # features_0 = scipy.sparse.load_npz(prefix + '/combined_company_features.npz').toarray()  #按照时间加权的特征  有初始特征
    # features_0 = scipy.sparse.load_npz(prefix + '/combined_company_features_trimmed.npz').toarray()  #按照时间加权的特征 没有初始特征
    # features_0 = scipy.sparse.load_npz(prefix + '/1company_features_sparse.npz').toarray()  #没有初始特征 有公司编号
    # features_0 = scipy.sparse.load_npz(prefix + '/2company_features_sparse.npz').toarray()  ##没有初始特征 无公司编号
    features_0 = scipy.sparse.load_npz(prefix + '/3whole_company_features_sparse.npz').toarray()  ##初始特征 有公司编号
    # features_0 = scipy.sparse.load_npz(prefix + '/4whole_company_features_sparse.npz').toarray()  ##初始特征  无公司编号


    features_1 = scipy.sparse.load_npz(prefix + '/person_features.npz').toarray()
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features = [features_0, features_1]

    labels = np.load(prefix + '/node_labels_sorted.npy')
    labels = labels[1]
    labels = torch.LongTensor(labels)
    print(len(labels))


    train_val_test_idx = np.load(prefix + '/train_val_test_idx4.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    num_classes = 2
    print(train_idx.shape)
    print(val_idx.shape)
    print(test_idx.shape)

    g1 = scipy.sparse.load_npz(prefix + '/adj_type_0_cp_metapath.npz').toarray()

    g2 = scipy.sparse.load_npz(prefix + '/adj_type_2_cp_metapath.npz').toarray()

    g3 = scipy.sparse.load_npz(prefix + '/adj_type_4_cp_metapath.npz').toarray()

    g4 = scipy.sparse.load_npz(prefix + '/adj_type_6_cc_metapath.npz').toarray()

    g5 = scipy.sparse.load_npz(prefix + '/adj_type_7_cc_metapath.npz').toarray()

    g6 = scipy.sparse.load_npz(prefix + '/adj_type_9_cp_metapath.npz').toarray()

    g7 = scipy.sparse.load_npz(prefix + '/adj_type_10_cc_metapath.npz').toarray()

    g8 = scipy.sparse.load_npz(prefix + '/adj_type_11_cc_metapath.npz').toarray()


    g9 = scipy.sparse.load_npz(prefix + '/adj_type_6_cc.npz').toarray()

    g10 = scipy.sparse.load_npz(prefix + '/adj_type_7_cc.npz').toarray()

    g11 = scipy.sparse.load_npz(prefix + '/adj_type_10_cc.npz').toarray()

    g12 = scipy.sparse.load_npz(prefix + '/adj_type_11_cc.npz').toarray()



    cp1 = scipy.sparse.load_npz(prefix + '/adj_type_0_cp.npz').toarray()

    cp2 = scipy.sparse.load_npz(prefix + '/adj_type_2_cp.npz').toarray()

    cp3 = scipy.sparse.load_npz(prefix + '/adj_type_4_cp.npz').toarray()

    cp4 = scipy.sparse.load_npz(prefix + '/adj_type_9_cp.npz').toarray()
    print(cp1.shape)

    meta_path = [g1,g2,g3,g4,g5,g6,g7,g8]
    # meta_path = [g1, g2, g3, g4, g5, g6, g7, g8]
    meta_path = [F.normalize(torch.FloatTensor(i),dim=1,p=2) for i in meta_path]


    c_c = [g9,g10,g11,g12]
    # c_c = [g13,g14,g15]
    c_p = [cp1,cp2,cp3,cp4]
    c_c = [F.normalize(torch.FloatTensor(i),dim=1,p=2) for i in c_c]
    c_p = [F.normalize(torch.FloatTensor(i),dim=1,p=2) for i in c_p]
    # H_hyper = F.normalize(torch.FloatTensor(g13), dim=1, p=2)


    ADJ = [c_c,meta_path,c_p]




    return ADJ,features, labels, num_classes, train_idx, val_idx, test_idx

    # 载入 metapath
    # g1 = scipy.sparse.load_npz(prefix + '/adj_type_0_cp_metapath.npz').toarray()
    # g2 = scipy.sparse.load_npz(prefix + '/adj_type_2_cp_metapath.npz').toarray()
    # g3 = scipy.sparse.load_npz(prefix + '/adj_type_4_cp_metapath.npz').toarray()
    # g4 = scipy.sparse.load_npz(prefix + '/adj_type_6_cc_metapath.npz').toarray()
    # g5 = scipy.sparse.load_npz(prefix + '/adj_type_7_cc_metapath.npz').toarray()
    # g6 = scipy.sparse.load_npz(prefix + '/adj_type_9_cp_metapath.npz').toarray()
    # g7 = scipy.sparse.load_npz(prefix + '/adj_type_10_cc_metapath.npz').toarray()
    # g8 = scipy.sparse.load_npz(prefix + '/adj_type_11_cc_metapath.npz').toarray()
    #
    # # 转换为无权图：只保留 0/1
    # meta_path = [(i > 0).astype(np.float32) for i in [g1, g2, g3, g4, g5, g6, g7, g8]]
    # meta_path = [F.normalize(torch.FloatTensor(i), dim=1, p=2) for i in meta_path]
    #
    # # c-c 边
    # g9 = scipy.sparse.load_npz(prefix + '/adj_type_6_cc.npz').toarray()
    # g10 = scipy.sparse.load_npz(prefix + '/adj_type_7_cc.npz').toarray()
    # g11 = scipy.sparse.load_npz(prefix + '/adj_type_10_cc.npz').toarray()
    # g12 = scipy.sparse.load_npz(prefix + '/adj_type_11_cc.npz').toarray()
    # c_c = [(i > 0).astype(np.float32) for i in [g9, g10, g11, g12]]
    # c_c = [F.normalize(torch.FloatTensor(i), dim=1, p=2) for i in c_c]
    #
    # # c-p 边
    # cp1 = scipy.sparse.load_npz(prefix + '/adj_type_0_cp.npz').toarray()
    # cp2 = scipy.sparse.load_npz(prefix + '/adj_type_2_cp.npz').toarray()
    # cp3 = scipy.sparse.load_npz(prefix + '/adj_type_4_cp.npz').toarray()
    # cp4 = scipy.sparse.load_npz(prefix + '/adj_type_9_cp.npz').toarray()
    # c_p = [(i > 0).astype(np.float32) for i in [cp1, cp2, cp3, cp4]]
    # c_p = [F.normalize(torch.FloatTensor(i), dim=1, p=2) for i in c_p]
    #
    # # 最终邻接矩阵集合
    # ADJ = [c_c, meta_path, c_p]

##以下是GCN
def load_company_data1(prefix=r'D:\桌面文件\中小企业破产预测'):

        # 加载商品和用户特征
        # features_0 = scipy.sparse.load_npz(prefix + '/2company_features_sparse.npz').toarray()  ##没有初始特征 无公司编号
        features_0 = scipy.sparse.load_npz(prefix + '/3whole_company_features_sparse.npz').toarray()  ##初始特征 有公司编号
        # features_0 = scipy.sparse.load_npz(prefix + '/4whole_company_features_sparse.npz').toarray()  ##初始特征  无公司编号

        features_1 = scipy.sparse.load_npz(prefix + '/person_features.npz').toarray()
        features_0 = torch.FloatTensor(features_0)
        features_1 = torch.FloatTensor(features_1)
        features = [features_0]

        labels = np.load(prefix + '/node_labels_sorted.npy')
        labels = labels[1]
        labels = torch.LongTensor(labels)

        # 加载训练、验证、测试索引
        train_val_test_idx = np.load(prefix + '/train_val_test_idx4.npz')
        train_idx = train_val_test_idx['train_idx']
        val_idx = train_val_test_idx['val_idx']
        test_idx = train_val_test_idx['test_idx']

        # 加载关系矩阵，并相加为一个整体的矩阵
        # 加载 item_item 数据并转换为稀疏矩阵
        g9 = scipy.sparse.load_npz(prefix + '/adj_type_6_cc.npz').toarray()

        g10 = scipy.sparse.load_npz(prefix + '/adj_type_7_cc.npz').toarray()

        g11 = scipy.sparse.load_npz(prefix + '/adj_type_10_cc.npz').toarray()

        g12 = scipy.sparse.load_npz(prefix + '/adj_type_11_cc.npz').toarray()




        item_item = g9 + g10 + g11 + g12



        # 转换为稀疏矩阵
        item_item_sparse = sp.csr_matrix(item_item)

        # 使用 dgl.from_scipy 转换为 DGL 图对象
        item_item = dgl.from_scipy(item_item_sparse)
        item_item = dgl.add_self_loop(item_item)

        # 将 item_item_dgl 放入 ADJ 列表中，以便在模型中使用
        ADJ = item_item

        num_classes = 2  # 设定类别数

        return ADJ, features, labels, num_classes, train_idx, val_idx, test_idx

if __name__ == "__main__":
    load_company_data()
