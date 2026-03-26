import torch
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics
from munkres import Munkres
import matplotlib.pyplot as plt
from kmeans import kmeans
import sklearn.preprocessing as preprocess
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.manifold import TSNE
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


# def umap_vis(X, y, num_classes, title='', seed=0, n_neighbors=15, min_dist=0.2):
#     reducer = umap.UMAP(
#         n_components=2, 
#         random_state=seed, 
#         n_neighbors=n_neighbors, 
#         min_dist=min_dist
#     )
#     X_umap = reducer.fit_transform(X)
#     plt.figure(figsize=(8, 6))
#     if np.min(y) == 1:
#         for i in range(1, num_classes + 1):
#             plt.scatter(X_umap[y == i, 0], X_umap[y == i, 1], label=str(i))
#     else:
#         for i in range(num_classes):
#             plt.scatter(X_umap[y == i, 0], X_umap[y == i, 1], label=str(i))

#     plt.xticks([])
#     plt.yticks([])
#     plt.title(title)
#     plt.legend()
#     plt.savefig(title + '.png')
#     plt.close()





def load_data(dataset_name, show_details=False):
    load_path = "./data/" + dataset_name + "/" + dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)
    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("undirected edge num:   ", int(np.nonzero(adj)[0].shape[0]/2))
        print("category num:          ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")

    return feat, label, adj #返回特征，标签，邻接矩阵



def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index #将整数添加到索引列表中


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape  #用于将稀疏矩阵转换为三元组形式 (coords, values, shape)


def decompose(adj, dataset, norm='sym', renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    evalue, evector = np.linalg.eig(laplacian.toarray())  #特征值和特征向量计算
    np.save(dataset + ".npy", evalue)
    # print(max(evalue))
    exit(1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    n, bins, patches = ax.hist(evalue, 50, facecolor='g')
    plt.xlabel('Eigenvalues')
    plt.ylabel('Frequncy')
    fig.savefig("eig_renorm_" + dataset + ".png")


def laplacian(adj):
    rowsum = np.array(adj.sum(1))
    degree_mat = sp.diags(rowsum.flatten())
    lap = degree_mat - adj
    return torch.FloatTensor(lap.toarray())


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score



# from scipy.optimize import linear_sum_assignment


# def cluster_acc(y_true, y_pred):
#     """
#     calculate clustering acc and f1-score
#     Args:
#         y_true: the ground truth (torch tensor or numpy array)
#         y_pred: the clustering id (torch tensor or numpy array)

#     Returns: acc and f1-score
#     """
#     # 确保 y_true 和 y_pred 是 NumPy 数组
#     if torch.is_tensor(y_true):
#         y_true = y_true.numpy()
#     if torch.is_tensor(y_pred):
#         y_pred = y_pred.numpy()
    
#     y_true = y_true.astype(int)
#     y_pred = y_pred.astype(int)

#     y_true = y_true - np.min(y_true)
#     l1 = list(set(y_true))
#     num_class1 = len(l1)
#     l2 = list(set(y_pred))
#     num_class2 = len(l2)

#     cost = np.zeros((num_class1, num_class2), dtype=int)
#     for i, c1 in enumerate(l1):
#         mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
#         for j, c2 in enumerate(l2):
#             mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
#             cost[i][j] = len(mps_d)

#     # 使用匈牙利算法（Kuhn-Munkres algorithm）
#     row_ind, col_ind = linear_sum_assignment(-cost)
#     new_predict = np.zeros(len(y_pred), dtype=int)
#     for i, c in enumerate(l1):
#         c2 = l2[col_ind[i]]
#         new_predict[y_pred == c2] = c

#     acc = metrics.accuracy_score(y_true, new_predict)
#     f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
#     return acc, f1_macro

from scipy.optimize import linear_sum_assignment


def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth (torch tensor or numpy array)
        y_pred: the clustering id (torch tensor or numpy array)

    Returns: acc and f1-score
    """
    # 确保 y_true 和 y_pred 是 NumPy 数组
    if torch.is_tensor(y_true):
        y_true = y_true.numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.numpy()
    
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    # 如果 y_true 和 y_pred 的长度不一致，可能需要调整
    min_length = min(len(y_true), len(y_pred))
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]

    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)

    # 如果标签数量不一致，创建一个平衡的矩阵
    cost = np.zeros((max(num_class1, num_class2), max(num_class1, num_class2)), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # 使用匈牙利算法（Kuhn-Munkres algorithm）
    row_ind, col_ind = linear_sum_assignment(-cost)
    new_predict = np.zeros(len(y_pred), dtype=int)
    for i, c in enumerate(l1):
        if i < len(col_ind) and col_ind[i] < num_class2:
            c2 = l2[col_ind[i]]
            new_predict[y_pred == c2] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro
# def cluster_acc(y_true, y_pred):
#     """
#     calculate clustering acc and f1-score
#     Args:
#         y_true: the ground truth
#         y_pred: the clustering id

#     Returns: acc and f1-score
#     """
#     y_true = y_true - np.min(y_true)
#     l1 = list(set(y_true))
#     num_class1 = len(l1)
#     l2 = list(set(y_pred))
#     num_class2 = len(l2)
#     ind = 0
#     if num_class1 != num_class2:
#         for i in l1:
#             if i in l2:
#                 pass
#             else:
#                 y_pred[ind] = i
#                 ind += 1
#     l2 = list(set(y_pred))
#     numclass2 = len(l2)
#     if num_class1 != numclass2:
#         print('error')
#         return
#     cost = np.zeros((num_class1, numclass2), dtype=int)
#     for i, c1 in enumerate(l1):
#         mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
#         for j, c2 in enumerate(l2):
#             mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
#             cost[i][j] = len(mps_d)
#     m = Munkres()
#     cost = cost.__neg__().tolist()
#     indexes = m.compute(cost)
#     new_predict = np.zeros(len(y_pred))
#     for i, c in enumerate(l1):
#         c2 = l2[indexes[i][1]]
#         ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
#         new_predict[ai] = c
#     acc = metrics.accuracy_score(y_true, new_predict)
#     f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
#     return acc, f1_macro


def eva(y_true, y_pred, show_details=True):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    if show_details:
        print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
              ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1


def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), adj_tmp)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)

    return norm_adj


def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clustering(feature, true_labels, cluster_num):
    predict_labels,  initial= kmeans(X=feature, num_clusters=cluster_num, distance="euclidean", device="cuda")
    acc, nmi, ari, f1 = eva(true_labels, predict_labels.numpy(), show_details=False)
    return 100 * acc, 100 * nmi, 100 * ari, 100 * f1, predict_labels.numpy(), initial



def loss_cal(x, x_aug):
    T = 1.0
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss


def normalize(data):
    m = data.mean()
    mx = data.max()
    mn = data.min()
    if mn < 0:
        data += torch.abs(mn)
        mn = data.min()
    dst = mx - mn
    return (data - mn).true_divide(dst)



def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)
