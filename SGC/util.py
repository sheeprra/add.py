import numpy as np
import pandas as pd
import torch


def load_data():
    # 导入数据：分隔符为空格
    # cora.content共有2708行，每一行代表一个样本点，即一篇论文。
    # 每一行由三部分组成，分别是论文的编号，如31336；论文的词向量，一个有1433位的二进制；论文的类别，如Neural_Networks
    raw_data = pd.read_csv('../data/cora/cora.content', sep='\t', header=None)
    num = raw_data.shape[0]  # 样本点数2708
    # 将论文的编号转[0,2707]
    # 索引列表
    index_list = list(raw_data.index)
    # 论文编号列表
    id_list = list(raw_data[0])
    id_index = zip(id_list, index_list)
    id_index_map = dict(id_index)
    # 将词向量提取为特征,第二行到倒数第二行  (2708, 1433)
    features = raw_data.iloc[:, 1:-1]
    labels = pd.get_dummies(raw_data[1434])
    # 论文引用数据
    # cora.cites共5429行， 每一行有两个论文编号，表示第一个编号的论文先写，第二个编号的论文引用第一个编号的论文
    raw_data_cites = pd.read_csv('../data/cora/cora.cites', sep='\t', header=None)
    # 创建一个规模和邻接矩阵一样大小的矩阵(2708, 2708)
    adj_matrix = np.zeros((num, num))
    # 创建邻接矩阵
    for thesis, cite_thesis in zip(raw_data_cites[0], raw_data_cites[1]):
        thesis_index = id_index_map[thesis]
        cite_thesis_index = id_index_map[cite_thesis]  # 替换论文编号为[0,2707]
        adj_matrix[thesis_index][cite_thesis_index] = adj_matrix[cite_thesis_index][thesis_index] = 1  # 有引用关系的样本点之间取1

    # 归一化邻接矩阵
    adj_matrix = normalize(adj_matrix)
    adj_matrix = torch.FloatTensor(adj_matrix)
    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(np.where(labels)[1])
    return adj_matrix, features, labels


def normalize(matrix):
    """对称归一化"""
    """A' = (D + I)^-0.5 * ( A + I ) * (D + I)^-0.5"""
    matrix = matrix + np.eye(matrix.shape[0])  # A+I
    row_sum = np.array(matrix.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()  # (D + I)^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)  # 构建成对角矩阵
    return d_mat_inv_sqrt.dot(matrix).dot(d_mat_inv_sqrt)


# 计算精确度
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
