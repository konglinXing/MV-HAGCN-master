# import numpy as np
# import torch
# from scipy.sparse import coo_matrix
# from scipy.sparse import csc_matrix
#
#
# def coototensor(A):
#     """
#     Convert a coo_matrix to a torch sparse tensor
#     """
#
#     values = A.data
#     indices = np.vstack((A.row, A.col))
#     i = torch.LongTensor(indices)
#     v = torch.FloatTensor(values)
#     shape = A.shape
#
#     return torch.sparse.FloatTensor(i, v, torch.Size(shape))
#
#
# # def adj_matrix_weight_merge(A, adj_weight):
# #     """扩展为支持5个矩阵的聚合"""
# #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #     N = A[0].shape[0]
# #     temp = coo_matrix((N, N))
# #     temp = coototensor(temp)
# #
# #     # 扩展为5个矩阵
# #     a = coototensor(csc_matrix(A[0]).tocoo())
# #     b = coototensor(csc_matrix(A[1]).tocoo())
# #     c = coototensor(csc_matrix(A[2]).tocoo())
# #     d = coototensor(csc_matrix(A[3]).tocoo())  # 新增：序列相似性
# #     e = coototensor(csc_matrix(A[4]).tocoo())  # 新增：家族相似性
# #
# #     A_t = torch.stack([a, b, c, d, e], dim=2).to_dense()  # 扩展为5个
# #
# #     A_t = A_t.to(device)
# #     temp = torch.matmul(A_t, adj_weight)
# #     temp = torch.squeeze(temp, 2)
# #
# #     return temp
# def adj_matrix_weight_merge(A, adj_weight):
#     """
#     Multiplex Relation Aggregation - 动态处理任意数量的矩阵
#     """
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     N = A[0].shape[0]
#     temp = coo_matrix((N, N))
#     temp = coototensor(temp)
#
#     # 动态处理矩阵数量
#     num_matrices = len(A)
#     matrices = []
#
#     for i in range(num_matrices):
#         matrices.append(coototensor(csc_matrix(A[i]).tocoo()))
#
#     # 使用动态数量的矩阵
#     A_t = torch.stack(matrices, dim=2).to_dense()
#
#     A_t = A_t.to(device)
#     temp = torch.matmul(A_t, adj_weight)
#     temp = torch.squeeze(temp, 2)
#
#     return temp
#

#预测疾病用的：
import numpy as np
import torch
from scipy.sparse import coo_matrix, csc_matrix


def adj_matrix_weight_merge(A, adj_weight):
    """
    简化版本的矩阵聚合，避免稀疏张量转换问题
    直接在numpy中处理，然后转换为tensor
    """
    device = adj_weight.device

    # 获取矩阵数量
    num_matrices = len(A)

    # 初始化结果矩阵
    if isinstance(A[0], np.ndarray):
        result_shape = A[0].shape
        result = np.zeros(result_shape, dtype=np.float32)
    else:
        result_shape = A[0].shape
        result = np.zeros(result_shape, dtype=np.float32)

    # 对每个矩阵应用权重并累加
    for i in range(num_matrices):
        if i < adj_weight.shape[0]:
            weight = adj_weight[i].item()

            # 处理不同类型的矩阵输入
            if isinstance(A[i], np.ndarray):
                matrix_data = A[i]
            elif hasattr(A[i], 'toarray'):  # 稀疏矩阵
                matrix_data = A[i].toarray()
            elif isinstance(A[i], torch.Tensor):
                matrix_data = A[i].cpu().numpy()
            else:
                matrix_data = np.array(A[i])

            # 应用权重并累加
            result += weight * matrix_data

    # 转换为torch tensor并移到目标设备
    return torch.from_numpy(result).float().to(device)


def adj_matrix_weight_merge_simple(A, adj_weight):
    """
    更简单的版本：直接使用numpy处理，避免所有torch稀疏张量问题
    """
    device = adj_weight.device

    # 在CPU上使用numpy处理所有计算
    num_matrices = len(A)
    result = np.zeros(A[0].shape, dtype=np.float32)

    for i in range(num_matrices):
        if i < adj_weight.shape[0]:
            weight = adj_weight[i].item()

            # 转换为numpy数组
            if isinstance(A[i], torch.Tensor):
                matrix_np = A[i].cpu().numpy()
            elif hasattr(A[i], 'toarray'):
                matrix_np = A[i].toarray()
            else:
                matrix_np = np.array(A[i])

            result += weight * matrix_np

    return torch.from_numpy(result).float().to(device)