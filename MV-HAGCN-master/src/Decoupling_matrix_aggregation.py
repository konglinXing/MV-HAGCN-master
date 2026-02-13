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
# #     """Expanded to support aggregation of 5 matrices"""
# #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #     N = A[0].shape[0]
# #     temp = coo_matrix((N, N))
# #     temp = coototensor(temp)
# #
# #
# #     a = coototensor(csc_matrix(A[0]).tocoo())
# #     b = coototensor(csc_matrix(A[1]).tocoo())
# #     c = coototensor(csc_matrix(A[2]).tocoo())
# #     d = coototensor(csc_matrix(A[3]).tocoo())
# #     e = coototensor(csc_matrix(A[4]).tocoo())
# #
# #     A_t = torch.stack([a, b, c, d, e], dim=2).to_dense()
# #
# #     A_t = A_t.to(device)
# #     temp = torch.matmul(A_t, adj_weight)
# #     temp = torch.squeeze(temp, 2)
# #
# #     return temp
# def adj_matrix_weight_merge(A, adj_weight):
#     """
#     Multiplex Relation Aggregation - Dynamically process any number of matrices
#     """
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     N = A[0].shape[0]
#     temp = coo_matrix((N, N))
#     temp = coototensor(temp)
#
#     # Number of Dynamic Processing Matrices
#     num_matrices = len(A)
#     matrices = []
#
#     for i in range(num_matrices):
#         matrices.append(coototensor(csc_matrix(A[i]).tocoo()))
#
#     # Using matrices with dynamic quantities
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
    Simplified matrix aggregation to avoid sparse tensor conversion issues
Processed directly in NumPy, then converted to a tensor
    """
    device = adj_weight.device

    # Get the number of matrices
    num_matrices = len(A)

    # Initialize the result matrix
    if isinstance(A[0], np.ndarray):
        result_shape = A[0].shape
        result = np.zeros(result_shape, dtype=np.float32)
    else:
        result_shape = A[0].shape
        result = np.zeros(result_shape, dtype=np.float32)

    # Apply weights to each matrix and accumulate them.
    for i in range(num_matrices):
        if i < adj_weight.shape[0]:
            weight = adj_weight[i].item()

            # Processing different types of matrix inputs
            if isinstance(A[i], np.ndarray):
                matrix_data = A[i]
            elif hasattr(A[i], 'toarray'):  # Sparse Matrix
                matrix_data = A[i].toarray()
            elif isinstance(A[i], torch.Tensor):
                matrix_data = A[i].cpu().numpy()
            else:
                matrix_data = np.array(A[i])

            # Apply weights and accumulate
            result += weight * matrix_data

    # Convert to a torch tensor and move it to the target device
    return torch.from_numpy(result).float().to(device)


def adj_matrix_weight_merge_simple(A, adj_weight):

    device = adj_weight.device

    # Perform all computations using NumPy on the CPU
    num_matrices = len(A)
    result = np.zeros(A[0].shape, dtype=np.float32)

    for i in range(num_matrices):
        if i < adj_weight.shape[0]:
            weight = adj_weight[i].item()

            # Convert to a NumPy array
            if isinstance(A[i], torch.Tensor):
                matrix_np = A[i].cpu().numpy()
            elif hasattr(A[i], 'toarray'):
                matrix_np = A[i].toarray()
            else:
                matrix_np = np.array(A[i])

            result += weight * matrix_np

    return torch.from_numpy(result).float().to(device)