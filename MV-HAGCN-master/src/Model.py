import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch import tensor
from src.Decoupling_matrix_aggregation import adj_matrix_weight_merge

from src.args import get_citation_args

args = get_citation_args()


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, stdv, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(stdv)

    def reset_parameters(self, stdv):
        print("stdv:", stdv)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input1, adj):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            input1 = input1.float()
        except:
            pass

        support = torch.mm(input1.to(device), self.weight)
        output = torch.mm(adj.to(torch.double), support.to(torch.double))
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class ECAAttention(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.unsqueeze(2)).view(b, 1, c)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c)
        return x * y.expand_as(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate Attention Score
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply Attention Weights
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.out_proj(out)

        return out


class EnhancedGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features,
                 attention_type='eca',  # choose: 'eca', 'coord', 'multihead', 'residual'
                 num_heads=8, dropout=0.0, activation='relu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.attention_type = attention_type
        self.dropout_rate = dropout

        # Weighting Parameters
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        if attention_type == 'multihead':
            self.attention = MultiHeadSelfAttention(embed_dim=in_features, num_heads=2, dropout=dropout)
        elif attention_type == 'eca':
            self.attention = ECAAttention(channels=in_features)
        else:
            self.attention = None

        # Dropout Layer
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            self.activation = nn.ReLU(inplace=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu' if self.activation else 'linear'))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, adj):
        device = next(self.parameters()).device

        if x.dim() != 2:
            raise ValueError(f"Expected 2D input for x, got {x.dim()}D")
        if adj.dim() != 2:
            raise ValueError(f"Expected 2D input for adj, got {adj.dim()}D")
        if x.size(0) != adj.size(0):
            raise ValueError(f"Dimension mismatch: x has {x.size(0)} nodes, adj has {adj.size(0)} nodes")

        x = x.to(device)
        adj = adj.to(device)

        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        if adj.dtype != self.weight.dtype:
            adj = adj.to(self.weight.dtype)

        # Applying Attention Mechanisms
        if self.attention is not None:
            if self.attention_type == 'multihead':
                # Multi-head attention requires 3D input [batch, seq, features]
                x_3d = x.unsqueeze(0)  # [1, num_nodes, in_features]
                x_attended = self.attention(x_3d)
                x = x_attended.squeeze(0)  # [num_nodes, in_features]
            else:
                x = self.attention(x)

        # Linear transformation
        support = torch.mm(x, self.weight)

        # Dropout
        if hasattr(self, 'dropout') and self.training:
            support = self.dropout(support)

        # Image Convolution Operation
        output = torch.mm(adj, support)

        # Add bias
        if self.bias is not None:
            output = output + self.bias

        # Activation function
        if self.activation is not None:
            output = self.activation(output)

        return output


class MVHAGCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout, stdv, layer):
        super(MVHAGCN, self).__init__()
        print("layer:", layer)
        self.input_attention = ECAAttention(channels=nfeat, kernel_size=3)

        self.gc1 = EnhancedGraphConvolution(
            in_features=nfeat,
            out_features=out,
            # attention_type='multihead',
            dropout=dropout,
            activation='relu'
        )
        self.gc2 = EnhancedGraphConvolution(
            in_features=out,
            out_features=out,
            attention_type='multihead',
            dropout=dropout,
            activation='relu'
        )
        self.gc3 = EnhancedGraphConvolution(
            in_features=out,
            out_features=out,
            attention_type='multihead',
            dropout=dropout,
            activation='relu'
        )
        self.gc4 = EnhancedGraphConvolution(
            in_features=out,
            out_features=out,
            # attention_type='multihead',
            dropout=dropout,
            activation='relu'
        )
        self.gc5 = GraphConvolution(out, out, stdv)

        self.dropout = dropout

        self.weight_meta_path = Parameter(torch.FloatTensor(1444, 1444), requires_grad=True)
        self.weight_b = torch.nn.Parameter(torch.FloatTensor(5, 1), requires_grad=True)
        self.weight_c = torch.nn.Parameter(torch.FloatTensor(3, 1), requires_grad=True)

        # Convolved fusion coefficient
        self.weight_f = torch.nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.weight_f1 = torch.nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.weight_f2 = torch.nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.weight_f3 = torch.nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.weight_f4 = torch.nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)

        # Initialization parameters
        torch.nn.init.uniform_(self.weight_b, a=0.08, b=0.12)
        torch.nn.init.uniform_(self.weight_c, a=0.08, b=0.12)
        torch.nn.init.uniform_(self.weight_f, a=0.90, b=1.05)
        torch.nn.init.uniform_(self.weight_f1, a=0.10, b=0.30)
        torch.nn.init.uniform_(self.weight_f2, a=0.40, b=0.50)
        torch.nn.init.uniform_(self.weight_f3, a=0.40, b=0.50)
        torch.nn.init.uniform_(self.weight_f4, a=0.40, b=0.50)

        self.w_out = nn.ModuleList([
            nn.Linear(2 * out, out),
            nn.Linear(out, 128),
            nn.Linear(128, 64)
        ])
        self.w_interaction = nn.Linear(64, 2)

    def forward(self, feature, A, B, o_ass, layer, use_relu=True):
        print(feature.shape)

        # Retrieve the device where the model is located
        device = next(self.parameters()).device

        # Subnetwork Integration
        final_A = adj_matrix_weight_merge(A, self.weight_b.to(device))
        final_B = adj_matrix_weight_merge(B, self.weight_c.to(device))

        # Threshold processing
        index = torch.where(final_A < 0.015)
        final_A[index[0], index[1]] = 0
        index_B = torch.where(final_B < 0.15)
        final_B[index_B[0], index_B[1]] = 0

        # Construct the final adjacency matrix
        all0 = np.zeros((591, 591))
        all01 = np.zeros((853, 853))

        # Ensure all tensors reside on the same device.
        o_ass_tensor = torch.as_tensor(o_ass).to(device)
        o_ass_T_tensor = torch.as_tensor(o_ass.T).to(device)

        adjacency_matrix1 = torch.cat((o_ass_tensor, final_B.to(device)), 0)
        adjacency_matrix2 = torch.cat(
            (final_A.to(device), torch.fliplr(torch.flipud(o_ass_T_tensor))), 0)
        final_hcg = torch.cat((adjacency_matrix1, adjacency_matrix2), 1)

        # Process feature inputs and ensure they are delivered to the correct device.
        try:
            feature = torch.tensor(feature.astype(float).toarray(), device=device)
        except:
            try:
                feature = torch.from_numpy(feature.toarray()).to(device)
            except:
                feature = feature.to(device)

        # Use EnhancedGraphConvolution as the first two layers
        U1 = self.gc1(feature, final_hcg)
        if layer >= 2:
            U2 = self.gc2(U1, final_hcg)
        if layer >= 3:
            U3 = self.gc3(U2, final_hcg)
        if layer >= 4:
            U4 = self.gc4(U3, final_hcg)
        if layer >= 5:
            U5 = self.gc5(U4, final_hcg)

        # Merge the outputs of each layer
        if layer == 1:
            H = U1 * self.weight_f.to(device)
        elif layer == 2:
            H = U1 * self.weight_f.to(device) + U2 * self.weight_f1.to(device)
        elif layer == 3:
            H = U1 * self.weight_f.to(device) + U2 * self.weight_f1.to(device) + U3 * self.weight_f2.to(device)
        elif layer == 4:
            H = U1 * self.weight_f.to(device) + U2 * self.weight_f1.to(device) + U3 * self.weight_f2.to(
                device) + U4 * self.weight_f3.to(device)
        elif layer == 5:
            H = U1 * self.weight_f.to(device) + U2 * self.weight_f1.to(device) + U3 * self.weight_f2.to(
                device) + U4 * self.weight_f3.to(device) + U5 * self.weight_f4.to(device)

        return H


if __name__ == "__main__":
    Model = MVHAGCN(nfeat=1444, nhid=384, out=200, dropout=0, stdv=1. / 72, layer=2)
    print(Model)