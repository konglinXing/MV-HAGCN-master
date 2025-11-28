import matplotlib.font_manager
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix
import time
import argparse
from src.Utils import load_our_data, get_model
from src.args import get_citation_args
from src.link_prediction_evaluate import predict_model
import os

args = get_citation_args()
parser = argparse.ArgumentParser()
myfont = matplotlib.font_manager.FontProperties(fname=r"C:\Windows\Fonts\CaeciliaLTStd-Roman.otf")

# 路径配置
args.dataset = 'result'
net_path = r"data/result_expanded.mat"

savepath = r'data/hcg_embedding'
eval_name = r'data'
file_name = r'data/train'
eval_type = 'all'


# === 新增：加载txt相似性矩阵的函数 ===
def load_txt_matrix(file_path, shape=(495, 495)):
    """加载txt格式的相似性矩阵"""
    try:
        if os.path.exists(file_path):
            data = np.loadtxt(file_path)
            if data.shape != shape:
                # 如果形状不匹配，尝试重塑
                if data.size == shape[0] * shape[1]:
                    data = data.reshape(shape)
                else:
                    print(f"警告: {file_path} 形状不匹配，使用零矩阵")
                    data = np.zeros(shape)
            return data
        else:
            print(f"警告: 文件 {file_path} 不存在，使用零矩阵")
            return np.zeros(shape)
    except Exception as e:
        print(f"加载 {file_path} 时出错: {e}，使用零矩阵")
        return np.zeros(shape)


# === 新增：扩展相似性矩阵 ===
def extend_similarity_matrices(original_A, original_B):
    """扩展原有的相似性矩阵，添加新的数据源"""

    # 加载新的miRNA相似性矩阵
    seq_sim = load_txt_matrix('data/mirna/SeqSim2_495x495.txt', shape=(495, 495))
    fam_sim = load_txt_matrix('data/mirna/FAM.txt', shape=(495, 495))


    print(f"Sequence similarity matrix shape: {seq_sim.shape}")
    print(f"Family similarity matrix shape: {fam_sim.shape}")

    A_expanded = [
        original_A[0],  # FM - 功能相似性
        original_A[1],  # KM - 高斯核相似性
        original_A[2],  # LM - lncRNA相似性
        seq_sim,  # 新增：序列相似性
        fam_sim  # 新增：家族相似性
    ]


    return np.array(A_expanded), original_B


# 加载原始.mat文件
mat = loadmat(net_path)
train = mat['A']  # 原始的A矩阵（3个miRNA相似性矩阵）
train1 = mat['B']  # B矩阵（疾病相似性矩阵）

# === 修改：扩展A矩阵 ===
print("Original A matrix shape:", train.shape)
A, B = extend_similarity_matrices(train, train1)
print("Expanded A matrix shape:", A.shape)
print("Expanded B matrix shape:", B.shape)


try:
    feature = mat['full_feature']
except:
    try:
        feature = mat['feature']
    except:
        try:
            feature = mat['features']
        except:
            feature = mat['node_feature']

feature = csc_matrix(feature) if type(feature) != csc_matrix else feature

node_matching = False
o_ass = mat['raw_association']
aus, f1s, recalls, accs, fprs, tprs, auprs = [], [], [], [], [], [], []


for i in [1]:
    print(i)
    t1 = time.time()
    # 构建模型
    model = get_model(args.model, 878, A, B, o_ass, args.hidden, 256, args.dropout, False, stdv=1 / 72, layer=2)
    auc, f1, recall, acc, fpr, tpr, aupr, ps, rs = predict_model(model, file_name, feature, A, B, o_ass, eval_type,
                                                                 0.0005,
                                                                 256, 2, 1)
    # auc, f1, recall, acc, fpr, tpr, aupr = predict_model(model, file_name, feature, A, B, o_ass, eval_type, 0.0005,
    #                                                      256, 2, 1)
    t2 = time.time()
    print('running time:{}'.format(t2-t1))
    aus.append(auc)
    f1s.append(f1)
    recalls.append(recall)
    accs.append(acc)
    fprs.append(fpr)
    tprs.append(tpr)
    auprs.append(aupr)
    print('Test auc: {:.10f}, F1: {:.10f}, aupr: {:.10f}, acc:{:.10f}, recall:{:.10f}'.format(auc, f1, aupr, acc,
                                                                                              recall))
print(aus)
print(f1s)
print(accs)
print(auprs)
print(recalls)




