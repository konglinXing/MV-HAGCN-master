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
net_path = r"data/result_expandedHMDDv3.2.mat"

savepath = r'data/hcg_embedding'
eval_name = r'data'
file_name = r'data/train'
eval_type = 'all'



def load_txt_matrix(file_path, shape=(853, 853)):

    try:
        if os.path.exists(file_path):
            data = np.loadtxt(file_path)
            if data.shape != shape:

                if data.size == shape[0] * shape[1]:
                    data = data.reshape(shape)
                else:
                    print(f"warning: {file_path} Shape mismatch，use zero matrix")
                    data = np.zeros(shape)
            return data
        else:
            print(f"Warning: file {file_path} Does not exist, use zero matrix")
            return np.zeros(shape)
    except Exception as e:
        print(f"load {file_path} error: {e}，use zero matrix")
        return np.zeros(shape)



def extend_similarity_matrices(original_A, original_B):



    seq_sim = load_txt_matrix('data/mirna/SeqSim_853x853.txt', shape=(853, 853))
    fam_sim = load_txt_matrix('data/mirna/FAM_853x853.txt', shape=(853, 853))


    print(f"Sequence similarity matrix shape: {seq_sim.shape}")
    print(f"Family similarity matrix shape: {fam_sim.shape}")

    A_expanded = [
        original_A[0],  # FM
        original_A[1],  # KM
        original_A[2],  # LM
        seq_sim,   # SM
        fam_sim    # FAM
    ]


    return np.array(A_expanded), original_B



mat = loadmat(net_path)
train = mat['A']
train1 = mat['B']


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
    # Model Construction
    model = get_model(args.model, 1444, A, B, o_ass, args.hidden, 256, args.dropout, False, stdv=1 / 72, layer=2)
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




