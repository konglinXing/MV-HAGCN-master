import numpy as np
from scipy.io import savemat


def preprocess_data():
    """预处理所有数据并生成新的.mat文件"""
    # 加载所有miRNA相似性矩阵
    fm = np.loadtxt('../data/mirna/functional similarity matrix.txt')
    gm = np.loadtxt('../data/mirna/M_GSM.txt')
    lm = np.loadtxt('../data/mirna/m_lnc_txt.txt')  # 可能需要调整
    seq_sim = np.loadtxt('../data/mirna/SeqSim2_495x495.txt')
    fam_sim = np.loadtxt('../data/mirna/FAM.txt')

    # 加载疾病相似性矩阵
    fd = np.loadtxt('../data/disease/disease semantic similarity matrix 1.txt')
    gd = np.loadtxt('../data/disease/D_GSM.txt')
    ld = np.loadtxt('../data/disease/d_lnc_txt.txt')
    # td = np.loadtxt('../data/disease/disease_target_similarity_383x383.txt')  # 新增靶标相似性

    # 加载关联矩阵
    association = np.loadtxt('../data/miRNA_disease_matrix.csv', delimiter=',')

    # 构建A和B矩阵
    A = np.array([fm, gm, lm, seq_sim, fam_sim])  # 5个miRNA相似性矩阵


    B = np.array([fd, gd, ld])  # 3个疾病相似性矩阵
    # B = np.array([fd, gd, ld, td])  # 4个疾病相似性矩阵（新增靶标相似性）
    # 保存为新的.mat文件
    data_dict = {
        'A': A,
        'B': B,
        'raw_association': association,
        'feature': np.eye(878)  # 单位矩阵作为特征
    }

    savemat('../data/result_expanded.mat', data_dict)
    print("Data preprocessing has been completed!")




if __name__ == "__main__":
    preprocess_data()