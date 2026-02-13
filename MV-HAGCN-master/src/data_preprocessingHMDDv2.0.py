import numpy as np
from scipy.io import savemat


def preprocess_data():
    """Preprocess all data and generate new .mat files."""
    # Load all miRNA similarity matrices
    fm = np.loadtxt('../data/mirna/functional similarity matrix.txt')
    gm = np.loadtxt('../data/mirna/M_GSM.txt')
    lm = np.loadtxt('../data/mirna/m_lnc_txt.txt')
    seq_sim = np.loadtxt('../data/mirna/SeqSim2_495x495.txt')
    fam_sim = np.loadtxt('../data/mirna/FAM.txt')

    # Load Disease similarity matrix
    fd = np.loadtxt('../data/disease/disease semantic similarity matrix 1.txt')
    gd = np.loadtxt('../data/disease/D_GSM.txt')
    ld = np.loadtxt('../data/disease/d_lnc_txt.txt')


    # Load the association matrix
    association = np.loadtxt('../data/miRNA_disease_matrix.csv', delimiter=',')

    # Construct matrices A and B
    A = np.array([fm, gm, lm, seq_sim, fam_sim])  # 5 miRNA similarity matrices


    B = np.array([fd, gd, ld])  # 3 Disease similarity matrices

    # Save as a new .mat file
    data_dict = {
        'A': A,
        'B': B,
        'raw_association': association,
        'feature': np.eye(878)
    }

    savemat('../data/result_expanded.mat', data_dict)
    print("Data preprocessing has been completed!")




if __name__ == "__main__":
    preprocess_data()