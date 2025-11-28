import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os
import glob


def plot_all_folds(results_dir='prediction_results'):
    """绘制所有fold的结果"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 收集所有fold的结果
        all_y_true = []
        all_y_scores = []
        fold_aucs = []
        fold_auprs = []

        # 读取所有fold文件
        fold_files = glob.glob(f'{results_dir}/fold_*_results.txt')
        fold_files.sort()

        for fold_file in fold_files:
            try:
                data = np.loadtxt(fold_file, skiprows=1)
                if len(data) > 0:
                    y_true = data[:, 0]
                    y_scores = data[:, 1]

                    all_y_true.extend(y_true)
                    all_y_scores.extend(y_scores)

                    # 计算当前fold的AUC和AUPR
                    fpr, tpr, _ = roc_curve(y_true, y_scores)
                    fold_auc = auc(fpr, tpr)
                    precision, recall, _ = precision_recall_curve(y_true, y_scores)
                    fold_aupr = auc(recall, precision)

                    fold_aucs.append(fold_auc)
                    fold_auprs.append(fold_aupr)

                    print(f"Loaded {fold_file}: {len(y_true)} samples, AUC={fold_auc:.4f}, AUPR={fold_aupr:.4f}")
            except Exception as e:
                print(f"Error loading {fold_file}: {e}")

        if not all_y_true:
            print("No data found!")
            return

        # 转换为numpy数组
        all_y_true = np.array(all_y_true)
        all_y_scores = np.array(all_y_scores)

        # 计算总体AUC和AUPR
        fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
        overall_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(all_y_true, all_y_scores)
        overall_aupr = auc(recall, precision)

        # 绘制曲线
        plt.figure(figsize=(12, 5))

        # ROC曲线
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'MV-HAGCN (AUC = {overall_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # PR曲线
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'MV-HAGCN (AUPR = {overall_aupr:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{results_dir}/combined_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 打印性能指标
        print(f"\n=== 总体性能指标 ===")
        print(f"AUC: {overall_auc:.4f}")
        print(f"AUPR: {overall_aupr:.4f}")
        print(f"平均AUC (±标准差): {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
        print(f"平均AUPR (±标准差): {np.mean(fold_auprs):.4f} ± {np.std(fold_auprs):.4f}")

    except Exception as e:
        print(f"绘图时出错: {e}")


if __name__ == "__main__":
    plot_all_folds()