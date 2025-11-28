import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def read_prediction_files(folder_path):
    """
    读取预测结果文件
    """
    results = {}
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

    for file_path in txt_files:
        filename = os.path.basename(file_path)
        print(f"正在处理文件: {filename}")

        # 从文件名提取embedding size
        size = None
        # 尝试匹配常见的embedding size模式
        if 'size=64' in filename or '64' in filename.replace('=', ''):
            size = 64
        elif 'size=128' in filename or '128' in filename.replace('=', ''):
            size = 128
        elif 'size=256' in filename or '256' in filename.replace('=', ''):
            size = 256
        elif 'size=512' in filename or '512' in filename.replace('=', ''):
            size = 512
        elif 'size=32' in filename or '32' in filename.replace('=', ''):
            size = 32
        elif 'size=16' in filename or '16' in filename.replace('=', ''):
            size = 16

        # 如果上述方法没有匹配到，尝试使用正则表达式
        if size is None:
            import re
            size_match = re.search(r'size[=_]?(\d+)', filename, re.IGNORECASE)
            if size_match:
                size = int(size_match.group(1))

        if size is None:
            print(f"无法从文件名推断embedding size，跳过: {filename}")
            continue

        true_labels = []
        pred_scores = []

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            # 跳过表头
            start_line = 0
            if 'True Labels' in lines[0] and 'Predicted Scores' in lines[0]:
                start_line = 1

            for line in lines[start_line:]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        true_label = int(float(parts[0]))  # 转换为整数
                        pred_score = float(parts[1])
                        true_labels.append(true_label)
                        pred_scores.append(pred_score)
                    except ValueError:
                        continue

        if true_labels and pred_scores:
            results[size] = {
                'true_labels': np.array(true_labels),
                'pred_scores': np.array(pred_scores),
                'filename': filename
            }
            print(f"成功读取: Embedding size={size}, 样本数={len(true_labels)}")
        else:
            print(f"文件没有有效数据: {filename}")

    return results


def plot_embedding_size_analysis(folder_path):
    """
    绘制embedding size参数敏感性分析的ROC和PR曲线
    """
    results = read_prediction_files(folder_path)

    if not results:
        print("未找到有效的结果文件")
        return

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 颜色和线型 - 使用更专业的配色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    line_styles = ['-', '--', '-.', ':', '-', '--']
    line_widths = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5]

    # 对embedding size进行排序
    sizes = sorted(results.keys())

    # 绘制ROC曲线
    for i, size in enumerate(sizes):
        true_labels = results[size]['true_labels']
        pred_scores = results[size]['pred_scores']

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(true_labels, pred_scores)
        roc_auc = auc(fpr, tpr)

        # 绘制ROC曲线
        ax1.plot(fpr, tpr,
                 color=colors[i % len(colors)],
                 linestyle=line_styles[i % len(line_styles)],
                 linewidth=line_widths[i % len(line_widths)],
                 label=f'Embedding size={size} (AUC = {roc_auc:.4f})')

    # 设置ROC图属性
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)  # 对角线
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=13)
    ax1.set_ylabel('True Positive Rate', fontsize=13)
    ax1.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 绘制PR曲线
    for i, size in enumerate(sizes):
        true_labels = results[size]['true_labels']
        pred_scores = results[size]['pred_scores']

        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
        avg_precision = average_precision_score(true_labels, pred_scores)

        # 绘制PR曲线
        ax2.plot(recall, precision,
                 color=colors[i % len(colors)],
                 linestyle=line_styles[i % len(line_styles)],
                 linewidth=line_widths[i % len(line_widths)],
                 label=f'Embedding size={size} (AUPR = {avg_precision:.4f})')

    # 设置PR图属性
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=13)
    ax2.set_ylabel('Precision', fontsize=13)
    ax2.set_title('P-R Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 添加总标题
    plt.suptitle('Analysis of Embedding Size', fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    # 先保存图片，再尝试显示
    plt.savefig('Embedding_size_analysis.png', dpi=300, bbox_inches='tight')
    print("图片已保存为 'Embedding_size_analysis.png'")

    # 尝试显示图片（在某些环境下可能无法显示）
    try:
        plt.show()
    except Exception as e:
        print(f"显示图片时出错: {e}")
        print("但图片已经成功保存，请查看文件 'Embedding_size_analysis.png'")

    # 打印详细指标
    print("\n详细指标:")
    print("尺寸\tAUC\t\tAUPR\t\t样本数\t文件名")
    for size in sizes:
        true_labels = results[size]['true_labels']
        pred_scores = results[size]['pred_scores']

        fpr, tpr, _ = roc_curve(true_labels, pred_scores)
        roc_auc = auc(fpr, tpr)
        avg_precision = average_precision_score(true_labels, pred_scores)

        print(f"{size}\t{roc_auc:.4f}\t\t{avg_precision:.4f}\t\t{len(true_labels)}\t\t{results[size]['filename']}")


# 使用示例
folder_path = r'C:\Users\Lenovo\Desktop\oursmodel 参数敏感性\embedding size'
plot_embedding_size_analysis(folder_path)