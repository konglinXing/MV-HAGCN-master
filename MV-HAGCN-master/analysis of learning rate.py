import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import re
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def read_prediction_files_enhanced(folder_path):
    """
    增强版文件读取，支持更多文件名格式
    """
    results = {}
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

    # 常见学习率值
    common_lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

    for file_path in txt_files:
        filename = os.path.basename(file_path)
        print(f"正在处理文件: {filename}")

        # 从文件名提取学习率
        lr = None

        # 方法1: 直接匹配常见学习率值
        for common_lr in common_lrs:
            # 尝试多种表示方式
            lr_str = str(common_lr)
            if lr_str in filename:
                lr = common_lr
                break
            # 检查科学计数法表示
            if 'e-' in filename:
                sci_notation = f"{common_lr:.0e}".replace('+', '')
                if sci_notation in filename:
                    lr = common_lr
                    break

        # 方法2: 使用正则表达式提取数字
        if lr is None:
            # 匹配各种学习率表示模式
            patterns = [
                r'lr[=_:]?([0-9.]+)',  # lr=0.001, lr:0.001, lr_0.001
                r'learning.rate[=_:]?([0-9.]+)',  # learning.rate=0.001
                r'learning_rate[=_:]?([0-9.]+)',  # learning_rate=0.001
                r'rate[=_:]?([0-9.]+)',  # rate=0.001
                r'([0-9.]+)\.txt'  # 0.001.txt
            ]

            for pattern in patterns:
                match = re.search(pattern, filename, re.IGNORECASE)
                if match:
                    try:
                        lr_str = match.group(1)
                        lr = float(lr_str)
                        # 检查是否是科学计数法
                        if 'e-' in filename:
                            # 找到科学计数法部分
                            sci_match = re.search(r'(\d+)e-?(\d+)', filename, re.IGNORECASE)
                            if sci_match:
                                base = float(sci_match.group(1))
                                exp = int(sci_match.group(2))
                                lr = base * (10 ** -exp)
                        break
                    except ValueError:
                        continue

        if lr is None:
            print(f"无法从文件名推断学习率，跳过: {filename}")
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
                        true_label = int(float(parts[0]))
                        pred_score = float(parts[1])
                        true_labels.append(true_label)
                        pred_scores.append(pred_score)
                    except ValueError:
                        continue

        if true_labels and pred_scores:
            results[lr] = {
                'true_labels': np.array(true_labels),
                'pred_scores': np.array(pred_scores),
                'filename': filename
            }
            print(f"成功读取: Learning rate={lr}, 样本数={len(true_labels)}")
        else:
            print(f"文件没有有效数据: {filename}")

    return results


def plot_learning_rate_comparison(folder_path):
    """
    绘制学习率参数敏感性分析
    """
    results = read_prediction_files_enhanced(folder_path)

    if not results:
        print("未找到有效的结果文件")
        return

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 颜色和线型
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    line_styles = ['-', '--', '-.', ':', '-', '--']

    # 对学习率进行排序
    lrs = sorted(results.keys())

    # 绘制ROC曲线
    for i, lr in enumerate(lrs):
        true_labels = results[lr]['true_labels']
        pred_scores = results[lr]['pred_scores']

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(true_labels, pred_scores)
        roc_auc = auc(fpr, tpr)

        # 格式化学习率显示
        if lr < 0.001:
            lr_label = f"{lr:.4f}"
        elif lr < 0.01:
            lr_label = f"{lr:.3f}"
        else:
            lr_label = f"{lr:.2f}"

        # 绘制ROC曲线
        ax1.plot(fpr, tpr,
                 color=colors[i],
                 linestyle=line_styles[i % len(line_styles)],
                 linewidth=2.5,
                 label=f'LR={lr_label} (AUC={roc_auc:.4f})')

    # 设置ROC图属性
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves for Different Learning Rates', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 绘制PR曲线
    for i, lr in enumerate(lrs):
        true_labels = results[lr]['true_labels']
        pred_scores = results[lr]['pred_scores']

        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
        avg_precision = average_precision_score(true_labels, pred_scores)

        # 格式化学习率显示
        if lr < 0.001:
            lr_label = f"{lr:.4f}"
        elif lr < 0.01:
            lr_label = f"{lr:.3f}"
        else:
            lr_label = f"{lr:.2f}"

        # 绘制PR曲线
        ax2.plot(recall, precision,
                 color=colors[i],
                 linestyle=line_styles[i % len(line_styles)],
                 linewidth=2.5,
                 label=f'LR={lr_label} (AUPR={avg_precision:.4f})')

    # 设置PR图属性
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves for Different Learning Rates', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    output_path = 'Learning_rate_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存为 '{output_path}'")

    # 尝试显示
    try:
        plt.show()
    except:
        print("图片已保存，但无法显示")

    # 打印详细指标
    print("\n详细指标:")
    print("学习率\tAUC\t\tAUPR\t\t样本数\t文件名")
    for lr in lrs:
        true_labels = results[lr]['true_labels']
        pred_scores = results[lr]['pred_scores']

        fpr, tpr, _ = roc_curve(true_labels, pred_scores)
        roc_auc = auc(fpr, tpr)
        avg_precision = average_precision_score(true_labels, pred_scores)

        print(f"{lr}\t{roc_auc:.4f}\t\t{avg_precision:.4f}\t\t{len(true_labels)}\t\t{results[lr]['filename']}")


# 使用示例
folder_path = r'C:\Users\Lenovo\Desktop\oursmodel 参数敏感性\learning rate'
plot_learning_rate_comparison(folder_path)