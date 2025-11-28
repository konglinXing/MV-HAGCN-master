import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os
import glob

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def load_data_from_file(file_path):
    """
    从单个文件加载预测数据
    """
    true_labels = []
    pred_scores = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or line == '':
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        true_labels.append(int(float(parts[0])))  # 处理可能的浮点数格式
                        pred_scores.append(float(parts[1]))
                    except ValueError:
                        # 跳过无法转换的行
                        continue
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None, None

    if len(true_labels) == 0:
        print(f"在 {file_path} 中没有找到有效数据")
        return None, None

    print(f"从 {os.path.basename(file_path)} 加载了 {len(true_labels)} 个样本")
    return np.array(true_labels), np.array(pred_scores)


def find_experiment_files(base_path):
    """
    在基础路径中查找实验文件并映射到对应的缩写
    """
    # 定义文件名关键词与缩写的映射
    file_patterns = {
        'without attention': 'w/o AT',
        'without enhancedgraphconvolution': 'w/o EGCN',
        'without multi-source data fusion': 'w/o MS'
    }

    experiment_files = {}

    # 获取基础路径中的所有文件
    all_files = [f for f in os.listdir(base_path)
                 if os.path.isfile(os.path.join(base_path, f))]

    print(f"在目录中找到 {len(all_files)} 个文件:")
    for file in all_files:
        print(f"  - {file}")

    # 匹配文件名
    for pattern, abbreviation in file_patterns.items():
        matched_files = [f for f in all_files if pattern.lower() in f.lower()]
        if matched_files:
            # 使用第一个匹配的文件
            experiment_files[abbreviation] = os.path.join(base_path, matched_files[0])
            print(f"匹配: {matched_files[0]} -> {abbreviation}")
        else:
            print(f"未找到匹配 {pattern} 的文件")

    return experiment_files


def plot_ablation_results(base_path):
    """
    绘制消融实验的ROC和PR曲线
    """
    # 查找实验文件
    experiment_files = find_experiment_files(base_path)

    if not experiment_files:
        print("没有找到任何实验文件！")
        return {}, {}

    # 颜色和线型
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    line_styles = ['-', '--', '-.', ':']

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 存储AUC和AUPR值用于图例
    auc_values = {}
    aupr_values = {}

    for i, (exp_name, file_path) in enumerate(experiment_files.items()):
        # 加载数据
        true_labels, pred_scores = load_data_from_file(file_path)

        if true_labels is None or len(true_labels) == 0:
            print(f"在 {file_path} 中没有有效数据")
            continue

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(true_labels, pred_scores)
        roc_auc = auc(fpr, tpr)
        auc_values[exp_name] = roc_auc

        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
        pr_auc = auc(recall, precision)
        aupr_values[exp_name] = pr_auc

        # 绘制ROC曲线
        ax1.plot(fpr, tpr, color=colors[i], linestyle=line_styles[i],
                 linewidth=2.5, label=f'{exp_name} (AUC = {roc_auc:.4f})')

        # 绘制PR曲线
        ax2.plot(recall, precision, color=colors[i], linestyle=line_styles[i],
                 linewidth=2.5, label=f'{exp_name} (AUPR = {pr_auc:.4f})')

        print(f"{exp_name}: AUC = {roc_auc:.4f}, AUPR = {pr_auc:.4f}")

    # 设置ROC图属性
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)  # 对角线
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves - Ablation Experiments', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 设置PR图属性
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('P-R Curves - Ablation Experiments', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(base_path, 'ablation_study_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n图表已保存到: {output_path}")

    # 返回性能指标
    return auc_values, aupr_values


# 主程序
if __name__ == "__main__":
    # 设置消融实验数据的基础路径
    base_path = r"C:\Users\Lenovo\Desktop\oursmodel消融实验"

    if not os.path.exists(base_path):
        print(f"基础路径不存在: {base_path}")
        print("请检查路径是否正确")
    else:
        print("开始绘制消融实验曲线图...")
        print(f"数据路径: {base_path}")

        # 绘制ROC和PR曲线
        auc_values, aupr_values = plot_ablation_results(base_path)

        # 创建性能比较表格
        if auc_values and aupr_values:
            create_performance_table(auc_values, aupr_values, base_path)

            # 打印总结
            print("\n=== 消融实验性能总结 ===")
            for method in auc_values.keys():
                print(f"{method}: AUC = {auc_values[method]:.4f}, AUPR = {aupr_values[method]:.4f}")
        else:
            print("没有成功加载任何数据，请检查文件格式和路径")