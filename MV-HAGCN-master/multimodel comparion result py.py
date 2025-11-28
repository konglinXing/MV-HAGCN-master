import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os
import glob
import pandas as pd

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_predictions_from_subfolders(main_folder):
    """
    从主文件夹下的各个子文件夹加载所有模型的预测结果
    """
    models_data = {}

    # 获取主文件夹下的所有子文件夹
    subfolders = [f for f in os.listdir(main_folder)
                  if os.path.isdir(os.path.join(main_folder, f))]

    print(f"找到的子文件夹: {subfolders}")

    for model_name in subfolders:
        model_folder = os.path.join(main_folder, model_name)
        print(f"处理模型: {model_name}, 路径: {model_folder}")

        # 查找该模型文件夹中的所有结果文件
        pattern = os.path.join(model_folder, "*fold*.txt")
        files = glob.glob(pattern)

        if not files:
            # 如果没有找到fold文件，尝试查找任何txt文件
            pattern = os.path.join(model_folder, "*.txt")
            files = glob.glob(pattern)

        print(f"  找到 {len(files)} 个文件")

        all_true_labels = []
        all_pred_scores = []

        for file in sorted(files):
            try:
                print(f"    读取文件: {os.path.basename(file)}")
                data = np.loadtxt(file, skiprows=1)  # 跳过表头
                if data.ndim == 2 and data.shape[1] == 2:
                    true_labels = data[:, 0]
                    pred_scores = data[:, 1]
                    all_true_labels.append(true_labels)
                    all_pred_scores.append(pred_scores)
                    print(f"      成功读取，数据形状: {data.shape}")
                else:
                    print(f"      文件格式不符合要求，维度: {data.ndim}")
            except Exception as e:
                print(f"      读取文件时出错: {e}")

        if all_true_labels:
            models_data[model_name] = {
                'true_labels': all_true_labels,
                'pred_scores': all_pred_scores
            }
            print(f"  模型 {model_name} 成功加载 {len(all_true_labels)} 个fold的数据")
        else:
            print(f"  模型 {model_name} 没有有效数据")

    return models_data


def calculate_mean_roc_curves(all_true_labels, all_pred_scores):
    """
    计算平均ROC曲线
    """
    base_fpr = np.linspace(0, 1, 101)
    tprs = []
    aucs = []

    for i in range(len(all_true_labels)):
        fpr, tpr, _ = roc_curve(all_true_labels[i], all_pred_scores[i])

        # 使用numpy.interp替代scipy.interp
        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        # 使用原始的fpr和tpr计算AUC
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    return base_fpr, mean_tpr, mean_auc, std_auc, tprs


def calculate_mean_pr_curves(all_true_labels, all_pred_scores):
    """
    计算平均PR曲线
    """
    base_recall = np.linspace(0, 1, 101)
    precisions = []
    auprs = []

    for i in range(len(all_true_labels)):
        precision, recall, _ = precision_recall_curve(all_true_labels[i], all_pred_scores[i])

        # 反转数组以确保recall是递增的
        precision = precision[::-1]
        recall = recall[::-1]

        # 使用numpy.interp
        precision_interp = np.interp(base_recall, recall, precision)
        precisions.append(precision_interp)

        # 计算AUPR
        fold_aupr = average_precision_score(all_true_labels[i], all_pred_scores[i])
        auprs.append(fold_aupr)

    mean_precision = np.mean(precisions, axis=0)
    mean_aupr = np.mean(auprs)
    std_aupr = np.std(auprs)

    return base_recall, mean_precision, mean_aupr, std_aupr, precisions


def plot_comparison_figures(main_folder, output_dir='./comparison_results'):
    """
    绘制多模型ROC和PR曲线对比图
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载所有模型的数据
    models_data = load_predictions_from_subfolders(main_folder)

    if not models_data:
        print("错误: 没有找到任何模型的有效数据")
        return None

    print(f"\n成功加载 {len(models_data)} 个模型的数据: {list(models_data.keys())}")

    # 定义颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 存储AUC和AUPR结果
    results_table = []

    # 为每个模型绘制曲线
    for i, (model_name, data) in enumerate(models_data.items()):
        color = colors[i % len(colors)]

        try:
            # 计算ROC曲线
            base_fpr, mean_tpr, mean_auc, std_auc, tprs = calculate_mean_roc_curves(
                data['true_labels'], data['pred_scores'])

            # 计算PR曲线
            base_recall, mean_precision, mean_aupr, std_aupr, precisions = calculate_mean_pr_curves(
                data['true_labels'], data['pred_scores'])

            # 绘制ROC曲线
            ax1.plot(base_fpr, mean_tpr,
                     color=color,
                     linestyle='-',
                     linewidth=2,
                     label=f'{model_name} (AUC = {mean_auc:.4f} ± {std_auc:.4f})')

            # 绘制PR曲线
            ax2.plot(base_recall, mean_precision,
                     color=color,
                     linestyle='-',
                     linewidth=2,
                     label=f'{model_name} (AUPR = {mean_aupr:.4f} ± {std_aupr:.4f})')

            # 存储结果
            results_table.append({
                'Model': model_name,
                'AUC': f'{mean_auc:.4f} ± {std_auc:.4f}',
                'AUPR': f'{mean_aupr:.4f} ± {std_aupr:.4f}'
            })

            print(f"模型 {model_name}: AUC = {mean_auc:.4f} ± {std_auc:.4f}, AUPR = {mean_aupr:.4f} ± {std_aupr:.4f}")

        except Exception as e:
            print(f"处理模型 {model_name} 时出错: {e}")
            continue

    # 设置ROC图
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 设置PR图
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('PR Curves Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(os.path.join(output_dir, 'multi_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'multi_model_comparison.pdf'), bbox_inches='tight')

    # 显示图片
    plt.show()

    # 创建并保存结果表格
    if results_table:
        df = pd.DataFrame(results_table)
        df.to_csv(os.path.join(output_dir, 'model_comparison_results.csv'), index=False)
        print("\n模型比较结果:")
        print(df.to_string(index=False))

    return results_table


def create_publication_style_plot(main_folder, output_dir='./publication_figures'):
    """
    创建适合发表的样式图
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载所有模型的数据
    models_data = load_predictions_from_subfolders(main_folder)

    if not models_data:
        print("错误: 没有找到任何模型的有效数据")
        return None

    # 更专业的颜色方案
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3E92CC']

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    results_table = []

    for i, (model_name, data) in enumerate(models_data.items()):
        color = colors[i % len(colors)]

        try:
            # ROC曲线
            base_fpr, mean_tpr, mean_auc, std_auc, _ = calculate_mean_roc_curves(
                data['true_labels'], data['pred_scores'])
            # PR曲线
            base_recall, mean_precision, mean_aupr, std_aupr, _ = calculate_mean_pr_curves(
                data['true_labels'], data['pred_scores'])

            # 绘制ROC
            ax1.plot(base_fpr, mean_tpr,
                     color=color,
                     linestyle='-',
                     linewidth=2.5,
                     label=f'{model_name}\n(AUC = {mean_auc:.3f})')

            # 绘制PR
            ax2.plot(base_recall, mean_precision,
                     color=color,
                     linestyle='-',
                     linewidth=2.5,
                     label=f'{model_name}\n(AUPR = {mean_aupr:.3f})')

            results_table.append({
                'Model': model_name,
                'AUC': f'{mean_auc:.3f}',
                'AUPR': f'{mean_aupr:.3f}'
            })

        except Exception as e:
            print(f"处理模型 {model_name} 时出错: {e}")
            continue

    # 美化ROC图
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax1.set_title('(a) ROC Curves', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.2)

    # 美化PR图
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax2.set_title('(b) PR Curves', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'publication_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'publication_comparison.pdf'), bbox_inches='tight')
    plt.show()

    return results_table


# 主程序
if __name__ == "__main__":
    # 设置你的主文件夹路径
    main_folder = r"C:\Users\Lenovo\Desktop\多模型ROC和PR曲线对比图5fold"

    print("开始绘制多模型对比图...")

    # 绘制标准对比图
    results = plot_comparison_figures(main_folder)

    # 绘制发表样式图
    publication_results = create_publication_style_plot(main_folder)

    print("\n绘图完成！结果保存在:")
    print("- comparison_results/ 文件夹：标准对比图")
    print("- publication_figures/ 文件夹：发表样式图")


