import torch
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import csc_matrix
import os
from src.Utils import get_model
from src.args import get_citation_args


def load_trained_model(model_path='trained_model.pth'):
    """加载训练好的模型"""
    # 确保在CPU上加载
    checkpoint = torch.load(model_path, map_location='cpu')

    # 重建模型
    args = checkpoint['args']
    model = get_model(args.model, 878, checkpoint['A'], checkpoint['B'],
                      checkpoint['o_ass'], args.hidden, 256, args.dropout,
                      False, stdv=1 / 72, layer=2)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def ensure_tensor_on_device(tensor, device):
    """确保张量在指定设备上"""
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor).to(device)
    else:
        return tensor


def predict_single_disease(model, feature, A, B, o_ass, layer, disease_name,
                           mirna_list, disease_list, top_k=50):
    """预测单个疾病的相关miRNA"""
    # 使用CPU设备
    device = torch.device("cpu")
    model.to(device)

    # 确保所有输入数据都在CPU上
    feature = ensure_tensor_on_device(feature, device)
    A = ensure_tensor_on_device(A, device)
    B = ensure_tensor_on_device(B, device)
    o_ass = ensure_tensor_on_device(o_ass, device)

    # 获取模型嵌入
    with torch.no_grad():
        model.eval()
        emb = model(feature, A, B, o_ass, layer)
        emb = emb.cpu().numpy()  # 确保在CPU上

    # 找到疾病索引
    disease_idx = None
    for i, disease in enumerate(disease_list):
        if disease_name.lower() in disease[1].lower():
            disease_idx = i
            break

    if disease_idx is None:
        print(f"错误: 未找到疾病 '{disease_name}'")
        return None

    # 计算相似度分数
    disease_embedding_idx = 495 + disease_idx
    disease_emb = emb[disease_embedding_idx]

    scores = []
    for mirna_idx in range(495):
        mirna_emb = emb[mirna_idx]
        # 使用余弦相似度
        norm_mirna = np.linalg.norm(mirna_emb)
        norm_disease = np.linalg.norm(disease_emb)
        if norm_mirna > 0 and norm_disease > 0:
            score = np.dot(mirna_emb, disease_emb) / (norm_mirna * norm_disease)
        else:
            score = 0.0
        scores.append((mirna_idx, score))

    # 排序并取前top_k
    scores.sort(key=lambda x: x[1], reverse=True)
    top_scores = scores[:top_k]

    # 构建结果
    results = []
    for rank, (mirna_idx, score) in enumerate(top_scores, 1):
        mirna_id = mirna_idx + 1
        mirna_name = mirna_list[mirna_idx][1] if mirna_idx < len(mirna_list) else f"miRNA-{mirna_id}"
        results.append({
            'Rank': rank,
            'miRNA_ID': mirna_id,
            'miRNA_Name': mirna_name,
            'Score': f"{score:.6f}",
            'Evidence': 'predicted'
        })

    return results


def load_list(file_path):
    """加载miRNA或疾病列表"""
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    results.append((int(parts[0]), parts[1]))
    except FileNotFoundError:
        print(f"警告: 未找到 {file_path} 文件")
    return results


def main():
    # 加载模型和数据
    print("加载训练好的模型...")
    model, checkpoint = load_trained_model()

    # 加载miRNA和疾病列表
    mirna_list = load_list('data/miRNA number.txt')
    disease_list = load_list('data/disease number.txt')

    print(f"加载了 {len(mirna_list)} 个miRNA和 {len(disease_list)} 个疾病")

    # 用户输入要预测的疾病
    print("\n可预测的疾病列表:")
    for i, (idx, name) in enumerate(disease_list[:20]):  # 显示前20个
        print(f"{idx}: {name}")
    print("...")

    disease_name = input("\n请输入要预测的疾病名称: ").strip()

    # 进行预测
    results = predict_single_disease(
        model, checkpoint['feature'], checkpoint['A'],
        checkpoint['B'], checkpoint['o_ass'], 2,
        disease_name, mirna_list, disease_list, 50
    )

    if results:
        # 保存结果
        output_dir = 'disease_predictions'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存为CSV
        df = pd.DataFrame(results)
        csv_file = os.path.join(output_dir, f"{disease_name.replace(' ', '_')}_predictions.csv")
        df.to_csv(csv_file, index=False)

        # 保存为文本表格
        txt_file = os.path.join(output_dir, f"{disease_name.replace(' ', '_')}_predictions.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Top 50 {disease_name}-related miRNAs predicted by MV-HAGCN\n")
            f.write("RANK\tmiRNA name\tEvidence\n")
            for pred in results:
                f.write(f"{pred['Rank']}\t{pred['miRNA_Name']}\t{pred['Evidence']}\n")

        print(f"\n预测完成! 结果已保存到:")
        print(f"CSV格式: {csv_file}")
        print(f"文本格式: {txt_file}")

        # 显示前10个结果
        print(f"\n{disease_name} 相关的前10个miRNA:")
        print("Rank\tmiRNA\tScore")
        for i in range(min(10, len(results))):
            print(f"{results[i]['Rank']}\t{results[i]['miRNA_Name']}\t{results[i]['Score']}")
    else:
        print("预测失败，请检查疾病名称是否正确")


if __name__ == "__main__":
    main()