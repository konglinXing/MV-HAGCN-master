import torch
torch.manual_seed(0)  # 设置CPU随机种子为0
torch.cuda.manual_seed_all(0)  # 设置所有GPU的随机种子为0
torch.backends.cudnn.deterministic = True  # 启用CuDNN确定性算法（保证结果可复现）
torch.backends.cudnn.benchmark = False  # 禁用CuDNN自动优化（避免引入随机性）
import torch.nn as nn  # 导入PyTorch神经网络模块

#定义逻辑回归模型 LogReg
class LogReg(nn.Module):
    """
    Logical classifier
    """

    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()  # 继承父类nn.Module的初始化方法
        self.fc = nn.Linear(ft_in, nb_classes)  # 定义全连接层

        # 遍历所有子模块并初始化权重
        for m in self.modules():
            self.weights_init(m)

# 权重初始化方法
    def weights_init(self, m):
        if isinstance(m, nn.Linear):  # 检查模块是否为线性层
            torch.nn.init.xavier_uniform_(m.weight.data)  # Xavier均匀分布初始化权重
            if m.bias is not None:
                m.bias.data.fill_(0.0)  # 偏置初始化为0

#前向传播
    def forward(self, seq):
        ret = self.fc(seq)  # 输入数据通过全连接层
        return ret  # 返回输出（未经过Softmax）