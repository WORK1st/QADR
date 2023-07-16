import torch
import torch.nn as nn

# 示例字典A
A = {
    1: [2, 3],
    2: [1, 4],
    3: [1, 4],
    4: [2, 3, 5],
    5: [4]
}

# 计算矩阵B的维度
m = len(A)  # 实体节点数量
n = max(len(neighbors) for neighbors in A.values())  # 邻居列表中的最大长度

# 创建矩阵B并用0填充
B = torch.zeros((m, n+1), dtype=torch.long)
for i, (key, neighbors) in enumerate(A.items()):
    B[i, 0] = key  # 实体id作为第一列
    B[i, 1:len(neighbors)+1] = torch.tensor(neighbors)  # 邻居实体id

# 将矩阵B转换为张量
tensor_B = torch.tensor(B)

# 构建图卷积神经网络模型
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        return x

# 设置通用参数
input_dim = m
hidden_dim = 100
output_dim = 200

# 创建GNN模型实例
model = GNN(input_dim, hidden_dim, output_dim)

# 对张量B进行特征抽取
embedded_tensor = model(tensor_B)

print(embedded_tensor.shape)  # 打印处理后的张量形状
