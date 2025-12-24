import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNetwork(nn.Module):
    def __init__(self):
        super(MNISTNetwork, self).__init__()
        # 定义仿射层 (Affine Layers: 权重 + 偏置)
        # 在 PyTorch 中，nn.Linear 执行 x @ W.T + b 操作
        # 第 1 层：784 输入 -> 50 隐藏 
        self.fc1 = nn.Linear(in_features=784, out_features=50)
        
        # 第 2 层：50 隐藏 -> 100 隐藏 
        self.fc2 = nn.Linear(in_features=50, out_features=100)
        
        # 第 3 层 (输出层)：100 隐藏 -> 10 输出 
        self.fc3 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        """
        对应 3.4 节的前向传播 (Forward Pass) 实现。
        
        参数:
            x: 包含图像批次 (batch) 的输入张量。
        """
        # --- 输入预处理 ---
        # 展平输入: (N, 1, 28, 28) -> (N, 784)
        # N = 批次大小 batch_size (例如书中的 100) [cite: 879]
        x = torch.flatten(x, start_dim=1)      # Shape: [N, 784]

        # --- 第 1 层 ---
        # 矩阵乘法 (点积) + 偏置 [cite: 461]
        a1 = self.fc1(x)                       # Shape: [N, 50]
        # 激活函数: Sigmoid [cite: 82, 479]
        z1 = torch.sigmoid(a1)                 # Shape: [N, 50]

        # --- 第 2 层 ---
        # 矩阵乘法 + 偏置
        a2 = self.fc2(z1)                      # Shape: [N, 100]
        # 激活函数: Sigmoid
        z2 = torch.sigmoid(a2)                 # Shape: [N, 100]

        # --- 第 3 层 (输出层) ---
        # 矩阵乘法 + 偏置
        a3 = self.fc3(z2)                      # Shape: [N, 10]
        
        # 输出激活: Softmax
        # 注意: 在训练中我们通常直接返回原始 logits (a3)，
        # 但为了推理 (prediction) 和解释概率，我们应用 Softmax [cite: 621, 698]
        y = F.softmax(a3, dim=1)               # Shape: [N, 10]
        
        return y

# --- 使用示例 (批处理 Batch Processing) ---
if __name__ == "__main__":
    # 模拟一个包含 100 张图像的批次数据 (28x28) [cite: 879]
    batch_size = 100
    dummy_input = torch.randn(batch_size, 1, 28, 28) # Shape: [100, 1, 28, 28]
    
    # 初始化模型
    model = MNISTNetwork()
    
    # 执行推理 (Inference)
    predictions = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")       # [100, 1, 28, 28]
    print(f"Output Shape: {predictions.shape}")      # [100, 10]
    print(f"Sum of probs (Row 0): {predictions[0].sum().item():.4f}") # 应该接近 1.0 [cite: 696]