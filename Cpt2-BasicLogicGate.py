import torch
import torch.nn as nn

class BasicLogicGate(nn.Module):
    """
    实现了书中定义的标准感知机逻辑。
    为了完全还原书中的 '阶跃函数' (Step Function) 行为，
    这里不使用 Sigmoid，而是使用硬阈值判定。
    """
    def __init__(self, w: list, b: float):
        super().__init__()
        # 将权重和偏置注册为 Buffer，因为这些是固定逻辑门，不需要梯度更新
        # shape: [2] -> [2, 1] 用于矩阵乘法广播
        self.register_buffer('w', torch.tensor(w, dtype=torch.float32).unsqueeze(1)) 
        self.register_buffer('b', torch.tensor(b, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: 输入 Tensor. Shape: [Batch_Size, 2]
        """
        # 1. 线性变换 (Linear Transformation)
        # x: [Batch, 2], w: [2, 1]
        # linear_out = x @ w + b
        linear_out = torch.matmul(x, self.w) + self.b  # Shape: [Batch, 1] [cite: 79]

        # 2. 激活函数 (Activation Function) - 阶跃函数
        # 书中定义：大于0输出1，否则输出0 
        # 使用 torch.gt (greater than) 生成布尔掩码，然后转为 float
        y = torch.gt(linear_out, 0).float()            # Shape: [Batch, 1]
        
        return y

# --- 2. 准备测试数据 ---
def get_test_inputs():
    """
    生成所有可能的输入组合 (真值表输入)
    """
    # X shape: [4, 2] -> 4个样本 (Batch Size), 每个样本2个特征 (x1, x2)
    x = torch.tensor([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ], dtype=torch.float32)
    return x

# --- 3. 验证函数 ---
def verify_gate(gate_name: str, model: nn.Module, inputs: torch.Tensor):
    print(f"\n--- Testing {gate_name} Gate ---")
    
    # 1. 前向传播 (Forward Pass)
    # 输入 inputs: [4, 2] -> 输出 outputs: [4, 1]
    with torch.no_grad(): # 测试模式不需要计算梯度
        outputs = model(inputs)
    
    # 2. 格式化打印结果
    # 将输入和输出拼接以便展示: [4, 2] cat [4, 1] -> [4, 3]
    results = torch.cat([inputs, outputs], dim=1) 
    
    print(f"x1  x2 |  y (Output)")
    print(f"-------|-----------")
    for row in results:
        # row[0]=x1, row[1]=x2, row[2]=y
        print(f" {int(row[0])}   {int(row[1])} |  {int(row[2])}")

# --- 4. 主执行逻辑 ---
if __name__ == "__main__":
    inputs = get_test_inputs()

    # A. 测试 AND 门
    # 参数引用: w=[0.5, 0.5], b=-0.7 [cite: 42]
    and_model = BasicLogicGate(w=[0.5, 0.5], b=-0.7)
    verify_gate("AND", and_model, inputs)

    # B. 测试 NAND 门
    # 参数引用: w=[-0.5, -0.5], b=0.7 [cite: 50]
    nand_model = BasicLogicGate(w=[-0.5, -0.5], b=0.7)
    verify_gate("NAND", nand_model, inputs)

    # C. 测试 OR 门
    # 参数引用: w=[0.5, 0.5], b=-0.2 [cite: 122, 155]
    or_model = BasicLogicGate(w=[0.5, 0.5], b=-0.2)
    verify_gate("OR", or_model, inputs)