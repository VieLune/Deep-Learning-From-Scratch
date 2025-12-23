import torch
import torch.nn as nn

# --- 1. 基础算子定义 (LogicGate) ---
class LogicGate(nn.Module):
    def __init__(self, w: list, b: float):
        super().__init__()
        self.register_buffer('w', torch.tensor(w, dtype=torch.float32).unsqueeze(1))
        self.register_buffer('b', torch.tensor(b, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = torch.matmul(x, self.w) + self.b
        return torch.gt(linear_out, 0).float()

# --- 2. 核心：XOR 多层感知机定义 ---
class XORModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 实例化基础门电路
        self.nand_gate = LogicGate(w=[-0.5, -0.5], b=0.7) # NAND
        self.or_gate   = LogicGate(w=[ 0.5,  0.5], b=-0.2) # OR
        self.and_gate  = LogicGate(w=[ 0.5,  0.5], b=-0.7) # AND

    def forward(self, x: torch.Tensor, return_hidden: bool=False):
        """
        Args:
            x: 输入 [Batch, 2]
            return_hidden: 是否返回中间层结果用于调试
        """
        # --- 第 1 层 (Hidden Layer) ---
        s1 = self.nand_gate(x) # Shape: [Batch, 1]
        s2 = self.or_gate(x)   # Shape: [Batch, 1]
        
        # 拼接 s1, s2 形成第 2 层的输入
        layer1_out = torch.cat([s1, s2], dim=1) # Shape: [Batch, 2]

        # --- 第 2 层 (Output Layer) ---
        y = self.and_gate(layer1_out) # Shape: [Batch, 1]
        
        if return_hidden:
            return y, s1, s2
        return y

# --- 3. 测试与验证代码 ---
def test_xor_logic():
    print("=== Testing XOR Module (2-Layer Perceptron) ===")
    
    # 构造标准 XOR 真值表输入
    inputs = torch.tensor([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ], dtype=torch.float32)

    # 实例化模型
    model = XORModule()

    # 推理模式
    with torch.no_grad():
        # 获取最终输出 y，以及中间变量 s1 (NAND), s2 (OR)
        y, s1, s2 = model(inputs, return_hidden=True)

    # --- 格式化打印 ---
    print(f"Input (x)     |  Hidden (s)     | Output (y)")
    print(f"x1  x2        | s1(NAND) s2(OR) | XOR Result")
    print(f"--------------|-----------------|-----------")
    
    # 遍历每一行数据进行展示
    for i in range(len(inputs)):
        x_val = inputs[i]
        s1_val = s1[i].item()
        s2_val = s2[i].item()
        y_val = y[i].item()
        
        # 可视化输出
        print(f" {int(x_val[0])}   {int(x_val[1])}        |    {int(s1_val)}       {int(s2_val)}    |     {int(y_val)}")

# 运行测试
if __name__ == "__main__":
    test_xor_logic()