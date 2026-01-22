import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口（可选）

class ConvGRUCell(nn.Module):  # 定义 ConvGRU 的单步 cell
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):  # 初始化
        super().__init__()  # 调父类初始化
        self.in_channels = in_channels  # 保存输入通道数
        self.hidden_channels = hidden_channels  # 保存隐状态通道数
        padding = kernel_size // 2  # same padding 让 H/W 不变

        self.conv_gates = nn.Conv2d(  # 门控卷积（同时算 z 和 r）
            in_channels + hidden_channels,  # 输入：concat(x, h_prev)
            2 * hidden_channels,  # 输出：z 和 r 两个门
            kernel_size=kernel_size,  # 卷积核
            padding=padding  # padding
        )

        self.conv_candidate = nn.Conv2d(  # 候选隐状态卷积
            in_channels + hidden_channels,  # 输入：concat(x, r*h_prev)
            hidden_channels,  # 输出：h_tilde
            kernel_size=kernel_size,  # 卷积核
            padding=padding  # padding
        )

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:  # 单步前向
        xh = torch.cat([x_t, h_prev], dim=1)  # 拼接输入和旧隐状态
        gates = self.conv_gates(xh)  # 计算门控输出
        z_gate, r_gate = torch.split(gates, self.hidden_channels, dim=1)  # 切分出 z/r
        z = torch.sigmoid(z_gate)  # update gate
        r = torch.sigmoid(r_gate)  # reset gate

        rh = r * h_prev  # r ⊙ h_prev
        x_rh = torch.cat([x_t, rh], dim=1)  # 拼接 x 和 r*h_prev
        h_tilde = torch.tanh(self.conv_candidate(x_rh))  # 候选隐状态

        h_new = (1 - z) * h_prev + z * h_tilde  # GRU 更新
        return h_new  # 返回新隐状态


class ConvStem(nn.Module):  # 普通卷积前端（类似 ResNet stem 的简化版）
    def __init__(self, in_channels: int = 3, base_channels: int = 64):  # 初始化
        super().__init__()  # 调父类初始化

        self.conv1 = nn.Conv2d(  # 第一层卷积
            in_channels,  # 输入通道 3
            base_channels,  # 输出通道 64
            kernel_size=7,  # 7x7 卷积
            stride=2,  # 下采样到 112x112
            padding=3,  # padding 保尺寸合理
            bias=False  # 配合 BN 通常不需要 bias
        )
        self.bn1 = nn.BatchNorm2d(base_channels)  # BN
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 再下采样到 56x56

        self.conv2 = nn.Conv2d(  # 再接一层 3x3 卷积增强表征
            base_channels,  # 输入 64
            base_channels,  # 输出 64
            kernel_size=3,  # 3x3
            stride=1,  # 不下采样
            padding=1,  # same padding
            bias=False  # 不要 bias
        )
        self.bn2 = nn.BatchNorm2d(base_channels)  # BN

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向
        x = self.conv1(x)  # 7x7 卷积 + stride2
        x = self.bn1(x)  # BN
        x = self.relu(x)  # ReLU
        x = self.pool(x)  # MaxPool -> 56x56
        x = self.conv2(x)  # 3x3 卷积增强
        x = self.bn2(x)  # BN
        x = self.relu(x)  # ReLU
        return x  # 返回 stem 特征 (B, 64, 56, 56)


class TwoLayerConvGRUNet(nn.Module):  # 整体模型：stem + 两层 ConvGRU + head
    def __init__(self, in_channels: int = 3, stem_channels: int = 64, h1_channels: int = 96, h2_channels: int = 128, num_classes: int = 8):
        super().__init__()  # 调父类初始化

        self.stem = ConvStem(in_channels=in_channels, base_channels=stem_channels)  # 普通卷积前端

        self.pre1 = nn.Conv2d(  # 把 stem 的输出通道映射到 RNN1 的输入通道
            stem_channels,  # 输入 64
            h1_channels,  # 输出到 96
            kernel_size=1,  # 1x1 变换通道
            stride=1,  # 不改变分辨率
            padding=0,  # 无 padding
            bias=False  # 不用 bias
        )
        self.bn_pre1 = nn.BatchNorm2d(h1_channels)  # BN
        self.rnn1 = ConvGRUCell(in_channels=h1_channels, hidden_channels=h1_channels, kernel_size=3)  # 第一层 ConvGRU（56x56）

        self.down12 = nn.Conv2d(  # 从 56x56 下采样到 28x28，并变换通道到 h2_channels
            h1_channels,  # 输入 96
            h2_channels,  # 输出 128
            kernel_size=3,  # 3x3
            stride=2,  # stride2 下采样
            padding=1,  # same padding
            bias=False  # 不用 bias
        )
        self.bn_down12 = nn.BatchNorm2d(h2_channels)  # BN
        self.rnn2 = ConvGRUCell(in_channels=h2_channels, hidden_channels=h2_channels, kernel_size=3)  # 第二层 ConvGRU（28x28）

        self.head_color = nn.Linear(h2_channels, num_classes)
        self.head_count = nn.Linear(h2_channels, 11 )
        self.head_per_color = nn.Linear(h2_channels, 8 * 11)

    def forward(self, x: torch.Tensor, steps: int = 3):
        # x: (B, 3, 224, 224)  # 输入图像
        # steps: 迭代次数（模拟循环推理）  # 比如 3、5、8
        # return_all: 是否返回每一步的 h1/h2 轨迹（用于可视化/调试）

        B, C, H, W = x.shape  # 解析输入维度

        feat = self.stem(x)  # stem 提取特征 -> (B, 64, 56, 56)

        x1 = self.pre1(feat)  # 通道映射 -> (B, 96, 56, 56)
        x1 = self.bn_pre1(x1)  # BN
        x1 = F.relu(x1, inplace=True)  # ReLU

        h1 = torch.zeros(B, x1.shape[1], x1.shape[2], x1.shape[3], device=x.device, dtype=x.dtype)  # 初始化 h1 为 0
        h2 = torch.zeros(B, 128, x1.shape[2] // 2, x1.shape[3] // 2, device=x.device, dtype=x.dtype)  # 初始化 h2 为 0（28x28）

        logits_colors = []  # 保存每一步color
        logits_counts = []
        logits_per_colors = []

        for t in range(steps):  # 迭代 steps 次（输入不变，状态更新）
            h1 = self.rnn1(x1, h1)  # 第一层 ConvGRU 更新（56x56）

            x2 = self.down12(h1)  # 下采样 + 通道变换 -> (B, 128, 28, 28)
            x2 = self.bn_down12(x2)  # BN
            x2 = F.relu(x2, inplace=True)  # ReLU

            h2 = self.rnn2(x2, h2)  # 第二层 ConvGRU 更新（28x28）

            pooled = F.adaptive_avg_pool2d(h2, output_size=1)  # GAP -> (B, 128, 1, 1)
            pooled = pooled.view(B, -1)  # 展平 -> (B, 128)
            logits_color = self.head_color(pooled)  # 当前步 logits -> (B, 8)
            logits_count = self.head_count(pooled)
            logits_per_color = self.head_per_color(pooled)
            logits_per_color = logits_per_color.view(B, 8, 11) 

            logits_colors.append(logits_color)
            logits_counts.append(logits_count)
            logits_per_colors.append(logits_per_color)

        logits_colors = torch.stack(logits_colors, dim=1)  # 堆叠 -> (B, steps, 8)
        logits_counts = torch.stack(logits_counts, dim=1)  # 堆叠 -> (B, steps, 11)
        logits_per_colors = torch.stack(logits_per_colors, dim=1)  # 堆叠 -> (B, steps, 8, 11)            

        return logits_colors, logits_counts, logits_per_colors  # 返回每一步 logits
