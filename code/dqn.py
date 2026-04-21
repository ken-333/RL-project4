import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):   #nn.Module：PyTorch 里所有神经网络的基类。自定义网络都要继承它
    """Standard DQN: 3 hidden layers of 128 units, ReLU activations."""
    # 输入维度是 2（状态坐标 x 和 y），输出维度是 4（四个动作的 Q 值），隐藏层维度是 128

    def __init__(self, input_dim=2, output_dim=4, hidden_dim=128):   # 初始化：搭建网络结构
        super().__init__() # 调用父类 nn.Module 的初始化方法，完成一些必要的设置
        self.layer1 = nn.Linear(input_dim, hidden_dim)     #nn.Linear(in, out) 是全连接层（fully connected layer）。每一层做一次变换  输出=W×输入+b  其中 W 和 b 是网络要学的参数，PyTorch 自动帮你创建。
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)            #这里加上self.，是因为这些层是 DQN 类的成员变量，需要在整个类中访问和使用。如果不加 self.，这些变量就只是 __init__ 方法中的局部变量，其他方法无法访问到它们。
        self.layer4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # 前向传播：描述数据怎么流过网络 参数 x 是输入数据。
        x = F.relu(self.layer1(x))  # F.relu 是 PyTorch 里 ReLU 激活函数的实现，作用是对输入进行非线性变换，输出 max(0, x)，即把负数变成 0，正数保持不变。这样可以让网络学到更复杂的函数关系。
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x) # 最后一层没有激活函数，因为我们需要输出 Q 值，可以是任意实数，不需要限制在某个范围内。


class DuelingDQN(nn.Module):  #网络先共享特征，再分叉成两条路径，最后合并成 Q 值。
    """
    Dueling DQN: shared feature extractor splits into value and advantage streams.
    Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]
    """

    def __init__(self, input_dim=2, output_dim=4, hidden_dim=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # 2 → 128
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 128 → 128
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        feat = self.feature(x)
        v = self.value_stream(feat)
        a = self.advantage_stream(feat)
        return v + a - a.mean(dim=-1, keepdim=True)
