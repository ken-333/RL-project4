import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from dqn import DQN, DuelingDQN


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)  #deque = 双端队列（double-ended queue），是 Python 标准库 collections 模块提供的一种数据结构。它允许在两端高效地添加和删除元素。maxlen 参数指定了队列的最大长度，当队列达到这个长度时，新的元素会自动覆盖掉最旧的元素。这使得 deque 非常适合用作固定大小的缓冲区，例如在强化学习中存储经验回放（experience replay）数据。

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, float(done)))

    def sample(self, batch_size): #从 buffer 里随机抽 batch_size 条，组装成 PyTorch 张量。
        batch = random.sample(self.buf, batch_size) #random.sample(序列, k) 从序列里无放回随机抽 k 个元素。假设 batch_size=64，就会抽 64 条经验。
        s, a, r, s_next, done = zip(*batch)  # 使用 zip(*batch) 将批次中的每个元素分别解压成状态、动作、奖励、下一个状态和完成标志的元组
        return (
            torch.FloatTensor(np.array(s)), # 将状态列表转换为 NumPy 数组，再转换为 PyTorch 的 FloatTensor 张量，作为神经网络的输入
            torch.LongTensor(a), # 将动作列表转换为 PyTorch 的 LongTensor 张量，作为神经网络的输出标签
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(s_next)),
            torch.FloatTensor(done),
        )

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    """
    Supports three modes:
      'standard' — vanilla DQN (target uses target-net for both select & eval)
      'double'   — Double DQN (Q-net selects, target-net evaluates)
      'dueling'  — Dueling network architecture with standard DQN target
    """

    def __init__(self, mode='standard', lr=5e-4, gamma=0.98,
                 buffer_size=10000, batch_size=64,
                 n_update=4, eta=1e-3):
        self.mode = mode
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_update = n_update
        self.eta = eta

        NetClass = DuelingDQN if mode == 'dueling' else DQN
        self.q_net     = NetClass()  #创建一个 Q 网络实例，根据 mode 参数选择使用标准 DQN 还是 Dueling DQN。这个网络将用于预测给定状态下每个动作的 Q 值。
        self.target_net = NetClass()
        self.target_net.load_state_dict(self.q_net.state_dict()) # 将 q_net 的参数复制到 target_net 中，确保它们初始时是一样的。之后 q_net 会不断更新，而 target_net 会通过软更新慢慢跟上 q_net 的变化。
        self.target_net.eval() # 将 target_net 设置为评估模式，这样在前向传播时会禁用 dropout 和 batch normalization 等训练专用的层，确保 target_net 的输出稳定可靠。

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr) # 使用 Adam 优化器来更新 q_net 的参数，学习率为 lr。Adam 是一种自适应学习率优化算法，能够根据参数的稀疏程度和梯度的一阶矩估计自动调整学习率，通常在训练深度神经网络时表现良好。
        self.memory = ReplayBuffer(buffer_size)  # 创建一个经验回放缓冲区，容量为 buffer_size，用来存储智能体在环境中经历的状态转移（s, a, r, s_next, done）。经验回放可以打破数据之间的相关性，提高训练的稳定性和效率。
        self._step = 0   # counts total pushes; update fires every n_update steps

    # ------------------------------------------------------------------
    def select_action(self, state_arr, epsilon):
        """Epsilon-greedy action selection."""
        if np.random.random() < epsilon:  # 随机数小于 epsilon 时，选择一个随机动作（探索），否则选择当前 Q 网络预测的最佳动作（利用）。这样可以在训练初期更多地探索环境，在训练后期更多地利用学到的知识。
            return np.random.randint(4)
        with torch.no_grad(): # torch.no_grad() 是一个上下文管理器，用于禁用 PyTorch 的自动求导机制。在这个上下文中，所有的计算都不会被记录到计算图中，因此不会计算梯度。这对于在选择动作时只需要前向传播而不需要反向传播的情况非常有用，可以节省内存和计算资源。
            q = self.q_net(torch.FloatTensor(state_arr).unsqueeze(0))  # 将输入状态数组转换为 PyTorch 的 FloatTensor 张量，并使用 unsqueeze(0) 在第 0 维添加一个批次维度，使其形状变为 (1, input_dim)，适合神经网络的输入。然后将这个张量传入 q_net 中进行前向传播，得到一个形状为 (1, output_dim) 的 Q 值张量。
            return q.argmax().item()  # q.argmax() 返回 Q 值张量中最大值的索引，即最佳动作的索引。由于 q 的形状是 (1, output_dim)，argmax() 会返回一个形状为 (1,) 的张量，使用 item() 将其转换为一个 Python 整数，作为最终选择的动作返回。

    def push(self, s, a, r, s_next, done):
        self.memory.push(s, a, r, s_next, done)
        self._step += 1

    def update(self):
        """Update Q-network every n_update steps. Returns loss or None."""
        if self._step == 0 or self._step % self.n_update != 0:  # 每 n_update 步更新一次 Q 网络，如果当前步数不是 n_update 的倍数，直接返回 None，不进行更新。
            return None
        if len(self.memory) < self.batch_size:  # 如果经验回放缓冲区中的样本数量不足一个批次的大小，也不进行更新，直接返回 None。
            return None

        s, a, r, s_next, done = self.memory.sample(self.batch_size)   # 从经验回放缓冲区中随机抽取一个批次的样本，得到状态、动作、奖励、下一个状态和完成标志的张量。这些张量的形状分别是 (batch_size, input_dim)、(batch_size,)、(batch_size,)、(batch_size, input_dim) 和 (batch_size,)，可以直接用于神经网络的训练。

        # Current Q(s, a)
        q_pred = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)  # 通过 q_net 前向传播得到一个形状为 (batch_size, output_dim) 的 Q 值张量，然后使用 gather(1, a.unsqueeze(1)) 从每行中选择对应动作 a 的 Q 值，得到一个形状为 (batch_size, 1) 的张量，最后使用 squeeze(1) 将其变为 (batch_size,) 的一维张量，作为当前状态动作对的 Q 值预测。

        # Target z_i
        with torch.no_grad():
            if self.mode == 'double':
                # Double DQN: Q-net selects action, target-net evaluates
                best_a = self.q_net(s_next).argmax(dim=1, keepdim=True)
                q_next = self.target_net(s_next).gather(1, best_a).squeeze(1)
            else:
                q_next = self.target_net(s_next).max(dim=1)[0]  

            targets = r + self.gamma * q_next * (1.0 - done)

        loss = nn.MSELoss()(q_pred, targets) # 计算当前 Q 值预测和目标 Q 值之间的均方误差损失，作为训练的目标函数。
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()  # 更新 q_net 的参数(w, b)，使其更接近目标 Q 值。优化器会根据计算得到的损失值自动调整 q_net 中的权重和偏置参数，以最小化损失函数，从而提高智能体的性能。

        self._soft_update()
        return loss.item()  # 返回当前批次的损失值，供训练过程中的监控和分析使用。

    def _soft_update(self): # 软更新：每次更新后，将 q_net 的参数以一定比例（eta）混合到 target_net 中，使 target_net 慢慢跟上 q_net 的变化，保持训练的稳定性。
        for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
            tp.data.copy_(self.eta * p.data + (1.0 - self.eta) * tp.data)
