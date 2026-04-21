import numpy as np
from maze import reset, step, state_to_array
from agent import DQNAgent


def epsilon_at(episode, decay=0.999):
    """ε = max(0.1, decay^episode), episode starts from 1."""
    return max(0.1, decay ** episode)


def moving_average(data, window=25):  
    """Avg_X[k] = mean of last min(25, k) episodes (1-indexed k)."""
    avgs = []
    for k in range(len(data)):
        m = min(window, k + 1)
        avgs.append(float(np.mean(data[k - m + 1: k + 1])))
    return avgs


def train(mode='standard', n_episodes=3000, t_epi=50,
          lr=5e-4, gamma=0.98, buffer_size=10000,
          batch_size=64, n_update=4, eta=1e-3, decay=0.999,
          verbose=True):
    """
    Run DQN training.

    Returns:
        agent         — trained DQNAgent
        epi_rewards   — list of per-episode total rewards
        epi_losses    — list of per-episode total losses (sum, not mean)
        epi_lengths   — list of per-episode step counts
    """
    agent = DQNAgent(mode=mode, lr=lr, gamma=gamma,
                     buffer_size=buffer_size, batch_size=batch_size,
                     n_update=n_update, eta=eta)

    epi_rewards = []
    epi_losses  = []
    epi_lengths = []

    for ep in range(1, n_episodes + 1):     # 外层：3000 个 episode
        eps = epsilon_at(ep, decay)
        s   = reset(random_start=True)

        ep_reward = 0.0
        ep_loss   = 0.0
        ep_steps  = t_epi   # default: timed out

        for t in range(1, t_epi + 1):  # 内层：最多 50 步
            s_arr = state_to_array(s) 
            a = agent.select_action(s_arr, eps)
            s_next, r, done = step(s, a)
            s_next_arr = state_to_array(s_next)

            agent.push(s_arr, a, r, s_next_arr, done)
            loss = agent.update()

            ep_reward += r # 累加 reward
            if loss is not None:
                ep_loss += loss   # 累加 loss
            s = s_next

            if done:
                ep_steps = t   # reached goal early  # 记录实际步数
                break               # 到 GOAL 就结束

        # 记录本 episode 的统计数据：总 reward、总 loss、步数  
        epi_rewards.append(ep_reward)
        epi_losses.append(ep_loss)
        epi_lengths.append(ep_steps)
        
        # 定期打印进度
        if verbose and ep % 200 == 0:
            avg_r = np.mean(epi_rewards[-25:])
            print(f"[{mode}] ep {ep:4d}/{n_episodes} | "
                  f"avg_reward={avg_r:7.2f} | eps={eps:.4f}")

    return agent, epi_rewards, epi_losses, epi_lengths   # 返回训练好的 agent 和每 episode 的 reward、loss、length 列表
