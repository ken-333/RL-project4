import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import torch

from maze import (GRID_SIZE, WALLS, YELLOW_CELLS, RED_CELLS,
                  GOAL_CELL, START_CELL, ALL_STATES,
                  state_to_array, step)
from train import moving_average

# Arrow offsets in plot coordinates (y-axis flipped: maze y=0 → plot top)
_ARROW = {0: (0, 0.3), 1: (0.3, 0), 2: (0, -0.3), 3: (-0.3, 0)}  #动作编号 → (dx, dy)  0: UP, 1: RIGHT, 2: DOWN, 3: LEFT


def _cell_color(x, y): #根据坐标 (x, y) 返回对应的颜色字符串，用于绘制迷宫中的不同类型的格子。
    if (x, y) in WALLS:      return 'black' #墙壁坐标返回黑色
    if (x, y) == GOAL_CELL:  return 'limegreen'
    if (x, y) == START_CELL: return 'dodgerblue'
    if (x, y) in YELLOW_CELLS: return 'yellow'
    if (x, y) in RED_CELLS:    return 'red'
    return 'white'


def _draw_grid(ax): #在给定的 Matplotlib 轴对象 ax 上绘制迷宫的网格和格子颜色。
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = patches.Rectangle(   #patches.Rectangle(左下角, 宽, 高) 画一个矩形，这里的左下角坐标是 (x, GRID_SIZE - 1 - y)，宽和高都是 1。坐标 (x, y) 是迷宫坐标系中的位置，但由于 Matplotlib 的坐标系中 y 轴是向上的，所以需要将 y 坐标转换为 GRID_SIZE - 1 - y 来正确显示。
                (x, GRID_SIZE - 1 - y), 1, 1,
                linewidth=0.5, edgecolor='gray', #矩形边界的线宽和颜色
                facecolor=_cell_color(x, y))
            ax.add_patch(rect)
    ax.set_xlim(0, GRID_SIZE)  #设置 x 轴范围为 [0, GRID_SIZE]
    ax.set_ylim(0, GRID_SIZE)  #设置 y 轴范围为 [0, GRID_SIZE]
    ax.set_aspect('equal')   #设置坐标轴的长宽比为相等，这样每个格子看起来是正方形
    ax.axis('off')     #关闭坐标轴显示


def _q_values(agent, s):
    t = torch.FloatTensor(state_to_array(s)).unsqueeze(0)
    with torch.no_grad():
        return agent.q_net(t).detach().numpy().flatten()  #将状态 s 转换为输入张量 t，传入 agent 的 q_net 中进行前向传播，得到 Q 值张量。使用 detach() 将其从计算图中分离出来，转换为 NumPy 数组，并使用 flatten() 将其展平为一维数组，返回给调用者。这个函数用于获取智能体在状态 s 下的所有动作的 Q 值，以便在可视化中显示最佳动作或状态值等信息。


# ── Q1-style: policy arrows ──────────────────────────────────────────
def plot_policy(agent, title='Policy'):
    fig, ax = plt.subplots(figsize=(7, 7))   #创建一个新的 Matplotlib 图形和轴对象，图形大小为 7x7 英寸。
    _draw_grid(ax)  #调用 _draw_grid 函数在轴对象 ax 上绘制迷宫的网格和格子颜色。

    for (x, y) in ALL_STATES:   #ALL_STATES 是所有非墙状态的集合（51 个）。对每个格子做一次
        if (x, y) == GOAL_CELL:  #目标格子不画箭头，直接跳过
            continue
        best_a = int(np.argmax(_q_values(agent, (x, y))))
        dx, dy = _ARROW[best_a]
        cx = x + 0.5 #箭头起点的 x 坐标，位于格子中心，所以加 0.5
        cy = GRID_SIZE - 1 - y + 0.5 #箭头起点的 y 坐标，位于格子中心，所以加 0.5
        ax.annotate('', xy=(cx + dx, cy + dy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.set_title(title)
    plt.tight_layout()
    return fig


# ── Q4-style: state value heatmap ───────────────────────────────────
def plot_state_values(agent, title='State Values'):
    vals = {s: float(np.max(_q_values(agent, s))) for s in ALL_STATES}  #对于每个非墙状态 s，计算其对应的状态值，即在状态 s 下所有动作的 Q 值中的最大值。将结果存储在一个字典 vals 中，键是状态 s，值是对应的状态值。这些状态值将用于后续的热力图可视化。
    vmin, vmax = min(vals.values()), max(vals.values())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.RdYlGn   #使用 Matplotlib 的 RdYlGn 颜色映射（红-黄-绿）来表示状态值的大小，vmin 和 vmax 分别是状态值的最小值和最大值，用于归一化状态值到 [0, 1] 的范围，以便正确映射到颜色上。

    fig, ax = plt.subplots(figsize=(7, 7))
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            py = GRID_SIZE - 1 - y
            if (x, y) in WALLS:
                color = 'black'
                text  = None
            else:
                v     = vals[(x, y)]  #获取状态 (x, y) 的状态值 v
                color = cmap(norm(v))
                text  = f'{v:.1f}'   #将状态值 v 格式化为字符串，保留一位小数，用于在格子中显示状态值的数值
            rect = patches.Rectangle((x, py), 1, 1,
                                      linewidth=0.5, edgecolor='gray',
                                      facecolor=color)  #根据状态值 v 的大小，使用颜色映射 cmap 和归一化器 norm 来确定格子的填充颜色 color。对于墙壁格子，直接使用黑色。然后创建一个矩形补丁来表示格子，位置为 (x, py)，宽和高为 1，边界线宽为 0.5，边界颜色为灰色，填充颜色为 color。
            ax.add_patch(rect)
            if text:
                ax.text(x + 0.5, py + 0.5, text,  #在格子中心添加文本，显示状态值的数值
                        ha='center', va='center', fontsize=6.5)

    ax.set_xlim(0, GRID_SIZE)  #设置 x 轴范围为 [0, GRID_SIZE]
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 fraction=0.035, pad=0.02)  #在图形的轴对象 ax 上添加一个颜色条，使用与热力图相同的归一化器 norm 和颜色映射 cmap 来显示状态值对应的颜色范围。fraction 参数控制颜色条占图形的比例，pad 参数控制颜色条与图形之间的距离。
    plt.tight_layout()
    return fig


# ── Q5-style: path on maze ───────────────────────────────────────────
def plot_path(agent, title='Path from Start', max_steps=200):
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_grid(ax)

    s    = START_CELL
    path = [s]
    done = False
    for _ in range(max_steps):
        a        = int(np.argmax(_q_values(agent, s)))  #在当前状态 s 下，选择 Q 值最大的动作 a 作为智能体的行动决策。
        s_next, _, done = step(s, a)
        path.append(s_next)
        s = s_next
        if done:  #如果智能体达到了目标状态（GOAL_CELL），done 将被设置为 True，此时跳出循环，停止继续执行动作。
            break

    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        if (x1, y1) == (x2, y2):   # wall hit: skip zero-length arrow
            continue
        ax.annotate('',
                    xy=(x2 + 0.5, GRID_SIZE - 1 - y2 + 0.5),  #xy=(x2 + 0.5, GRID_SIZE - 1 - y2 + 0.5) 其中 x2 + 0.5 和 GRID_SIZE - 1 - y2 + 0.5 分别是箭头终点的 x 和 y 坐标，位于格子中心。由于 Matplotlib 的坐标系中 y 轴是向上的，所以需要将 y 坐标转换为 GRID_SIZE - 1 - y 来正确显示。
                    xytext=(x1 + 0.5, GRID_SIZE - 1 - y1 + 0.5),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=2))

    status = 'reached goal!' if done else 'did NOT reach goal'
    ax.set_title(f'{title}  ({status})')
    plt.tight_layout()
    return fig


# ── Training curves (reward, loss, length) ───────────────────────────
def plot_curves(epi_rewards, epi_losses, epi_lengths, title_prefix=''):
    avg_r   = moving_average(epi_rewards)
    avg_l   = moving_average(epi_losses)
    avg_len = moving_average(epi_lengths)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(avg_r);          axes[0].set_title(f'{title_prefix} Avg Reward')
    axes[1].plot(avg_l,  'C1');   axes[1].set_title(f'{title_prefix} Avg Loss')
    axes[2].plot(avg_len,'C2');   axes[2].set_title(f'{title_prefix} Avg Length')
    for ax in axes:
        ax.set_xlabel('Episode')
    axes[0].set_ylabel('Reward');  axes[1].set_ylabel('Loss')
    axes[2].set_ylabel('Steps')
    plt.tight_layout()
    return fig, avg_r, avg_l, avg_len


# ── Multi-curve comparison on a single figure ────────────────────────
def plot_compare(curves_dict, ylabel='Avg Reward', title='Comparison'):
    """curves_dict: {label: list_of_values}"""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, vals in curves_dict.items():
        ax.plot(vals, label=label)
    ax.set_xlabel('Episode')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig
