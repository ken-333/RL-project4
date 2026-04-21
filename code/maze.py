import numpy as np

GRID_SIZE = 8
N_ACTIONS = 4  # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT

# (dx, dy) in maze coordinates: y increases downward
ACTION_DELTAS = {  # in (x, y) order, y-axis flipped for plotting 字典的结构：动作编号 → (dx, dy)
    0: (0, -1),   # UP
    1: (1,  0),   # RIGHT
    2: (0,  1),   # DOWN
    3: (-1, 0),   # LEFT
}

# Perpendicular actions for stochastic transitions
PERPENDICULAR = {
    0: [3, 1],  # UP  -> LEFT, RIGHT (UP的垂直方向是左右) 3: LEFT, 1: RIGHT
    1: [0, 2],  # RIGHT -> UP, DOWN
    2: [3, 1],  # DOWN -> LEFT, RIGHT
    3: [0, 2],  # LEFT -> UP, DOWN
}

P = 0.025  # noise probability

# Maze layout: (x=col, y=row), y=0 is top row
WALLS = {
    (3, 1),
    (2, 2), (3, 2), (4, 2), (5, 2), (7, 2),
    (3, 3),
    (5, 4),
    (0, 5), (1, 5), (2, 5), (5, 5),
    (5, 6),
    (5, 7),
}

YELLOW_CELLS = {(1, 0), (6, 2), (1, 4), (6, 4)}   # reward -5
RED_CELLS    = {(1, 2), (3, 5), (2, 6), (6, 6)}    # reward -10
GOAL_CELL    = (5, 0)                               # reward +100, terminal
START_CELL   = (1, 6)                               # blue cell


def _all_states():
    return [(x, y) for y in range(GRID_SIZE)
                   for x in range(GRID_SIZE) #相当于嵌套 for 循环  生成所有坐标点，排除墙壁坐标
                   if (x, y) not in WALLS]  

ALL_STATES = _all_states() # 生成所有非墙壁的坐标点列表，供训练时随机选择起始状态使用
NON_GOAL_STATES = [s for s in ALL_STATES if s != GOAL_CELL]


def is_valid(x, y):  # 判断坐标 (x, y) 是否在迷宫内且不是墙壁，如果是有效的坐标返回 True，否则返回 False
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and (x, y) not in WALLS


def _compute_reward(s_next, hit_wall):
    r = -1.0
    if hit_wall:
        r += -0.8
    else:
        if s_next in YELLOW_CELLS:
            r += -5.0
        if s_next in RED_CELLS:
            r += -10.0
        if s_next == GOAL_CELL:
            r += 100.0
    return r


def step(s, action):
    """Stochastic transition. Returns (s_next, reward, done)."""
    x, y = s
    rand = np.random.random()
    if rand < 1 - P:
        act = action
    elif rand < 1 - P / 2:
        act = PERPENDICULAR[action][0]
    else:
        act = PERPENDICULAR[action][1]

    dx, dy = ACTION_DELTAS[act]
    nx, ny = x + dx, y + dy

    if is_valid(nx, ny):
        s_next = (nx, ny) # 更新状态为新坐标 (nx, ny)，表示智能体成功移动到新位置
        hit_wall = False
    else:
        s_next = s # 如果新坐标无效（越界或撞墙），状态保持不变，智能体没有移动成功
        hit_wall = True

    reward = _compute_reward(s_next, hit_wall)
    done = (s_next == GOAL_CELL) # 如果智能体到达目标位置，done 标志为 True，表示这一轮训练结束；否则为 False，继续训练下一步
    return s_next, reward, done


def reset(random_start=True):  
    """Return a starting state. Training uses random start (per assignment)."""
    if random_start:
        return NON_GOAL_STATES[np.random.randint(len(NON_GOAL_STATES))] # 从非目标状态列表中随机选择一个状态作为起始状态返回
    return START_CELL


def state_to_array(s):  # 把状态 (x, y) 转换成一个归一化的 numpy 数组，用作神经网络的输入  神经网络（DQN）不能吃元组，它只认数值张量/数组
  #     坐标归一化：0~7 → 0.0~1.0
  # 格式转换：tuple → numpy 数组（float32）
    """Normalize (x,y) to [0,1]^2 for neural network input."""
    x, y = s
    return np.array([x / (GRID_SIZE - 1), y / (GRID_SIZE - 1)], dtype=np.float32)
