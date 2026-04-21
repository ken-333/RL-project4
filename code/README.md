# Project 4 — DQN on Maze (EECE 5614, Spring 2026)

## Overview

Use Deep Q-Network (DQN) and its variants to train an agent to navigate an 8×8 stochastic maze from a random start state to the goal (green cell).

## Maze Layout

```
x:  0    1    2    3    4    5    6    7
y=0:[   ][Yel][   ][   ][   ][Grn][   ][   ]   ← Goal at (5,0)
y=1:[   ][   ][   ][###][   ][   ][   ][   ]
y=2:[   ][Red][###][###][###][###][Yel][###]
y=3:[   ][   ][   ][###][   ][   ][   ][   ]
y=4:[   ][Yel][   ][   ][   ][###][Yel][   ]
y=5:[###][###][Red][   ][###][###][   ][   ]
y=6:[   ][Blu][Red][   ][   ][###][Red][   ]   ← Start at (1,6)
y=7:[   ][   ][   ][   ][   ][###][   ][   ]
```

`###`=wall, `Grn`=goal(+100), `Blu`=start, `Yel`=yellow(-5), `Red`=red(-10)

**State representation:** `s = [x/7, y/7]` — normalized to `[0,1]²`

## Reward Structure

| Event | Reward |
|-------|--------|
| Every action taken | −1 |
| Hit a wall (stay in place) | −0.8 |
| Land on yellow cell | −5 |
| Land on red cell | −10 |
| Reach goal | +100 |

Rewards are **additive** (e.g., landing on red = −1 − 10 = −11 total).

## Transition Probability

With `P = 0.025`:
- Probability `1−P = 0.975`: move in intended direction
- Probability `P/2 = 0.0125`: slide to each perpendicular direction

If the resulting cell is a wall, the agent stays in place (and gets −0.8).

---

## File Structure

```
project4/
├── maze.py       # Maze environment: step(), reset(), state_to_array()
├── dqn.py        # Neural network architectures: DQN, DuelingDQN
├── agent.py      # ReplayBuffer + DQNAgent (standard / double / dueling)
├── train.py      # Training loop, epsilon schedule, moving average
├── visualize.py  # All plotting functions (policy, values, path, curves)
└── main.py       # Runs all experiments, saves all figures
```

---

## Algorithms

### Standard DQN
Target:
```
z_i = r_i + γ · max_a Q^{w-}(s'_i, a) · (1 − done_i)
```

### Double DQN (Q7)
Decouples action selection and evaluation to reduce overestimation:
```
z_i = r_i + γ · Q^{w-}(s'_i, argmax_{a'} Q^w(s'_i, a')) · (1 − done_i)
```

### Dueling DQN (Q8)
Network outputs Value stream V(s) and Advantage stream A(s,a), combined as:
```
Q(s,a) = V(s) + A(s,a) − mean_{a'} A(s,a')
```

---

## Hyperparameters

| Parameter | Symbol | Value | Range (assignment) |
|-----------|--------|-------|--------------------|
| Episodes | N_epi | 3000 | — |
| Max steps/episode | T_epi | 50 | ~50 |
| Replay buffer size | \|D\| | 10000 | ~10^4 |
| Minibatch size | N_batch | 64 | 64 |
| Discount factor | γ | 0.98 | 0.95 < γ < 0.995 |
| Learning rate | α | 5e-4 | 10^−4 < α < 10^−1 |
| Q-net update interval | N_QU | 4 | — |
| Soft update rate | η | 1e-3 | 10^−4 < η < 10^−1 |
| Epsilon decay | decay | 0.999 | close to 1 |

**Epsilon schedule:** `ε = max(0.1, 0.999^episode)`

---

## Network Architecture

### DQN / Double DQN
```
Input(2) → Linear(128) → ReLU
         → Linear(128) → ReLU
         → Linear(128) → ReLU
         → Linear(4)              ← Q-values for 4 actions
```

### Dueling DQN
```
Input(2) → Linear(128) → ReLU → Linear(128) → ReLU   [shared feature]
                ├── Linear(128) → ReLU → Linear(1)    [Value stream V(s)]
                └── Linear(128) → ReLU → Linear(4)    [Advantage stream A(s,a)]
                Combined: Q = V + A − mean(A)
```

---

## How to Run

### Install dependencies
```bash
pip install torch numpy matplotlib
```

### Run all experiments
```bash
python main.py
```

This trains Standard DQN, Double DQN, and Dueling DQN sequentially and saves all figures.

---

## Output Files

| File | Content | Question |
|------|---------|---------|
| `std_curves.png` | Avg Reward, Avg Loss, Avg Length | Q2, Q6 |
| `std_policy.png` | Arrow policy on maze | Q3 |
| `std_values.png` | State value heatmap | Q4 |
| `std_path.png` | Path from start to goal | Q5 |
| `q7_lr_comparison.png` | 3 learning rates compared | Q7 |
| `double_curves.png` | Double DQN training curves | Q8-2, Q8-6 |
| `double_policy.png` | Double DQN policy | Q8-3 |
| `double_values.png` | Double DQN state values | Q8-4 |
| `double_path.png` | Double DQN path | Q8-5 |
| `dueling_curves.png` | Dueling DQN training curves | Q9-2, Q9-6 |
| `dueling_policy.png` | Dueling DQN policy | Q9-3 |
| `dueling_values.png` | Dueling DQN state values | Q9-4 |
| `dueling_path.png` | Dueling DQN path | Q9-5 |
| `comparison_reward.png` | All 3 methods: Avg Reward | Q8, Q9 |
| `comparison_length.png` | All 3 methods: Avg Length | Q8, Q9 |
