# EECE 5614 — Project 4: DQN Variants on Stochastic Maze

> **Course:** EECE 5614 Reinforcement Learning and Decision Making Under Uncertainty, Northeastern University, Spring 2026
> **Author:** Shoukang Yu

---

## What This Project Does

This project trains a reinforcement learning agent to navigate an **8×8 stochastic maze** from a random start position to a fixed goal using three Deep Q-Network variants:

| Algorithm | Key Idea |
|-----------|----------|
| **Standard DQN** | Target network evaluates both action selection and Q-value |
| **Double DQN** | Online network selects action; target network evaluates it — reduces overestimation bias |
| **Dueling DQN** | Splits Q-value into state value V(s) and advantage A(s,a) — faster convergence |

---

## Maze Layout

![Maze Layout](code/result/std_policy.png)

Green = goal (+100) · Blue = start · Yellow = −5 penalty · Red = −10 penalty · Black = wall

**Reward structure:**

| Event | Reward |
|-------|--------|
| Every step | −1 |
| Hit wall (stay) | −1.8 total |
| Land on yellow cell | −6 total |
| Land on red cell | −11 total |
| Reach goal | +99 total |

**Transition noise:** `P = 0.025` — with probability 0.975 the agent moves as intended; with probability 0.0125 each it slips to either perpendicular direction.

---

## Results

### Standard DQN — Training Curves
![STD Training Curves](code/result/std_curves.png)

### Standard DQN — Learned Policy
![STD Policy](code/result/std_policy.png)

### Standard DQN — State Values
![STD State Values](code/result/std_values.png)

### Standard DQN — Path from Start to Goal
![STD Path](code/result/std_path.png)

---

### Q7: Effect of Learning Rate
![Learning Rate Comparison](code/result/q7_lr_comparison.png)

The baseline α = 5×10⁻⁴ gives the best balance of convergence speed and stability. A 10× larger rate converges faster initially but becomes unstable after episode 2000; a 10× smaller rate has not converged after 3000 episodes.

---

### Double DQN — Training Curves
![Double Training Curves](code/result/double_curves.png)

### Double DQN — Learned Policy
![Double Policy](code/result/double_policy.png)

### Double DQN — State Values
![Double State Values](code/result/double_values.png)

### Double DQN — Path from Start to Goal
![Double Path](code/result/double_path.png)

---

### Dueling DQN — Training Curves
![Dueling Training Curves](code/result/dueling_curves.png)

### Dueling DQN — Learned Policy
![Dueling Policy](code/result/dueling_policy.png)

### Dueling DQN — State Values
![Dueling State Values](code/result/dueling_values.png)

### Dueling DQN — Path from Start to Goal
![Dueling Path](code/result/dueling_path.png)

---

### Three-Way Comparison

| | Avg Reward |
|-|------------|
| ![Reward Comparison](code/result/comparison_reward.png) | |

![Length Comparison](code/result/comparison_length.png)

**Key findings:**
- All three methods successfully solve the maze and converge to similar final performance (~85–90 avg reward)
- **Dueling DQN** converges fastest (reaches 50 avg reward by episode ~400)
- **Double DQN** produces the most accurate (least overestimated) Q-values
- **Standard DQN** is the simplest baseline

---

## Network Architectures

### DQN / Double DQN
```
Input(2) → Linear(128) → ReLU
         → Linear(128) → ReLU
         → Linear(128) → ReLU
         → Linear(4)        ← Q-values for UP / RIGHT / DOWN / LEFT
```

### Dueling DQN
```
Input(2) → Linear(128) → ReLU → Linear(128) → ReLU    [shared feature trunk]
                ├── Linear(128) → ReLU → Linear(1)     [Value stream  V(s)]
                └── Linear(128) → ReLU → Linear(4)     [Advantage stream A(s,a)]
                Q(s,a) = V(s) + A(s,a) − mean_a'[A(s,a')]
```

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Episodes | 3000 |
| Max steps / episode | 50 |
| Replay buffer | 10,000 |
| Minibatch size | 64 |
| Discount γ | 0.98 |
| Learning rate α | 5×10⁻⁴ |
| Q-net update interval | every 4 steps |
| Soft update rate η | 1×10⁻³ |
| ε schedule | max(0.1, 0.999^episode) |

---

## File Structure

```
├── maze.py        # Maze environment: step(), reset(), reward, walls
├── dqn.py         # Neural networks: DQN, DuelingDQN
├── agent.py       # ReplayBuffer + DQNAgent (standard / double / dueling)
├── train.py       # Training loop + epsilon schedule
├── visualize.py   # Plotting: policy arrows, value heatmap, path, curves
├── main.py        # Runs all experiments, saves all figures
└── code/result/        # All output figures (15 PNG files)
```

## How to Run

```bash
pip install torch numpy matplotlib
python main.py
```