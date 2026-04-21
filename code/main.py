"""
Project 4 — DQN / Double DQN / Dueling DQN on Maze
Runs all experiments and saves figures to the project directory.
"""
import os
import matplotlib.pyplot as plt
from train import train, moving_average
from visualize import plot_curves, plot_policy, plot_state_values, plot_path, plot_compare

# ── Hyperparameters ─────────────────────────────────────────────────
N_EPISODES   = 3000
T_EPI        = 50
BUFFER_SIZE  = 10000
BATCH_SIZE   = 64
GAMMA        = 0.98
LR           = 5e-4
N_UPDATE     = 4
ETA          = 1e-3
DECAY        = 0.999   # epsilon decay rate per episode

OUT_DIR = os.path.dirname(os.path.abspath(__file__)) 


def save(fig, name): #保存图形到指定路径，并关闭图形以释放内存
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {name}')


def run_all_plots(agent, rewards, losses, lengths, tag): #
    """Generate and save the 5 required plots for questions 2-6."""
    fig, avg_r, avg_l, avg_len = plot_curves(rewards, losses, lengths,
                                             title_prefix=tag.upper())
    save(fig, f'{tag}_curves.png')

    save(plot_policy(agent,       title=f'{tag.upper()} — Policy'),
         f'{tag}_policy.png')
    save(plot_state_values(agent, title=f'{tag.upper()} — State Values'),
         f'{tag}_values.png')
    save(plot_path(agent,         title=f'{tag.upper()} — Path'),
         f'{tag}_path.png')

    return avg_r, avg_l, avg_len


# ════════════════════════════════════════════════════════════════════
def main():

    # ── Q1-2: Standard DQN ──────────────────────────────────────────
    print('\n=== Standard DQN ===')
    agent_std, r_std, l_std, len_std = train(
        mode='standard', n_episodes=N_EPISODES, t_epi=T_EPI,
        lr=LR, gamma=GAMMA, buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE, n_update=N_UPDATE, eta=ETA, decay=DECAY)

    avg_r_std, avg_l_std, avg_len_std = run_all_plots(
        agent_std, r_std, l_std, len_std, tag='std')

    # ── Q7: Learning rate comparison ────────────────────────────────
    print('\n=== Q7: LR comparison ===')
    _, r_slow, _, _ = train(
        mode='standard', n_episodes=N_EPISODES, t_epi=T_EPI,
        lr=LR / 10, gamma=GAMMA, buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE, n_update=N_UPDATE, eta=ETA, decay=DECAY)

    _, r_fast, _, _ = train(
        mode='standard', n_episodes=N_EPISODES, t_epi=T_EPI,
        lr=LR * 10, gamma=GAMMA, buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE, n_update=N_UPDATE, eta=ETA, decay=DECAY)

    save(
        plot_compare({
            f'α/10 = {LR/10:.1e}': moving_average(r_slow),
            f'α   = {LR:.1e}':     avg_r_std,
            f'α×10 = {LR*10:.1e}': moving_average(r_fast),
        }, ylabel='Avg Reward', title='Q7 — Learning Rate Comparison'),
        'q7_lr_comparison.png')

    # ── Q8: Double DQN ──────────────────────────────────────────────
    print('\n=== Double DQN ===')
    agent_dbl, r_dbl, l_dbl, len_dbl = train(
        mode='double', n_episodes=N_EPISODES, t_epi=T_EPI,
        lr=LR, gamma=GAMMA, buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE, n_update=N_UPDATE, eta=ETA, decay=DECAY)

    avg_r_dbl, avg_l_dbl, avg_len_dbl = run_all_plots(
        agent_dbl, r_dbl, l_dbl, len_dbl, tag='double')

    # ── Q9: Dueling DQN ─────────────────────────────────────────────
    print('\n=== Dueling DQN ===')
    agent_duel, r_duel, l_duel, len_duel = train(
        mode='dueling', n_episodes=N_EPISODES, t_epi=T_EPI,
        lr=LR, gamma=GAMMA, buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE, n_update=N_UPDATE, eta=ETA, decay=DECAY)

    avg_r_duel, avg_l_duel, avg_len_duel = run_all_plots(
        agent_duel, r_duel, l_duel, len_duel, tag='dueling')

    # ── Final 3-way comparison ───────────────────────────────────────
    save(
        plot_compare({
            'Standard DQN': avg_r_std,
            'Double DQN':   avg_r_dbl,
            'Dueling DQN':  avg_r_duel,
        }, ylabel='Avg Reward', title='All Methods — Avg Reward'),
        'comparison_reward.png')

    save(
        plot_compare({
            'Standard DQN': avg_len_std,
            'Double DQN':   avg_len_dbl,
            'Dueling DQN':  avg_len_duel,
        }, ylabel='Avg Episode Length', title='All Methods — Avg Length'),
        'comparison_length.png')

    print('\nDone! All figures saved.')


if __name__ == '__main__':
    main()
