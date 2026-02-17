"""
4-Way Comparison Runner — QRM vs HRL vs HRL-RM vs DDQN on the Chop Cup Magic Trick.

Trains all four agents on the same environment, evaluates periodically,
and generates side-by-side comparison plots.

Methods compared:
  1. QRM      — Q-Learning for Reward Machines (epistemic obs + RM, flat actions)
  2. HRL      — Hierarchical RL with options (physical obs only, no RM)
  3. HRL-RM   — Hierarchical RL + epistemic obs + RM (best of both)
  4. DDQN     — Double Deep Q-Network (neural network, epistemic obs + RM)

Usage:
    python run_comparison.py --episodes 1000 --max-steps 100 --max-limes 5 --seed 42
    python run_comparison.py --seeds 5   # multi-seed averaging
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from chop_cup_env import ChopCupEnv
from reward_machine import RMState, RewardMachine
from qrm import QRMAgent
from hrl_agent import HRLAgent, SUBGOALS
from herl_qrm import HERLQRMAgent
from ddqn_agent import DDQNAgent, _obs_to_vector


# ==========================================================================
# Evaluation helpers
# ==========================================================================

def evaluate_qrm(
    env: ChopCupEnv,
    agent: QRMAgent,
    num_episodes: int,
    max_steps: int,
    seed: int,
) -> Dict:
    """Evaluate QRM with greedy policy."""
    successes = 0
    rewards_list: List[float] = []
    lengths_list: List[int] = []

    for ep in range(num_episodes):
        random.seed(seed + ep)
        env.reset()
        obs = env.get_env_observation()
        ep_reward = 0.0
        step = 0

        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break
            action = agent.get_policy_action(obs, env.reward_machine.get_state(), available)
            _, _, reward, _, _ = env.step(action)
            obs = env.get_env_observation()
            ep_reward += reward
            step += 1

        if RMState.UT in env.reward_machine.reached_states:
            successes += 1
        rewards_list.append(ep_reward)
        lengths_list.append(step)

    n = max(num_episodes, 1)
    return {
        "success_rate": successes / n,
        "mean_reward": float(np.mean(rewards_list)),
        "mean_length": float(np.mean(lengths_list)),
    }


def evaluate_hrl(
    env: ChopCupEnv,
    agent: HRLAgent,
    num_episodes: int,
    max_steps: int,
    seed: int,
) -> Dict:
    """Evaluate HRL with greedy policy at both levels."""
    successes = 0
    rewards_list: List[float] = []
    lengths_list: List[int] = []

    for ep in range(num_episodes):
        random.seed(seed + ep)
        env.reset()
        phys = agent.get_physical_obs(env)
        ep_reward = 0.0
        step = 0
        current_sg = None
        option_steps = 0

        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break

            if current_sg is None:
                current_sg = agent.select_option_greedy(phys)
                option_steps = 0

            action = agent.select_action_greedy(phys, current_sg, available)
            _, _, reward, _, _ = env.step(action)
            phys = agent.get_physical_obs(env)
            ep_reward += reward
            step += 1
            option_steps += 1

            achieved = SUBGOALS[current_sg].check(env.state)
            timeout = option_steps >= SUBGOALS[current_sg].max_steps
            if achieved or timeout or env.done:
                current_sg = None

        if RMState.UT in env.reward_machine.reached_states:
            successes += 1
        rewards_list.append(ep_reward)
        lengths_list.append(step)

    n = max(num_episodes, 1)
    return {
        "success_rate": successes / n,
        "mean_reward": float(np.mean(rewards_list)),
        "mean_length": float(np.mean(lengths_list)),
    }


def evaluate_herl_qrm(
    env: ChopCupEnv,
    agent: HERLQRMAgent,
    num_episodes: int,
    max_steps: int,
    seed: int,
) -> Dict:
    """Evaluate HERL-QRM with greedy policy at both levels."""
    successes = 0
    rewards_list: List[float] = []
    lengths_list: List[int] = []

    for ep in range(num_episodes):
        random.seed(seed + ep)
        env.reset()
        ep_reward = 0.0
        step = 0
        current_sg = None
        option_steps = 0

        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break

            if current_sg is None:
                current_sg = agent.select_option_greedy(env)
                option_steps = 0

            action = agent.select_action_greedy(env, current_sg, available)
            _, _, reward, _, _ = env.step(action)
            ep_reward += reward
            step += 1
            option_steps += 1

            achieved = SUBGOALS[current_sg].check(env.state)
            timeout = option_steps >= SUBGOALS[current_sg].max_steps
            if achieved or timeout or env.done:
                current_sg = None

        if RMState.UT in env.reward_machine.reached_states:
            successes += 1
        rewards_list.append(ep_reward)
        lengths_list.append(step)

    n = max(num_episodes, 1)
    return {
        "success_rate": successes / n,
        "mean_reward": float(np.mean(rewards_list)),
        "mean_length": float(np.mean(lengths_list)),
    }


def evaluate_ddqn(
    env: ChopCupEnv,
    agent: DDQNAgent,
    num_episodes: int,
    max_steps: int,
    seed: int,
) -> Dict:
    """Evaluate DDQN with greedy policy."""
    successes = 0
    rewards_list: List[float] = []
    lengths_list: List[int] = []

    for ep in range(num_episodes):
        random.seed(seed + ep)
        env.reset()
        obs_vec = _obs_to_vector(env)
        ep_reward = 0.0
        step = 0

        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break
            action = agent.get_policy_action(obs_vec, available)
            _, _, reward, _, _ = env.step(action)
            obs_vec = _obs_to_vector(env)
            ep_reward += reward
            step += 1

        if RMState.UT in env.reward_machine.reached_states:
            successes += 1
        rewards_list.append(ep_reward)
        lengths_list.append(step)

    n = max(num_episodes, 1)
    return {
        "success_rate": successes / n,
        "mean_reward": float(np.mean(rewards_list)),
        "mean_length": float(np.mean(lengths_list)),
    }


# ==========================================================================
# Training loops
# ==========================================================================

def train_qrm(
    num_episodes: int,
    max_steps: int,
    eval_interval: int,
    eval_episodes: int,
    discount: float,
    lr: float,
    epsilon: float,
    q_init: float,
    enable_limes: bool,
    max_limes: int,
    seed: int,
    **_extra,
) -> List[Dict]:
    """Train QRM agent and return evaluation stats at regular intervals."""
    random.seed(seed)
    np.random.seed(seed)

    env = ChopCupEnv(enable_limes=enable_limes, max_limes=max_limes, max_steps=max_steps)
    eval_env = ChopCupEnv(enable_limes=enable_limes, max_limes=max_limes, max_steps=max_steps)

    agent = QRMAgent(
        reward_machine=env.reward_machine,
        discount_factor=discount,
        learning_rate=lr,
        epsilon=epsilon,
        q_init=q_init,
    )

    stats: List[Dict] = []
    result = evaluate_qrm(eval_env, agent, eval_episodes, max_steps, seed + 10000)
    stats.append({"episode": 0, **result})

    for ep in range(1, num_episodes + 1):
        random.seed(seed + ep)
        env.reset()
        obs = env.get_env_observation()
        step = 0

        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break

            old_obs = obs
            old_rm = env.reward_machine.get_state()
            action = agent.select_action(obs, old_rm, available)

            _, _, reward, _, _ = env.step(action)
            obs = env.get_env_observation()
            new_rm = env.reward_machine.get_state()
            new_available = env.get_available_actions()

            agent.update(old_obs, action, obs, old_rm, new_rm, reward, new_available)
            step += 1

        agent.training_episodes = ep

        if ep % eval_interval == 0:
            result = evaluate_qrm(eval_env, agent, eval_episodes, max_steps,
                                  seed + 10000 + ep)
            stats.append({"episode": ep, **result})

    return stats


def train_hrl(
    num_episodes: int,
    max_steps: int,
    eval_interval: int,
    eval_episodes: int,
    discount: float,
    lr: float,
    epsilon: float,
    meta_epsilon: float,
    q_init: float,
    enable_limes: bool,
    max_limes: int,
    seed: int,
    **_extra,
) -> List[Dict]:
    """Train HRL agent and return evaluation stats at regular intervals."""
    random.seed(seed)
    np.random.seed(seed)

    env = ChopCupEnv(enable_limes=enable_limes, max_limes=max_limes, max_steps=max_steps)
    eval_env = ChopCupEnv(enable_limes=enable_limes, max_limes=max_limes, max_steps=max_steps)

    agent = HRLAgent(
        discount=discount,
        lr=lr,
        epsilon=epsilon,
        meta_epsilon=meta_epsilon,
        q_init=q_init,
    )

    stats: List[Dict] = []
    result = evaluate_hrl(eval_env, agent, eval_episodes, max_steps, seed + 10000)
    stats.append({"episode": 0, **result})

    for ep in range(1, num_episodes + 1):
        random.seed(seed + ep)
        env.reset()
        phys = agent.get_physical_obs(env)
        agent.begin_episode()
        step = 0

        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break

            if agent.needs_option():
                agent.start_option(phys)

            old_phys = phys
            action = agent.select_action(phys, available)

            _, _, reward, _, _ = env.step(action)
            phys = agent.get_physical_obs(env)
            new_available = env.get_available_actions()

            agent.update(old_phys, action, phys, env.state, reward,
                         new_available, env.done)
            step += 1

        agent.training_episodes = ep

        if ep % eval_interval == 0:
            result = evaluate_hrl(eval_env, agent, eval_episodes, max_steps,
                                  seed + 10000 + ep)
            stats.append({"episode": ep, **result})

    return stats


def train_herl_qrm(
    num_episodes: int,
    max_steps: int,
    eval_interval: int,
    eval_episodes: int,
    discount: float,
    lr: float,
    epsilon: float,
    meta_epsilon: float,
    q_init: float,
    enable_limes: bool,
    max_limes: int,
    seed: int,
    **_extra,
) -> List[Dict]:
    """Train HERL-QRM agent and return evaluation stats at regular intervals."""
    random.seed(seed)
    np.random.seed(seed)

    env = ChopCupEnv(enable_limes=enable_limes, max_limes=max_limes, max_steps=max_steps)
    eval_env = ChopCupEnv(enable_limes=enable_limes, max_limes=max_limes, max_steps=max_steps)

    agent = HERLQRMAgent(
        reward_machine=env.reward_machine,
        discount=discount,
        lr=lr,
        epsilon=epsilon,
        meta_epsilon=meta_epsilon,
        q_init=q_init,
    )

    stats: List[Dict] = []
    result = evaluate_herl_qrm(eval_env, agent, eval_episodes, max_steps, seed + 10000)
    stats.append({"episode": 0, **result})

    for ep in range(1, num_episodes + 1):
        random.seed(seed + ep)
        env.reset()
        agent.begin_episode()
        step = 0

        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break

            if agent.needs_option():
                agent.start_option(env)

            old_obs = agent._make_obs(env)
            action = agent.select_action(env, available)

            _, _, reward, _, _ = env.step(action)
            new_obs = agent._make_obs(env)
            new_available = env.get_available_actions()

            agent.update(
                old_env=env,
                old_obs=old_obs,
                action=action,
                new_obs=new_obs,
                env_state=env.state,
                extrinsic_reward=reward,
                available_new=new_available,
                episode_done=env.done,
            )
            step += 1

        agent.training_episodes = ep

        if ep % eval_interval == 0:
            result = evaluate_herl_qrm(eval_env, agent, eval_episodes, max_steps,
                                       seed + 10000 + ep)
            stats.append({"episode": ep, **result})

    return stats


def train_ddqn(
    num_episodes: int,
    max_steps: int,
    eval_interval: int,
    eval_episodes: int,
    discount: float,
    lr: float,
    epsilon: float,
    q_init: float,
    enable_limes: bool,
    max_limes: int,
    seed: int,
    **_extra,
) -> List[Dict]:
    """Train DDQN agent and return evaluation stats at regular intervals."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = ChopCupEnv(enable_limes=enable_limes, max_limes=max_limes, max_steps=max_steps)
    eval_env = ChopCupEnv(enable_limes=enable_limes, max_limes=max_limes, max_steps=max_steps)

    agent = DDQNAgent(
        reward_machine=env.reward_machine,
        discount=discount,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=500,
        batch_size=64,
        buffer_size=50_000,
        target_update=100,
    )

    stats: List[Dict] = []
    result = evaluate_ddqn(eval_env, agent, eval_episodes, max_steps, seed + 10000)
    stats.append({"episode": 0, **result})

    for ep in range(1, num_episodes + 1):
        random.seed(seed + ep)
        env.reset()
        obs_vec = _obs_to_vector(env)
        step = 0

        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break

            old_vec = obs_vec
            action = agent.select_action(obs_vec, available)

            _, _, reward, _, _ = env.step(action)
            obs_vec = _obs_to_vector(env)
            done = env.done

            agent.store_transition(old_vec, action, reward, obs_vec, done)
            if step % 4 == 0:
                agent.train_step()
            step += 1

        agent.training_episodes = ep

        if ep % eval_interval == 0:
            result = evaluate_ddqn(eval_env, agent, eval_episodes, max_steps,
                                   seed + 10000 + ep)
            stats.append({"episode": ep, **result})

    return stats


# ==========================================================================
# Multi-seed averaging
# ==========================================================================

def run_multi_seed(
    train_fn,
    num_seeds: int,
    base_seed: int,
    **kwargs,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run training over multiple seeds, return averaged metrics.

    Returns:
        (episodes, mean_success, std_success, mean_reward, std_reward,
         mean_length, std_length)
    """
    all_stats = []
    for s in range(num_seeds):
        seed = base_seed + s * 1000
        print(f"    Seed {s+1}/{num_seeds} (seed={seed})")
        stats = train_fn(seed=seed, **kwargs)
        all_stats.append(stats)

    episodes = [s["episode"] for s in all_stats[0]]
    success_matrix = np.array([[s["success_rate"] for s in run] for run in all_stats])
    reward_matrix = np.array([[s["mean_reward"] for s in run] for run in all_stats])
    length_matrix = np.array([[s["mean_length"] for s in run] for run in all_stats])

    return (
        episodes,
        np.mean(success_matrix, axis=0), np.std(success_matrix, axis=0),
        np.mean(reward_matrix, axis=0), np.std(reward_matrix, axis=0),
        np.mean(length_matrix, axis=0), np.std(length_matrix, axis=0),
    )


# ==========================================================================
# Plotting — 4-way comparison
# ==========================================================================

def plot_comparison(
    qrm_data: Tuple,
    hrl_data: Tuple,
    herl_data: Tuple,
    output_dir: str,
    ddqn_data: Tuple = None,
) -> None:
    """Generate comparison plots.

    Each *_data tuple: (episodes, mean_success, std_success,
                         mean_reward, std_reward,
                         mean_length, std_length)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    os.makedirs(out, exist_ok=True)

    q_ep, q_sr, q_sr_s, q_rw, q_rw_s, q_ln, q_ln_s = qrm_data
    r_ep, r_sr, r_sr_s, r_rw, r_rw_s, r_ln, r_ln_s = hrl_data
    h_ep, h_sr, h_sr_s, h_rw, h_rw_s, h_ln, h_ln_s = herl_data

    q_ep = np.array(q_ep)
    r_ep = np.array(r_ep)
    h_ep = np.array(h_ep)

    colors = {
        "qrm":  "#2196F3",   # blue  — user's method
        "hrl":  "#FF9800",   # orange
        "herl": "#4CAF50",   # green
        "ddqn": "#E91E63",   # pink
    }
    labels = {
        "qrm":  "QRM",
        "hrl":  "HRL",
        "herl": "HRL-RM",
        "ddqn": "DDQN",
    }

    # Build datasets list (3 or 4 methods)
    all_datasets = [
        (q_ep, qrm_data, "qrm"),
        (r_ep, hrl_data, "hrl"),
        (h_ep, herl_data, "herl"),
    ]
    if ddqn_data is not None:
        d_ep = np.array(ddqn_data[0])
        all_datasets.append((d_ep, ddqn_data, "ddqn"))

    def _plot_metric(ax, metric_idx, ylabel, title, ylim=None):
        """Plot a single metric for all methods.
        metric_idx: 0=success, 1=reward, 2=length (indexing mean/std pairs)
        """
        for ep_arr, data, key in all_datasets:
            mean = data[1 + metric_idx * 2]
            std  = data[2 + metric_idx * 2]
            ax.plot(ep_arr, mean, color=colors[key], linewidth=2, label=labels[key])
            ax.fill_between(ep_arr, mean - std, mean + std,
                            alpha=0.12, color=colors[key])
        ax.set_xlabel("Training Episode", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)

    # --- Individual plots ---

    # 1. Success Rate
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_metric(ax, 0, "Success Rate", "Success Rate Comparison", ylim=(-0.05, 1.05))
    fig.tight_layout()
    fig.savefig(out / "comparison_success.png", dpi=150)
    fig.savefig(out / "comparison_success.pdf")
    plt.close(fig)

    # 2. Total Reward
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_metric(ax, 1, "Mean Total Reward", "Reward Comparison")
    fig.tight_layout()
    fig.savefig(out / "comparison_reward.png", dpi=150)
    fig.savefig(out / "comparison_reward.pdf")
    plt.close(fig)

    # 3. Episode Length
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_metric(ax, 2, "Mean Episode Length", "Episode Length Comparison")
    fig.tight_layout()
    fig.savefig(out / "comparison_length.png", dpi=150)
    fig.savefig(out / "comparison_length.pdf")
    plt.close(fig)

    # 4. Dashboard (2×2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    _plot_metric(axes[0, 0], 0, "Rate", "Success Rate", ylim=(-0.05, 1.05))
    _plot_metric(axes[0, 1], 1, "Mean Reward", "Total Reward")
    _plot_metric(axes[1, 0], 2, "Mean Steps", "Episode Length")

    # 4th panel: reward gap (QRM − others)
    ax = axes[1, 1]
    all_rw = [q_rw, r_rw, h_rw]
    if ddqn_data is not None:
        all_rw.append(ddqn_data[3])
    min_len = min(len(rw) for rw in all_rw)
    ep_c = np.array(q_ep[:min_len])
    gap_hrl  = np.array(q_rw[:min_len]) - np.array(r_rw[:min_len])
    gap_herl = np.array(q_rw[:min_len]) - np.array(h_rw[:min_len])
    ax.plot(ep_c, gap_hrl,  color=colors["hrl"],  linewidth=2, label="QRM − HRL")
    ax.plot(ep_c, gap_herl, color=colors["herl"], linewidth=2, label="QRM − HRL-RM")
    if ddqn_data is not None:
        d_rw = ddqn_data[3]
        gap_ddqn = np.array(q_rw[:min_len]) - np.array(d_rw[:min_len])
        ax.plot(ep_c, gap_ddqn, color=colors["ddqn"], linewidth=2, label="QRM − DDQN")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Reward Gap", fontsize=12)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Reward Difference")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    title = "Chop Cup — QRM vs HRL vs HRL-RM"
    if ddqn_data is not None:
        title += " vs DDQN"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "comparison_dashboard.png", dpi=150)
    fig.savefig(out / "comparison_dashboard.pdf")
    plt.close(fig)

    print(f"  Plots saved to {output_dir}/")


# ==========================================================================
# Save results
# ==========================================================================

def save_stats(stats: List[Dict], path: Path, method_name: str) -> None:
    """Save evaluation stats to CSV."""
    if not stats:
        return
    csv_path = path / f"{method_name}.stats.csv"
    fieldnames = list(stats[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="4-way comparison: QRM vs HRL vs HRL-RM vs DDQN on the Chop Cup."
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--meta-epsilon", type=float, default=0.2)
    parser.add_argument("--max-limes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds for averaging (default: 1)")
    parser.add_argument("--output", type=str, default="results_comparison")
    args = parser.parse_args()

    out = Path(args.output)
    os.makedirs(out, exist_ok=True)

    common_kwargs = dict(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        discount=args.discount,
        lr=args.lr,
        epsilon=args.epsilon,
        q_init=0.0,
        enable_limes=True,
        max_limes=args.max_limes,
    )

    print("=" * 70)
    print("  Chop Cup — QRM vs HRL vs HRL-RM vs DDQN Comparison")
    print("=" * 70)
    print(f"  Episodes:     {args.episodes}")
    print(f"  Max steps:    {args.max_steps}")
    print(f"  Max limes:    {args.max_limes}")
    print(f"  Seeds:        {args.seeds}")
    print(f"  Eval every:   {args.eval_interval} episodes ({args.eval_episodes} eval eps)")
    print(f"  Output:       {args.output}/")
    print("=" * 70)

    # --- 1. QRM (user's method) ---
    print("\n  [1/4] Training QRM (epistemic + Reward Machine, flat actions)...")
    qrm_data = run_multi_seed(
        train_qrm,
        num_seeds=args.seeds,
        base_seed=args.seed,
        **common_kwargs,
    )
    print(f"    Final success: {qrm_data[1][-1]:.2%}  |  reward: {qrm_data[3][-1]:.1f}")

    # --- 2. HRL baseline ---
    print("\n  [2/4] Training HRL (options, physical only)...")
    hrl_kwargs = {**common_kwargs, "meta_epsilon": args.meta_epsilon}
    hrl_data = run_multi_seed(
        train_hrl,
        num_seeds=args.seeds,
        base_seed=args.seed,
        **hrl_kwargs,
    )
    print(f"    Final success: {hrl_data[1][-1]:.2%}  |  reward: {hrl_data[3][-1]:.1f}")

    # --- 3. HRL-RM ---
    print("\n  [3/4] Training HRL-RM (options + epistemic + RM)...")
    herl_kwargs = {**common_kwargs, "meta_epsilon": args.meta_epsilon}
    herl_data = run_multi_seed(
        train_herl_qrm,
        num_seeds=args.seeds,
        base_seed=args.seed,
        **herl_kwargs,
    )
    print(f"    Final success: {herl_data[1][-1]:.2%}  |  reward: {herl_data[3][-1]:.1f}")

    # --- 4. DDQN ---
    print("\n  [4/4] Training DDQN (neural network, epistemic obs + RM)...")
    ddqn_data = run_multi_seed(
        train_ddqn,
        num_seeds=args.seeds,
        base_seed=args.seed,
        **common_kwargs,
    )
    print(f"    Final success: {ddqn_data[1][-1]:.2%}  |  reward: {ddqn_data[3][-1]:.1f}")

    # --- Save per-method CSV ---
    for data, name in [(qrm_data, "qrm"), (hrl_data, "hrl"),
                       (herl_data, "herl_qrm"), (ddqn_data, "ddqn")]:
        rows = [
            {"episode": e, "success_rate": s, "mean_reward": r, "mean_length": l}
            for e, s, r, l in zip(data[0], data[1], data[3], data[5])
        ]
        save_stats(rows, out, name)

    # --- Info JSON ---
    info = {
        "methods": ["QRM", "HRL", "HRL-RM", "DDQN"],
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "max_limes": args.max_limes,
        "discount": args.discount,
        "lr": args.lr,
        "epsilon": args.epsilon,
        "meta_epsilon": args.meta_epsilon,
        "seeds": args.seeds,
        "base_seed": args.seed,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "qrm_final_success": float(qrm_data[1][-1]),
        "hrl_final_success": float(hrl_data[1][-1]),
        "herl_final_success": float(herl_data[1][-1]),
        "ddqn_final_success": float(ddqn_data[1][-1]),
        "qrm_final_reward": float(qrm_data[3][-1]),
        "hrl_final_reward": float(hrl_data[3][-1]),
        "herl_final_reward": float(herl_data[3][-1]),
        "ddqn_final_reward": float(ddqn_data[3][-1]),
    }
    with open(out / "comparison.info.json", "w") as f:
        json.dump(info, f, indent=4)

    # --- Plots ---
    plot_comparison(qrm_data, hrl_data, herl_data, args.output, ddqn_data=ddqn_data)

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("  Comparison Complete!")
    print("-" * 70)
    print(f"  {'Method':<25} {'Success':>10} {'Reward':>10} {'Length':>10}")
    print("-" * 70)
    for label, data in [("QRM", qrm_data),
                        ("HRL", hrl_data),
                        ("HRL-RM", herl_data),
                        ("DDQN", ddqn_data)]:
        print(f"  {label:<25} {data[1][-1]:>9.2%} {data[3][-1]:>10.1f} {data[5][-1]:>10.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
