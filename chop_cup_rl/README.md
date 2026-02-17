# Chop Cup RL — Epistemic Reinforcement Learning with Reward Machines

A pure-Python implementation of epistemic reinforcement learning for the Chop Cup
magic trick, based on the BB-POMDP framework (Engesser et al., AAMAS 2025).

## Architecture

| Module | Description |
|--------|-------------|
| `epistemic_logic.py` | SAT-based epistemic logic engine (explicit/implicit beliefs) |
| `reward_machine.py` | Reward Machine automaton from the report |
| `chop_cup_env.py` | Chop Cup environment with belief-based state |
| `q_learning.py` | Batch Q-learning agent with transition memory |
| `main.py` | Training runner, evaluation, CSV/plot output |

## Quick Start

```bash
cd chop_cup_rl
pip install -r requirements.txt
python main.py
```

Results are saved to `results/`.

## Key Concepts

- **Explicit Beliefs (Δφ)**: Direct observations stored in the state.
- **Implicit Beliefs (□φ)**: Derived by SAT solver from Common Ground + Explicit Beliefs.
- **Reward Machine**: Finite automaton tracking magic trick phases, providing rewards on transitions.
- **Epistemic Divergence**: Success = Observer believes φ but reality is ¬φ.
