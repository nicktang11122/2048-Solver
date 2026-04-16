# 2048 AI Solver

**CS 4100 — Artificial Intelligence | Northeastern University**  

**Team:** Brandon Zau, Jonathan Wang, Nicolas Tang, Daniel Wan

## Abstract

2048 is a single-player sliding tile puzzle on a 4×4 grid where the player merges tiles of equal value by swiping in one of four directions. After each move, a new tile (2 or 4) spawns at a random empty cell. The game ends when no valid moves remain. 
The challenge lies in making optimal sequential decisions under uncertainty — each player action is deterministic, but the environment's response (tile placement) is stochastic.

This project explores and compares multiple AI approaches to playing 2048, all built from scratch:

- **Baseline Agent** — a baseline agent implemented through greedy or random (does not count towards an actual AI approach)
- **Local Search** — models the game as alternating max (player) and chance (random tile) nodes with a heuristic evaluation function
- **Adversial Search** — implement expectiminimax or alpha-beta pruning
- **Reinforcement Learning** — formulates the game as an MDP and trains a policy via Q-learning

Each approach is benchmarked over hundreds of games against baseline agents (random and greedy) using metrics including average score, max tile distribution, and win rate (% reaching 2048).

## Repository Structure

```
├── game.py                  # Core 2048 game engine (board logic, move validation, scoring)
├── main.py                  # Pygame GUI — play manually or watch any agent via --ai flag
│
├── expectimax_agent.py      # Expectimax agent with heuristic evaluation
├── run_expectimax.py        # Train/benchmark the expectimax agent
│
├── rl_agent.py              # Tabular Q-learning agent (state bucketing, harmonic decay)
├── dqn_agent.py             # Deep Q-Network agent (replay buffer, target network)
├── run_rl.py                # Train and benchmark the tabular RL agent
├── run_dqn.py               # Train and benchmark the DQN agent (CSV logging, checkpoints)
├── watch_rl.py              # Watch a trained DQN or tabular RL agent play in real time
├── rl_analysis.py           # Generate training reward curve and action heatmap plots
│
├── requirements.txt         # Python dependencies
└── README.md
```
