# 2048 AI Solver

**CS 4100 — Artificial Intelligence | Northeastern University**
**Team:** Brandon Zau, Jonathan Wang, Nicolas Tang, Daniel Wan

## Abstract

2048 is a single-player sliding tile puzzle on a 4×4 grid where the player merges tiles of equal value by swiping in one of four directions. After each move, a new tile (2 or 4) spawns at a random empty cell. The game ends when no valid moves remain. The challenge lies in making optimal sequential decisions under uncertainty — each player action is deterministic, but the environment's response (tile placement) is stochastic.

This project explores and compares multiple AI approaches to playing 2048, all built from scratch:

- **Expectimax Search** — models the game as alternating max (player) and chance (random tile) nodes with a heuristic evaluation function
- **Heuristic Design** — custom board evaluation using features such as monotonicity, smoothness, empty cell count, and corner positioning
- **Genetic Algorithms** — evolves optimal weights for the heuristic evaluation function across generations of simulated gameplay
- **Monte Carlo Tree Search** — uses random playouts to estimate move quality without full tree expansion
- **Reinforcement Learning** *(stretch goal)* — formulates the game as an MDP and trains a policy via Q-learning

Each approach is benchmarked over hundreds of games against baseline agents (random and greedy) using metrics including average score, max tile distribution, and win rate (% reaching 2048).

## Repository Structure

```
├── game2048.py          # Game engine (Pygame GUI + headless mode)
├── agents/
│   ├── random_agent.py  # Baseline: random valid moves
│   ├── greedy_agent.py  # Baseline: highest immediate score
│   ├── expectimax.py    # Expectimax search agent
│   ├── mcts.py          # Monte Carlo Tree Search agent
│   └── genetic.py       # Genetic algorithm for heuristic weight optimization
├── heuristics.py        # Board evaluation functions
├── benchmark.py         # Run agents and collect performance metrics
├── main.py              # Entry point
└── README.md
```

## Setup

```bash
git clone https://github.com/nicktang11122/2048-Solver.git
cd 2048-Solver
pip install -r requirements.txt
python main.py
```

## Requirements

- Python 3.8+
- NumPy
- Pygame
