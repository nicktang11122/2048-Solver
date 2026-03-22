# 2048 AI Solver

**CS 4100 — Artificial Intelligence | Northeastern University**  

**Team:** Brandon Zau, Jonathan Wang, Nicolas Tang, Daniel Wan

## Abstract

2048 is a single-player sliding tile puzzle on a 4×4 grid where the player merges tiles of equal value by swiping in one of four directions. After each move, a new tile (2 or 4) spawns at a random empty cell. The game ends when no valid moves remain. 
The challenge lies in making optimal sequential decisions under uncertainty — each player action is deterministic, but the environment's response (tile placement) is stochastic.

This project explores and compares multiple AI approaches to playing 2048, all built from scratch:

- **Baseline Agent** — a baseline agent implemented through greedy or random (does not count towards an actual AI approach)
- **Local Search** — models the game as alternating max (player) and chance (random tile) nodes with a heuristic evaluation function
- **Adversial Search** — implement expect-minmax or alpha-beta pruning
- **Reinforcement Learning** *(stretch goal)* — formulates the game as an MDP and trains a policy via Q-learning

Each approach is benchmarked over hundreds of games against baseline agents (random and greedy) using metrics including average score, max tile distribution, and win rate (% reaching 2048).

## Repository Structure (TBD)

```
..... Current Structure .....
├── game2048.py          # Game engine (Pygame GUI + headless mode)
├── 2048plan.txt         # our current plan of attack for game2048.py written in psuedo code


..... To Be Implemented ...
├── agents/
│   ├── baseline.py      # Baseline: random valid moves/highest immediate score
│   ├── localsearch.py   # local search agent
├── heuristics.py        # Board evaluation functions
├── reinforcement.py     # reinforcement learning agent

├── benchmark.py         # Run agents and collect performance metrics
├── main.py              # Entry point
└── README.md
```
