import random
import time
import statistics
from collections import Counter
from game import Game2048

import matplotlib.pyplot as plt
import numpy as np


class RandomAgent:
    def select_move(self, game: Game2048) -> int:
        """Return a random valid move (0=left, 1=right, 2=up, 3=down)."""
        valid = game.get_valid_moves()
        return random.choice(valid) if valid else None


def run(num_games: int = 200):
    agent = RandomAgent()
    scores = []
    max_tiles = []
    moves_per_game = []
    move_times = []

    for _ in range(num_games):
        game = Game2048()
        while not game.game_over:
            t0 = time.perf_counter()
            move = agent.select_move(game)
            move_times.append(time.perf_counter() - t0)
            game.step(move)
        scores.append(game.score)
        max_tiles.append(int(game.board.max()))
        moves_per_game.append(game.moves)

    # Tile distribution
    tile_counts = Counter(max_tiles)
    tile_dist = {
        tile: count / num_games * 100
        for tile, count in sorted(tile_counts.items())
    }
    win_rate = sum(1 for t in max_tiles if t >= 2048) / num_games * 100

    # Print report
    sep = "-" * 42
    print(f"\n{'=' * 42}")
    print(f"  Agent: Random  ({num_games} games)")
    print(f"{'=' * 42}")

    print(f"\n  Score metrics")
    print(sep)
    print(f"  {'Average':20s} {statistics.mean(scores):>10.1f}")
    print(f"  {'Median':20s} {statistics.median(scores):>10.1f}")
    print(f"  {'Std dev':20s} {statistics.stdev(scores):>10.1f}")
    print(f"  {'Max':20s} {max(scores):>10}")

    print(f"\n  Tile metrics")
    print(sep)
    print(f"  {'Avg max tile':20s} {statistics.mean(max_tiles):>10.1f}")
    print(f"  {'Max tile':20s} {max(max_tiles):>10}")
    print(f"  {'Win rate (>=2048)':20s} {win_rate:>9.1f}%")
    print(f"\n  Tile distribution:")
    for tile, pct in tile_dist.items():
        bar = "#" * int(pct / 2)
        print(f"    {tile:>5}  {pct:5.1f}%  {bar}")

    print(f"\n  Efficiency metrics")
    print(sep)
    print(f"  {'Avg moves/game':20s} {statistics.mean(moves_per_game):>10.1f}")
    print(f"  {'Avg ms/move':20s} {statistics.mean(move_times) * 1000:>10.3f}")
    print()

    # Plots
    _plot_scores(scores)
    _plot_tile_distribution(tile_dist)


def _plot_scores(scores):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(scores, patch_artist=True,
               boxprops=dict(facecolor=plt.cm.Set2.colors[0]))
    ax.set_title("Score distribution — Random agent")
    ax.set_ylabel("Score")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xticks([1])
    ax.set_xticklabels(["Random"])
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def _plot_tile_distribution(tile_dist):
    tiles = list(tile_dist.keys())
    pcts = list(tile_dist.values())
    x = np.arange(len(tiles))

    fig, ax = plt.subplots(figsize=(max(6, len(tiles) * 1.2), 5))
    ax.bar(x, pcts, color=plt.cm.Set2.colors[0])
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in tiles])
    ax.set_xlabel("Max tile reached")
    ax.set_ylabel("% of games")
    ax.set_title("Max tile distribution — Random agent")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    run(num_games=200)
