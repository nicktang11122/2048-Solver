"""
Score metrics for 2048 agents.

Usage
-----
from evaluate import run_session, print_report, compare

results = run_session(agent, num_games=200)
print_report("Random", results)

# Side-by-side comparison + plots
compare({"Random": results_random, "Expectimax": results_exp})
"""

import time
import statistics
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from game import Game2048


# Session runner


def run_session(agent, num_games: int = 200) -> dict:
    """
    Run num_games games to completion with agent, collecting all metrics.

    The agent must implement:
        select_move(game: Game2048) -> int  (0=left,1=right,2=up,3=down)

    Returns a dict with keys:
        num_games, scores, max_tiles, moves_per_game,
        avg_score, median_score, std_score, max_score,
        avg_max_tile, max_tile,
        tile_distribution   {tile_value: pct},
        win_rate            (% games reaching 2048),
        avg_moves,
        avg_time_per_move_ms
    """
    scores: list[int] = []
    max_tiles: list[int] = []
    moves_per_game: list[int] = []
    move_times: list[float] = []

    for i in range(num_games):
        game = Game2048()
        while not game.game_over:
            t0 = time.perf_counter()
            move = agent.select_move(game)
            move_times.append(time.perf_counter() - t0)
            game.step(move)

        scores.append(game.score)
        max_tiles.append(int(game.board.max()))
        moves_per_game.append(game.moves)

    tile_counts = Counter(max_tiles)
    tile_dist = {
        tile: count / num_games * 100
        for tile, count in sorted(tile_counts.items())
    }
    win_rate = sum(1 for t in max_tiles if t >= 2048) / num_games * 100

    return {
        "num_games": num_games,
        # raw series (needed for plots)
        "scores": scores,
        "max_tiles": max_tiles,
        "moves_per_game": moves_per_game,
        # score metrics
        "avg_score": statistics.mean(scores),
        "median_score": statistics.median(scores),
        "std_score": statistics.stdev(scores) if num_games > 1 else 0.0,
        "max_score": max(scores),
        # tile metrics
        "avg_max_tile": statistics.mean(max_tiles),
        "max_tile": max(max_tiles),
        "tile_distribution": tile_dist,
        "win_rate": win_rate,
        # efficiency metrics
        "avg_moves": statistics.mean(moves_per_game),
        "avg_time_per_move_ms": statistics.mean(move_times) * 1000,
    }


# Single-agent report

def print_report(name: str, r: dict) -> None:
    """Print a formatted metrics report for one agent."""
    sep = "-" * 42
    print(f"\n{'=' * 42}")
    print(f"  Agent: {name}  ({r['num_games']} games)")
    print(f"{'=' * 42}")

    print(f"\n  Score metrics")
    print(sep)
    print(f"  {'Average':20s} {r['avg_score']:>10.1f}")
    print(f"  {'Median':20s} {r['median_score']:>10.1f}")
    print(f"  {'Std dev':20s} {r['std_score']:>10.1f}")
    print(f"  {'Max':20s} {r['max_score']:>10}")

    print(f"\n  Tile metrics")
    print(sep)
    print(f"  {'Avg max tile':20s} {r['avg_max_tile']:>10.1f}")
    print(f"  {'Max tile':20s} {r['max_tile']:>10}")
    print(f"  {'Win rate (≥2048)':20s} {r['win_rate']:>9.1f}%")
    print(f"\n  Tile distribution:")
    for tile, pct in r["tile_distribution"].items():
        bar = "#" * int(pct / 2)
        print(f"    {tile:>5}  {pct:5.1f}%  {bar}")

    print(f"\n  Efficiency metrics")
    print(sep)
    print(f"  {'Avg moves/game':20s} {r['avg_moves']:>10.1f}")
    print(f"  {'Avg ms/move':20s} {r['avg_time_per_move_ms']:>10.3f}")
    print()


# Multi-agent comparison

def compare(named_results: dict[str, dict], save_path: str | None = None) -> None:
    """
    Print a side-by-side comparison table and show/save visualisations.

    Parameters
    ----------
    named_results : {"AgentName": results_dict, ...}
    save_path     : if given, figures are saved to "<save_path>_scores.png"
                    and "<save_path>_tiles.png" instead of displayed.
    """
    names = list(named_results.keys())
    results = list(named_results.values())

    # Comparison table
    col_w = max(14, max(len(n) for n in names) + 2)
    header_fmt = f"  {{:<22}}" + (f" {{:>{col_w}}}" * len(names))
    row_fmt = header_fmt  # same widths

    print(f"\n{'=' * (24 + col_w * len(names))}")
    print(f"  Agent comparison  ({results[0]['num_games']} games each)")
    print(f"{'=' * (24 + col_w * len(names))}\n")
    print(header_fmt.format("Metric", *names))
    print("  " + "-" * (22 + col_w * len(names)))

    metrics = [
        ("Avg score",         lambda r: f"{r['avg_score']:.1f}"),
        ("Median score",      lambda r: f"{r['median_score']:.1f}"),
        ("Std dev score",     lambda r: f"{r['std_score']:.1f}"),
        ("Max score",         lambda r: f"{r['max_score']}"),
        ("Avg max tile",      lambda r: f"{r['avg_max_tile']:.1f}"),
        ("Max tile",          lambda r: f"{r['max_tile']}"),
        ("Win rate ≥2048",    lambda r: f"{r['win_rate']:.1f}%"),
        ("Avg moves/game",    lambda r: f"{r['avg_moves']:.1f}"),
        ("Avg ms/move",       lambda r: f"{r['avg_time_per_move_ms']:.3f}"),
    ]

    for label, fmt in metrics:
        vals = [fmt(r) for r in results]
        print(row_fmt.format(label, *vals))

    print()

    # visualization
    _plot_score_boxplots(names, results, save_path)
    _plot_tile_distributions(names, results, save_path)


def _plot_score_boxplots(names, results, save_path):
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(names) + 2), 5))
    data = [r["scores"] for r in results]
    bp = ax.boxplot(data, patch_artist=True, labels=names)

    colors = plt.cm.Set2.colors
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_title("Score distribution by agent")
    ax.set_ylabel("Score")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}_scores.png", dpi=150)
        print(f"  Saved {save_path}_scores.png")
    else:
        plt.show()
    plt.close(fig)


def _plot_tile_distributions(names, results, save_path):
    # Collect every tile value seen across all agents
    all_tiles = sorted(
        set(tile for r in results for tile in r["tile_distribution"])
    )

    x = np.arange(len(all_tiles))
    width = 0.8 / len(names)
    colors = plt.cm.Set2.colors

    fig, ax = plt.subplots(figsize=(max(8, len(all_tiles) * 1.2), 5))

    for i, (name, r) in enumerate(zip(names, results)):
        pcts = [r["tile_distribution"].get(t, 0.0) for t in all_tiles]
        offset = (i - (len(names) - 1) / 2) * width
        ax.bar(x + offset, pcts, width, label=name, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in all_tiles])
    ax.set_xlabel("Max tile reached")
    ax.set_ylabel("% of games")
    ax.set_title("Max tile distribution by agent")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(f"{save_path}_tiles.png", dpi=150)
        print(f"  Saved {save_path}_tiles.png")
    else:
        plt.show()
    plt.close(fig)
