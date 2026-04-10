import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch

from game import Game2048
from dqn_agent import DQNAgent


# ================================================================
# Logging & model persistence helpers
# ================================================================

def _make_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _open_csv(path, headers):
    """Open a CSV file for writing and return (file_handle, writer)."""
    f = open(path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(headers)
    return f, writer


def save_model(agent, model_dir, label):
    """Save online network weights to models/<label>.pth."""
    path = os.path.join(model_dir, f"dqn_{label}.pth")
    torch.save(agent.online_net.state_dict(), path)
    return path


def load_model(agent, path):
    """Restore online (and target) network weights from a saved file."""
    state = torch.load(path)
    agent.online_net.load_state_dict(state)
    agent.target_net.load_state_dict(state)


# ================================================================
# Training
# ================================================================

def train(
    agent,
    n_episodes=500_000,
    max_steps=2000,
    mobility_weight=0.1,
    verbose_every=1000,
    log_dir='logs',
    model_dir='models',
    save_every=10_000,
):
    """Train the DQN agent and write a per-episode CSV training log.

    CSV columns (saved to logs/dqn_training_<timestamp>.csv):
        episode, score, max_tile, steps, epsilon, loss, buffer_size, total_steps

    These columns directly support the writeup's learning curve and
    results section — score/max_tile over episodes for the learning curve,
    loss for training stability, tile distribution for key results.

    Model checkpoints saved to models/dqn_ep<N>.pth every save_every episodes.
    Final weights saved to models/dqn_final.pth.

    Returns (agent, timestamp) so benchmark() can write to the same log folder.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    _make_dirs(log_dir, model_dir)

    train_log_path = os.path.join(log_dir, f"dqn_training_{timestamp}.csv")
    log_f, log_writer = _open_csv(
        train_log_path,
        ['episode', 'score', 'max_tile', 'steps', 'episode_reward', 'epsilon', 'loss', 'buffer_size', 'total_steps'],
    )
    print(f"Training log  -> {train_log_path}")

    game = Game2048()
    last_loss = None

    try:
        for episode in range(1, n_episodes + 1):
            game.reset()
            steps = 0
            episode_reward = 0.0

            while not game.game_over and steps < max_steps:
                valid_moves = game.get_valid_moves()
                if not valid_moves:
                    break

                board_before = game.board.copy()
                score_before = game.score

                action = agent.select_action(board_before, valid_moves)
                game.step(action)

                score_gained = game.score - score_before
                empty_count  = int(np.count_nonzero(game.board == 0))

                reward  = score_gained / 1000.0
                reward += mobility_weight * (empty_count / 16.0)
                if game.game_over:
                    reward -= 1.0

                episode_reward += reward
                agent.store(board_before, action, reward, game.board, game.game_over)
                loss = agent.train_step()
                if loss is not None:
                    last_loss = loss
                steps += 1

            agent.update_epsilon(episode, n_episodes)

            # Write one row per episode — fine-grained enough for smooth plots
            log_writer.writerow([
                episode,
                game.score,
                int(np.max(game.board)),
                steps,
                round(episode_reward, 4),
                round(agent.epsilon, 6),
                round(last_loss, 6) if last_loss is not None else '',
                len(agent.replay_buffer),
                agent.total_steps,
            ])

            # Periodic checkpoint
            if episode % save_every == 0:
                path = save_model(agent, model_dir, f'ep{episode}')
                print(f"  [Checkpoint] {path}")

            if episode % verbose_every == 0:
                loss_str = f"{last_loss:.4f}" if last_loss is not None else "   n/a"
                print(
                    f"Episode {episode:>7,} | "
                    f"Score: {game.score:>7,} | "
                    f"Max tile: {int(np.max(game.board)):>5} | "
                    f"Steps: {steps:>5} | "
                    f"ε: {agent.epsilon:.4f} | "
                    f"Loss: {loss_str} | "
                    f"Buffer: {len(agent.replay_buffer):,}"
                )

    finally:
        log_f.close()

    path = save_model(agent, model_dir, 'final')
    print(f"\nFinal model   -> {path}")
    print(f"Training log  -> {train_log_path}")

    return agent, timestamp


# ================================================================
# Evaluation (greedy — no exploration)
# ================================================================

def play_game(agent, max_steps=5000):
    """Run one greedy game with the trained agent. Returns stats dict."""
    game = Game2048()
    steps = 0

    saved_eps = agent.epsilon
    agent.epsilon = 0.0

    while not game.game_over and steps < max_steps:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            break
        action = agent.select_action(game.board, valid_moves)
        game.step(action)
        steps += 1

    agent.epsilon = saved_eps

    return {
        'score':    game.score,
        'max_tile': int(np.max(game.board)),
        'steps':    steps,
    }


def benchmark(agent, n_games=50, log_dir='logs', timestamp=None):
    """Run n_games greedy evaluation games, print and save results.

    CSV columns (saved to logs/dqn_benchmark_<timestamp>.csv):
        game, score, max_tile, steps

    Use this file to build the max-tile distribution chart and comparison
    table for the writeup (tabular Q-learning vs DQN vs expectimax).
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    _make_dirs(log_dir)
    bench_path = os.path.join(log_dir, f"dqn_benchmark_{timestamp}.csv")
    bench_f, bench_writer = _open_csv(bench_path, ['game', 'score', 'max_tile', 'steps'])

    results = []
    print(f"\nEvaluating over {n_games} games (greedy)...\n")

    for i in range(n_games):
        stats = play_game(agent)
        results.append(stats)
        bench_writer.writerow([i + 1, stats['score'], stats['max_tile'], stats['steps']])
        print(
            f"  Game {i+1:>3}/{n_games} | "
            f"Score: {stats['score']:>7,} | "
            f"Max tile: {stats['max_tile']:>5} | "
            f"Steps: {stats['steps']:>4}"
        )

    bench_f.close()

    scores    = [r['score']    for r in results]
    max_tiles = [r['max_tile'] for r in results]
    steps     = [r['steps']    for r in results]

    print(f"\nRESULTS — {n_games} greedy games")
    print(f"  Score    — mean: {np.mean(scores):>8,.0f}  median: {np.median(scores):>8,.0f}  std: {np.std(scores):>8,.0f}")
    print(f"  Steps    — mean: {np.mean(steps):>6.0f}  median: {np.median(steps):>6.0f}")

    tile_counts = {}
    for t in max_tiles:
        tile_counts[t] = tile_counts.get(t, 0) + 1

    print(f"\n  Max tile distribution:")
    for tile in sorted(tile_counts.keys(), reverse=True):
        count = tile_counts[tile]
        pct   = count / n_games * 100
        bar   = "█" * int(pct / 2)
        print(f"    {tile:>5}: {count:>3} ({pct:5.1f}%) {bar}")

    win_count = sum(1 for t in max_tiles if t >= 2048)
    print(f"\n  Win rate (≥2048): {win_count}/{n_games} ({win_count/n_games*100:.1f}%)")
    print(f"\nBenchmark log -> {bench_path}")


# ================================================================
# Entry point
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the DQN 2048 agent")
    parser.add_argument("--episodes",    type=int,   default=500_000, help="Training episodes (default: 500000)")
    parser.add_argument("--max-steps",   type=int,   default=2000,    help="Max steps per episode (default: 2000)")
    parser.add_argument("--mobility",    type=float, default=0.1,     help="Empty-cell reward weight (default: 0.1)")
    parser.add_argument("--eval",        type=int,   default=50,      help="Greedy eval games after training (default: 50)")
    parser.add_argument("--verbose",     type=int,   default=1000,    help="Print progress every N episodes (default: 1000)")
    parser.add_argument("--log-dir",     type=str,   default='logs',  help="Directory for CSV logs (default: logs/)")
    parser.add_argument("--model-dir",   type=str,   default='models',help="Directory for saved models (default: models/)")
    parser.add_argument("--save-every",  type=int,   default=10_000,  help="Save model checkpoint every N episodes (default: 10000)")
    args = parser.parse_args()

    agent = DQNAgent()

    print(f"Training DQN for {args.episodes:,} episodes...")
    agent, timestamp = train(
        agent,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        mobility_weight=args.mobility,
        verbose_every=args.verbose,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        save_every=args.save_every,
    )

    benchmark(agent, n_games=args.eval, log_dir=args.log_dir, timestamp=timestamp)


if __name__ == "__main__":
    main()
