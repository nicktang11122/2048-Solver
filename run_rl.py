import argparse
import numpy as np
from game import Game2048
from rl_agent import QLearningAgent


# ================================================================
# Reward shaping helpers
# ================================================================

def _mono_score(board):
    """Monotonicity score for reward shaping.

    For each row and column, computes the violation penalty in both
    directions (increasing and decreasing) and takes the better one.
    Returns a value in [-1, 0]: 0 = perfectly ordered, -1 = maximally chaotic.
    """
    log_board = np.where(board > 0, np.log2(np.where(board > 0, board, 1)), 0.0)
    score = 0.0
    for i in range(4):
        row_inc, row_dec = 0.0, 0.0
        col_inc, col_dec = 0.0, 0.0
        for j in range(3):
            rd = log_board[i][j + 1] - log_board[i][j]
            if rd > 0:
                row_dec -= rd
            elif rd < 0:
                row_inc += rd   # rd is negative, so this subtracts
            cd = log_board[j + 1][i] - log_board[j][i]
            if cd > 0:
                col_dec -= cd
            elif cd < 0:
                col_inc += cd
        score += max(row_inc, row_dec) + max(col_inc, col_dec)
    # Max penalty: 8 sequences * 3 pairs * max log2(4096)=12 = 288
    return score / 288.0


# ================================================================
# Training
# ================================================================

def train(
    agent,
    n_episodes=100_000,
    max_steps=2000,
    mobility_weight=0.3,
    mono_weight=1.0,
    verbose_every=1000,
):
    """Train the agent over n_episodes games.

    Reward per step:
        score_gained / 1000                       (core merge signal)
        + mobility_weight * empty_count / 16       (board openness)
        + mono_weight * (mono_after - mono_before) (monotonicity delta)
        - 5.0                                      (on game over)

    Args:
        agent:           QLearningAgent instance
        n_episodes:      number of training games
        max_steps:       episode truncation to prevent early-training loops
        mobility_weight: weight for the empty-cell bonus
        mono_weight:     weight for the monotonicity delta bonus
        verbose_every:   print a progress line every N episodes
    """
    game = Game2048()

    for episode in range(1, n_episodes + 1):
        game.reset()
        steps = 0

        while not game.game_over and steps < max_steps:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break

            board_before = game.board.copy()
            score_before = game.score
            mono_before  = _mono_score(board_before)

            action = agent.select_action(board_before, valid_moves)
            game.step(action)

            score_gained = game.score - score_before
            empty_count  = int(np.count_nonzero(game.board == 0))
            mono_after   = _mono_score(game.board)

            reward  = score_gained / 1000.0
            reward += mobility_weight * (empty_count / 16.0)
            reward += mono_weight * (mono_after - mono_before)
            if game.game_over:
                reward -= 5.0

            agent.update(board_before, action, reward, game.board, game.game_over)
            steps += 1

        agent.decay_epsilon()

        if episode % verbose_every == 0:
            print(
                f"Episode {episode:>7,} | "
                f"Score: {game.score:>7,} | "
                f"Max tile: {int(np.max(game.board)):>5} | "
                f"Steps: {steps:>5} | "
                f"ε: {agent.epsilon:.4f} | "
                f"Q-table: {len(agent.Q):,}"
            )

    return agent


# ================================================================
# Evaluation (greedy — no exploration)
# ================================================================

def play_game(agent, max_steps=5000):
    """Run one greedy game with the trained agent. Returns stats dict."""
    game = Game2048()
    steps = 0

    while not game.game_over and steps < max_steps:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            break

        saved_eps = agent.epsilon
        agent.epsilon = 0.0
        action = agent.select_action(game.board, valid_moves)
        agent.epsilon = saved_eps

        game.step(action)
        steps += 1

    return {
        'score':    game.score,
        'max_tile': int(np.max(game.board)),
        'steps':    steps,
    }


def benchmark(agent, n_games=50):
    """Run n_games greedy evaluation games and print summary statistics."""
    results = []
    print(f"\nEvaluating over {n_games} games (greedy)...\n")

    for i in range(n_games):
        stats = play_game(agent)
        results.append(stats)
        print(
            f"  Game {i+1:>3}/{n_games} | "
            f"Score: {stats['score']:>7,} | "
            f"Max tile: {stats['max_tile']:>5} | "
            f"Steps: {stats['steps']:>4}"
        )

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


# ================================================================
# Entry point
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the RL 2048 agent")
    parser.add_argument("--episodes",  type=int,   default=100_000, help="Training episodes (default: 100000)")
    parser.add_argument("--max-steps", type=int,   default=2000,    help="Max steps per episode (default: 2000)")
    parser.add_argument("--mobility",  type=float, default=0.3,     help="Empty-cell reward weight (default: 0.3)")
    parser.add_argument("--mono",      type=float, default=1.0,     help="Monotonicity delta reward weight (default: 1.0)")
    parser.add_argument("--eval",      type=int,   default=50,      help="Greedy eval games after training (default: 50)")
    parser.add_argument("--verbose",   type=int,   default=1000,    help="Print progress every N episodes (default: 1000)")
    args = parser.parse_args()

    agent = QLearningAgent()

    print(f"Training for {args.episodes:,} episodes...")
    train(
        agent,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        mobility_weight=args.mobility,
        mono_weight=args.mono,
        verbose_every=args.verbose,
    )

    benchmark(agent, n_games=args.eval)


if __name__ == "__main__":
    main()
