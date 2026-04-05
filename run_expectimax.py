import argparse
import time
import numpy as np
from game import Game2048
from expectimax_agent import get_best_move

# Play a single game

def play_game(depth=3, verbose=False):
    """Run one full game. Returns dict of stats"""
    game = Game2048()
    moves = 0
    total_time = 0.0
    direction_names = {0: 'Left', 1: 'Right', 2: 'Up', 3: 'Down'}

    while not game.game_over:
        t0 = time.time()
        move = get_best_move(game.board, depth=depth)
        elapsed = time.time() - t0
        total_time += elapsed

        if move is None:
            break

        valid, _, _ = game.step(move)
        if not valid:
            for d in game.get_valid_moves():
                game.step(d)
                break
            else:
                break

        moves += 1

        if verbose and moves % 50 == 0:
            print(f"Move {moves:>4d} | Score: {game.score:>7d} | "
                f"Max Tile: {np.max(game.board):>5d} | Last move: {direction_names[move]:>5s} | "
                f"{elapsed*1000:.0f}ms")

    max_tile = int(np.max(game.board))
    avg_time = (total_time / moves * 1000) if moves > 0 else 0

    return {
        'score': game.score,
        'max_tile': max_tile,
        'moves': moves,
        'avg_ms_per_move': avg_time,
        'total_time': total_time
    }

# Run N games and aggregate
def benchmark(n_games=20, depth=3):
    """Run n_games and print summary statistics"""
    results = []

    print(f"Running {n_games} games at depth {depth}...\n")
    
    for i in range(n_games):
        stats = play_game(depth=depth, verbose=False)
        results.append(stats)
        print(f"  Game {i+1:>3d}/{n_games} | "
            f"Score: {stats['score']:>7d} | "
            f"Max tile: {stats['max_tile']:>5d} | "
            f"Moves: {stats['moves']:>4d} | "
            f"Avg: {stats['avg_ms_per_move']:.0f}ms/move")
    
    scores = [r['score'] for r in results]
    max_tiles = [r['max_tile'] for r in results]
    move_counts = [r['moves'] for r in results]
    times = [r['avg_ms_per_move'] for r in results]

    print(f"RESULTS — {n_games} games, depth {depth}")
    print(f"  Score    — mean: {np.mean(scores):,.0f}  "
        f"median: {np.median(scores):,.0f}  "
        f"std: {np.std(scores):,.0f}")
    print(f"  Moves    — mean: {np.mean(move_counts):.0f}  "
        f"median: {np.median(move_counts):.0f}")
    print(f"  Time     — mean: {np.mean(times):.0f}ms/move")

    # Tile distribution
    tile_counts = {}
    for t in max_tiles:
        tile_counts[t] = tile_counts.get(t, 0) + 1

    print(f"\n  Max tile distribution:")
    for tile in sorted(tile_counts.keys(), reverse=True):
        count = tile_counts[tile]
        pct = count / n_games * 100
        bar = "█" * int(pct / 2)
        print(f"    {tile:>5d}: {count:>3d} ({pct:5.1f}%) {bar}")

    win_count = sum(1 for t in max_tiles if t >= 2048)
    print(f"\n  Win rate (≥2048): {win_count}/{n_games} "
        f"({win_count/n_games*100:.1f}%)")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run expectimax 2048 agent")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run multiple games and show stats")
    parser.add_argument("-n", type=int, default=20,
                        help="Number of games for benchmark (default: 20)")
    parser.add_argument("--depth", type=int, default=3,
                        help="Search depth (default: 3)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print progress during single game")
    args = parser.parse_args()

    if args.benchmark:
        benchmark(n_games=args.n, depth=args.depth)
    else:
        print(f"Playing one game at depth {args.depth}...\n")
        stats = play_game(depth=args.depth, verbose=True)
        print(f"\nFinal score: {stats['score']:,}")
        print(f"Max tile:    {stats['max_tile']}")
        print(f"Moves:       {stats['moves']}")
        print(f"Avg time:    {stats['avg_ms_per_move']:.0f}ms/move")
        print(f"Total time:  {stats['total_time']:.1f}s")