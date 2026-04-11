import random
from game import Game2048
from evaluate import run_session, print_report


class RandomAgent:
    def select_move(self, game: Game2048) -> int:
        """Return a random valid move (0=left, 1=right, 2=up, 3=down)."""
        valid = game.get_valid_moves()
        return random.choice(valid) if valid else None


def run(num_games: int = 200):
    agent = RandomAgent()
    results = run_session(agent, num_games=num_games)
    print_report("Random", results)
    return results


if __name__ == "__main__":
    run(num_games=200)
