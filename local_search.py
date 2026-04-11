import numpy as np
import random
import math
import time
from game import Game2048

class SimulatedAnnealingSolver:
    def __init__(self, initial_temp=5000.0, alpha=0.99995, min_temp=0.1):
        self.t = initial_temp
        self.alpha = alpha
        self.min_temp = min_temp
        self.iteration = 0

    #  measures how well the tiles are arranged in a non-increasing order
    def get_monotonicity(self, board):
        scores = [0, 0, 0, 0] 
        # Check horizontal monotonicity
        for r in range(4):
            row = board[r, :]
            for c in range(3):
                if row[c] >= row[c+1] and row[c+1] != 0: scores[0] += row[c]
                elif row[c] <= row[c+1] and row[c] != 0: scores[1] += row[c+1]
        # Check vertical monotonicity
        for c in range(4):
            col = board[:, c]
            for r in range(3):
                if col[r] >= col[r+1] and col[r+1] != 0: scores[2] += col[r]
                elif col[r] <= col[r+1] and col[r] != 0: scores[3] += col[r+1]
        return max(scores[0], scores[1]) + max(scores[2], scores[3])

    # measures how close the tile values are to each other
    def get_smoothness(self, board):
        smoothness = 0
        for r in range(4):
            for c in range(4):
                if board[r, c] != 0:
                    val = math.log2(board[r, c])
                    if c < 3 and board[r, c+1] != 0:
                        smoothness -= abs(val - math.log2(board[r, c+1]))
                    if r < 3 and board[r+1, c] != 0:
                        smoothness -= abs(val - math.log2(board[r+1, c]))
        return smoothness

    # measures how clustered the empty tiles are, which can indicate better maneuverability
    def get_empty_cohesion(self, board):
        empty_coords = np.argwhere(board == 0)
        cohesion_score = 0
         # Create a set for O(1) lookup
        empty_set = set(map(tuple, empty_coords))
        for r, c in empty_coords:
        # Check Right and Down (to avoid double counting)
            for dr, dc in [(0, 1), (1, 0)]:
                if (r + dr, c + dc) in empty_set:
                    cohesion_score += 1
        return cohesion_score

    # The heuristic function that evaluates the board state. 
    def calculate_heuristic(self, board):
        if np.any(board >= 2048): return 1000000
        empty_count = np.count_nonzero(board == 0)
        max_tile = np.max(board)
        
        # Corner Strategy: Massive bonus for keeping max tile in a corner
        corner_bonus = 0
        corners = [(0,0), (0,3), (3,0), (3,3)]
        if any(board[r, c] == max_tile for r, c in corners):
            corner_bonus = max_tile * 50 
        
        # Weighted Score
        return (
            (empty_count * 2000) +               # Survival (highest weight)
            (self.get_monotonicity(board) * 20.0) + # Snake pattern
            (self.get_smoothness(board) * 100.0) +  # No gaps
            corner_bonus                         # Positioning
        )
    
    # Metropolis criterion for accepting worse moves based on temperature
    def get_acceptance_probability(self, delta_e):
        try:
            return math.exp(delta_e / self.t)
        except (OverflowError, ZeroDivisionError):
            return 0.0

    # evaluates all valid moves, scores them using the heuristic, and applies Simulated Annealing logic to select the best move.
    def get_best_move(self, game):
        valid_moves = game.get_valid_moves()
        if not valid_moves: return None

        scored_moves = []
        current_h = self.calculate_heuristic(game.board)

        # Evaluate all moves
        for move in valid_moves:
            new_board, _, _ = game._apply_move(move)
            h_score = self.calculate_heuristic(new_board)
            if move == 2: # UP
                h_score -= 50000 # Massive penalty, NEVER go up unless forced
            elif move == 1: # RIGHT
                h_score -= 10000 # Moderate penalty, avoid if possible
                
            scored_moves.append((h_score, move))

        #  Sort moves from Best to Worst
        scored_moves.sort(reverse=True, key=lambda x: x[0])

        # Try to accept the best moves first
        for h_score, move in scored_moves:
            delta_e = h_score - current_h
            
            # Accept if it's an improvement OR if the Metropolis probability allows it
            if delta_e > 0 or random.random() < self.get_acceptance_probability(delta_e):
                self.iteration += 1
                self.t = max(self.min_temp, self.t * self.alpha)
                return move
        
        # 4. GREEDY FALLBACK
        # If the temperature is freezing and SA rejected all moves
        # force the AI to take the least-bad move, otherwise the game stalls.
        self.iteration += 1
        self.t = max(self.min_temp, self.t * self.alpha)
        return scored_moves[0][1] # Return the absolute best move we found


def run_ai_search():
    num_trials = 3  # How many times it will restart
    for trial in range(num_trials):
        game = Game2048()
        solver = SimulatedAnnealingSolver(initial_temp=2000.0, alpha=0.995)
        
        print(f"\n--- STARTING TRIAL {trial + 1} ---")
        
        while not game.game_over:
            move = solver.get_best_move(game)
            if move is not None:
                game.step(move)
                
                if solver.iteration % 100 == 0:
                    print(f"Iter: {solver.iteration} | T: {solver.t:.1f} | Max: {np.max(game.board)}")
            
            if solver.iteration > 400 and np.max(game.board) < 64:
                print("Bad start detected. Restarting trial.")
                break 

        print("\n" + "="*35)
        print(f"TRIAL {trial + 1} RESULTS")
        print(f"Max Tile Reached: {np.max(game.board)}")
        print(f"Final Score: {game.score}")
        print("="*35)

if __name__ == "__main__":
    run_ai_search()