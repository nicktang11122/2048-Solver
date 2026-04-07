import numpy as np

# Heuristic Features

def count_empty(board):
    """More empty cells = more room to maneuver"""
    return np.count_nonzero(board == 0)

def corner_max(board):
    """Bonus if the largest tile is in a corner"""
    max_tile = np.max(board)
    corners = [board[0][0], board[0][3], board[3][0], board[3][3]]
    return max_tile if max_tile in corners else 0

def monotonicity(board):
    """Measures how consistently tile values increase/decrease along
    rows and columns. Higher = more monotonic = better.
    For each row and column, check both directions (increasing and decreasing)
    and take the better one. We use log2 of tile values so that the difference
    in higher tiles isn't weighted more heavily than the difference in lower tiles.
    """

    log_board = np.zeros_like(board, dtype=float)
    mask = board > 0
    log_board[mask] = np.log2(board[mask])

    score = 0
    for i in range(4):
        # row i check left to right vs right to left
        row_inc = 0
        row_dec = 0

        for j in range(3):
            diff = log_board[i][j + 1] - log_board[i][j]
            if diff > 0:
                row_dec -= diff
            elif diff < 0:
                row_inc -= (-diff)
        
        # col i check top to bottom vs bottom to top
        col_inc = 0
        col_dec = 0
        for j in range(3):
            diff = log_board[j + 1][i] - log_board[j][i]
            if diff > 0:
                col_dec -= diff
            elif diff < 0:
                col_inc -= (-diff)
        
        # take the less penalized direction for each
        score += max(row_inc, row_dec)
        score += max(col_inc, col_dec)
    
    return score

def smoothness(board):
    """Measures how similar adjacent tiles are. Lower difference between
    neighbors = smoother = tiles more likely to merge.

    Returns a negative value (sum of negative differences), so higher
    (closer to 0) is better.
    """
    log_board = np.zeros_like(board, dtype=float)
    mask = board > 0
    log_board[mask] = np.log2(board[mask])

    score = 0
    for i in range(4):
        for j in range(4):
            if log_board[i][j] == 0:
                continue
            
            # check right neighbor
            if j + 1 < 4 and log_board[i][j + 1] > 0:
                score -= abs(log_board[i][j] - log_board[i][j + 1])
            # check down neighbor
            if i + 1 < 4 and log_board[i + 1][j] > 0:
                score -= abs(log_board[i][j] - log_board[i + 1][j])

    return score

def snake_score(board):
    """Dot product of board values with a snake weight matrix. Rewards
    keeping large tiles in a carner with values flowing outward in a snake pattern.
    """
    weights = np.array([
        [2**15, 2**14, 2**13, 2**12],
        [2**8,  2**9,  2**10, 2**11],
        [2**7,  2**6,  2**5,  2**4],
        [2**0,  2**1,  2**2,  2**3],
    ])
    return np.sum(board * weights)

def merge_potential(board):
    """Count adjacent equal tiles (pairs that could merge)"""
    count = 0
    for i in range(4):
        for j in range(4):
            if board[i][j] == 0:
                continue
            if j + 1 < 4 and board[i][j] == board[i][j + 1]:
                count += 1
            if i + 1 < 4 and board[i][j] == board[i + 1][j]:
                count += 1

    return count

# Evaluation
def evaluate(board):
    """Weighted combination of all heuristic features.
    """
    return (10.0 * count_empty(board)
            +  5.0  * corner_max(board)
            +  3.0  * monotonicity(board)
            +  2.0  * smoothness(board)
            +  1.0  * snake_score(board)
            +  5.0  * merge_potential(board)
            )

# Expectimax Search 
def expectimax(board, depth, is_max_node):
    """
    Expectimax search for 2048
    
    Parameters:
    board : np.array (4x4)
        current board state
    depth : int
        remaining depth to search
    is_max_node : bool
        True = player's turn (pick best move)
        False = chance node (average over tile spawns)

    Returns:
    float
        The expectimax value of this board state
    """

    # Base case: evaluate at depth 0 or game over
    if depth == 0:
        return evaluate(board)
    
    if is_max_node:
        return _max_node(board, depth)
    else:
        return _chance_node(board, depth)
    
def _max_node(board, depth):
    """Player's turn: try all 4 moves, pick the one with the highest expected value"""
    best_value = -float('inf')

    for direction in range(4):
        new_board, score_gained, _ = _simulate_move(board, direction)

        # Skip moves that don't change the board (invalid)
        if np.array_equal(new_board, board):
            continue

        # After our move, it's the change node's turn (tile spawn)
        value = score_gained + expectimax(new_board, depth - 1, is_max_node=False)

        if value > best_value:
            best_value = value

    return best_value

def _chance_node(board, depth):
    """Randome tile spawn: average over all possible placements of 2 (90%) and 4 (10%)"""
    empty_cells = list(zip(*np.where(board == 0)))

    if not empty_cells:
        # No empty cells, pass back to max node
        return expectimax(board, depth - 1, is_max_node=True)
    
    expected_value = 0.0
    num_empty = len(empty_cells)

    for r, c in empty_cells:
        for tile_va, tile_prob in [(2, 0.9), (4, 0.1)]:
            # Place tile
            new_board = board.copy()
            new_board[r][c] = tile_va

            # Weight = probability of this specific cell and this tile value
            weight = tile_prob / num_empty

            # After tile spawns, its the player's turn again
            expected_value += weight * expectimax(new_board, depth - 1, is_max_node=True)

    return expected_value


# Move simulation

def _slide_and_merge_row(row):
    """Slide non-zero tiles left and merge. Returns (new_row, score)"""
    tiles = [v for v in row if v != 0]
    result = []
    score = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = tiles[i] * 2
            score += merged
            result.append(merged)
            i += 2
        else:
            result.append(tiles[i])
            i += 1
    return result + [0] * (4 - len(result)), score

def _simulate_move(board, direction):
    """Simulate a move on a board copy. Returns (new_board, score_gained, None)"""
    b = board.copy()

    # Rotate so every direction becomes a left-slide

    # right
    if direction == 1:
        b = np.fliplr(b)
    
    # up
    elif direction == 2:
        b = b.T.copy()
    
    # down
    elif direction == 3:
        b = np.fliplr(b.T)
    
    new_rows = []
    total_score = 0
    for row in b:
        new_row, s = _slide_and_merge_row(list(row))
        new_rows.append(new_row)
        total_score += s

    new_b = np.array(new_rows, dtype=int)

    # undo rotation
    if direction == 1:
        new_b = np.fliplr(new_b)
    elif direction == 2:
        new_b = new_b.T.copy()
    elif direction == 3:
        new_b = np.flipud(new_b.T).copy()
    
    return new_b, total_score, None

# Agent interface 

def get_best_move(board, depth=3):
    """Given a board state, return the best move (0=left, 1=right, 2=up, 3=down)"""
    best_move = None
    best_value = -float('inf')

    for direction in range(4):
        new_board, score_gained, _ = _simulate_move(board, direction)

        if np.array_equal(new_board, board):
            continue

        value = score_gained + expectimax(new_board, depth - 1, is_max_node=False)

        if value > best_value:
            best_value = value
            best_move = direction

    return best_move