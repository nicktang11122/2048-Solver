import numpy as np

# Heuristic Evaluation  
def count_empty(board):
    """More empty cells = more room to maneuver"""
    return np.count_nonzero(board == 0)

def corner_max(board):
    """Bonus if the largest tile is in a corner"""
    max_tile = np.max(board)
    corners = [board[0][0], board[0][3], board[3][0], board[3][3]]
    return max_tile if max_tile in corners else 0

def evaluate(board):
    """Simple heuristic: weighted combination of features
    Weights are hand-tuned for now
    """
    w_empty = 10.0
    w_corner = 5.0

    return (w_empty * count_empty(board) +
            w_corner * corner_max(board))

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