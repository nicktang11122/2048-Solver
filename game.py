import numpy as np
import random


class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.game_over = False
        self._spawn_tile()
        self._spawn_tile()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.game_over = False
        self._spawn_tile()
        self._spawn_tile()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _spawn_tile(self):
        empty = list(zip(*np.where(self.board == 0)))
        if empty:
            r, c = random.choice(empty)
            self.board[r][c] = 4 if random.random() < 0.1 else 2
            return (int(r), int(c))
        return None

    def _slide_and_merge_row(self, row):
        """Slide non-zero tiles left, merge equal adjacent pairs.
        Returns (new_row, score_gained, moves).
        moves: list of (from_col, to_col, display_value)
        Score is returned, NOT added to self.score here, so validity checks
        don't accidentally corrupt the score.
        """
        tiles = [(i, int(row[i])) for i in range(4) if row[i] != 0]
        result = []
        score_gained = 0
        moves = []
        write = 0
        i = 0
        while i < len(tiles):
            fi, val = tiles[i]
            if i + 1 < len(tiles) and val == tiles[i + 1][1]:
                fj, _ = tiles[i + 1]
                merged = val * 2
                score_gained += merged
                result.append(merged)
                moves.append((fi, write, val))   # first tile slides to merge position
                moves.append((fj, write, val))   # second tile slides to same position
                i += 2
            else:
                result.append(val)
                moves.append((fi, write, val))
                i += 1
            write += 1
        new_row = np.array(result + [0] * (4 - len(result)), dtype=int)
        return new_row, score_gained, moves

    def _apply_move(self, direction):
        """Apply move without mutating state.
        Returns (new_board, score_gained, tile_moves).
        tile_moves: list of (from_r, from_c, to_r, to_c, display_value)
        """
        board = self.board.copy()

        # Rotate/reflect so every direction becomes a left-slide
        if direction == 1:   # right → flip, slide left, flip back
            board = np.fliplr(board)
        elif direction == 2: # up    → transpose, slide left, transpose back
            board = board.T.copy()
        elif direction == 3: # down  → fliplr(T), slide left, undo
            board = np.fliplr(board.T)

        new_rows = []
        score_gained = 0
        raw_moves = []   # (row_idx_in_transformed, from_col, to_col, value)

        for ri, row in enumerate(board):
            new_row, s, moves = self._slide_and_merge_row(row)
            new_rows.append(new_row)
            score_gained += s
            for fc, tc, val in moves:
                raw_moves.append((ri, fc, tc, val))

        new_board = np.array(new_rows)

        # Undo the transformation
        if direction == 1:
            new_board = np.fliplr(new_board)
        elif direction == 2:
            new_board = new_board.T.copy()
        elif direction == 3:
            # Inverse of fliplr(board.T) is flipud(new_board.T)
            new_board = np.flipud(new_board.T).copy()

        # Convert transformed-space coordinates back to original board coordinates
        tile_moves = []
        for ri, fc, tc, val in raw_moves:
            if direction == 0:   # left: no change
                fr, fc_b, tr, tc_b = ri, fc, ri, tc
            elif direction == 1: # right: cols were flipped
                fr, fc_b, tr, tc_b = ri, 3 - fc, ri, 3 - tc
            elif direction == 2: # up: board was transposed → (row,col) ↔ (col,row)
                fr, fc_b, tr, tc_b = fc, ri, tc, ri
            elif direction == 3: # down: fliplr(T)[ri][fc] = board[3-fc][ri]
                fr, fc_b, tr, tc_b = 3 - fc, ri, 3 - tc, ri
            tile_moves.append((fr, fc_b, tr, tc_b, val))

        return new_board, score_gained, tile_moves

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def step(self, direction):
        """Apply a move, spawn a new tile if the board changed.
        direction: 0=left, 1=right, 2=up, 3=down
        Returns (valid, tile_moves, spawn_pos).
        """
        if self.game_over:
            return False, [], None

        new_board, score_gained, tile_moves = self._apply_move(direction)

        if np.array_equal(new_board, self.board):
            return False, [], None

        self.board = new_board
        self.score += score_gained
        spawn_pos = self._spawn_tile()
        self.game_over = not self._has_valid_moves()
        return True, tile_moves, spawn_pos

    def _has_valid_moves(self):
        if np.any(self.board == 0):
            return True
        for d in range(4):
            nb, _, _ = self._apply_move(d)
            if not np.array_equal(nb, self.board):
                return True
        return False

    def get_valid_moves(self):
        return [d for d in range(4) if not np.array_equal(self._apply_move(d)[0], self.board)]

    def __repr__(self):
        lines = [f"Score: {self.score}"]
        for row in self.board:
            lines.append(" ".join(f"{v:5d}" for v in row))
        return "\n".join(lines)
