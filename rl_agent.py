import math
import random
import numpy as np


# ================================================================
# State Encoding
# ================================================================

def _bucket(tile):
    """Map a tile value to one of 5 bucket categories (0-4).

    0 = empty
    1 = small  (2, 4, 8)      log2 1-3
    2 = medium (16, 32, 64)   log2 4-6
    3 = large  (128, 256, 512) log2 7-9
    4 = xlarge (1024+)         log2 10+
    """
    if tile == 0:
        return 0
    log_val = int(math.log2(tile))
    if log_val <= 3:
        return 1
    elif log_val <= 6:
        return 2
    elif log_val <= 9:
        return 3
    else:
        return 4


def encode_state(board):
    """Convert a 4x4 board into a single integer state ID.

    Each cell is bucketed into one of 5 categories, then the 16
    bucket values are combined via base-5 positional encoding:
        state_id = sum(bucket[i] * 5**i for i in range(16))
    """
    flat = board.flatten()
    state_id = 0
    for i, tile in enumerate(flat):
        state_id += _bucket(int(tile)) * (5 ** i)
    return state_id


# ================================================================
# Q-Learning Agent
# ================================================================

class QLearningAgent:
    """Vanilla Q-learning agent for 2048.

    Design choices:
    - Q-table and N-table are dicts, lazily initialized to zeros on
      first encounter (best-effort approximation, not true convergence).
    - Learning rate uses harmonic decay: eta = 1 / N(s, a).
    - Actions are always sampled from game.get_valid_moves() so
      invalid moves are never attempted.
    - Epsilon decays multiplicatively after each episode.
    """

    def __init__(
        self,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.999990,
        epsilon_min=0.05,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # state_id -> np.array of shape (4,) — one Q-value per direction
        self.Q = {}
        # (state_id, action) -> visit count for harmonic decay
        self.N = {}

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _get_q(self, state_id):
        """Return Q-values for state_id, initializing to zeros if unseen."""
        if state_id not in self.Q:
            self.Q[state_id] = np.zeros(4)
        return self.Q[state_id]

    def _increment_n(self, state_id, action):
        """Increment visit count for (state_id, action) and return new count."""
        key = (state_id, action)
        self.N[key] = self.N.get(key, 0) + 1
        return self.N[key]

    # ----------------------------------------------------------------
    # Public interface
    # ----------------------------------------------------------------

    def select_action(self, board, valid_moves):
        """Epsilon-greedy action selection over valid moves only.

        Ties in Q-values are broken uniformly at random to avoid
        systematic bias when Q-values are all zero (early training).
        """
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        state_id = encode_state(board)
        q_vals = self._get_q(state_id)
        best_val = max(q_vals[a] for a in valid_moves)
        best_actions = [a for a in valid_moves if q_vals[a] == best_val]
        return random.choice(best_actions)

    def update(self, board, action, reward, next_board, done):
        """Q-learning update with harmonic decay learning rate.

        Q(s,a) = (1 - eta) * Q(s,a) + eta * (reward + gamma * max_a' Q(s',a'))
        eta     = 1 / N(s,a)   where N counts updates to this pair
        """
        state_id = encode_state(board)
        next_state_id = encode_state(next_board)

        n = self._increment_n(state_id, action)
        eta = 1.0 / n

        q_vals = self._get_q(state_id)

        if done:
            target = reward
        else:
            next_q = self._get_q(next_state_id)
            target = reward + self.gamma * np.max(next_q)

        q_vals[action] = (1.0 - eta) * q_vals[action] + eta * target

    def decay_epsilon(self):
        """Multiply epsilon by decay rate, floored at epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
