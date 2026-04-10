import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


# ================================================================
# State Encoding
# ================================================================

def encode_state(board):
    """Log2-encode and normalize a 4x4 board to a 16-float vector.

    Each cell is mapped as:
        0          -> 0.0          (empty)
        tile > 0   -> log2(tile) / 12.0   (12 = log2(4096), our cap)

    Returns a float32 numpy array of shape (16,).
    """
    flat = board.flatten().astype(float)
    encoded = np.where(flat > 0, np.log2(np.where(flat > 0, flat, 1)) / 12.0, 0.0)
    return encoded.astype(np.float32)


# ================================================================
# Replay Buffer
# ================================================================

class ReplayBuffer:
    """Fixed-capacity circular buffer storing (s, a, r, s', done) transitions.

    Written by us — not a library call. Uses a deque for O(1) append/pop.
    """

    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size):
        """Sample a random mini-batch. Returns five numpy arrays."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ================================================================
# Q-Network
# ================================================================

class QNetwork(nn.Module):
    """Two hidden-layer MLP mapping board state -> 4 Q-values.

    Architecture:  16 -> 256 -> ReLU -> 256 -> ReLU -> 4
    Output is raw (unbounded) Q-values, one per direction.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        return self.net(x)


# ================================================================
# DQN Agent
# ================================================================

class DQNAgent:
    """DQN agent for 2048.

    What we wrote:
        - ReplayBuffer (above)
        - Epsilon-greedy action selection restricted to valid moves
        - Q-target computation: r + gamma * max_a Q_target(s', a')
        - Target network hard-copy schedule
        - Training frequency (every train_every steps)
        - Linear epsilon decay schedule

    What PyTorch handles:
        - Layer definitions and forward pass
        - Autograd / backpropagation
        - Adam optimizer and gradient clipping
    """

    def __init__(
        self,
        gamma=0.99,
        lr=1e-4,
        batch_size=64,
        buffer_capacity=200_000,
        min_buffer=1_000,
        target_update_steps=1_000,
        train_every=4,
        epsilon=1.0,
        epsilon_min=0.01,
    ):
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer = min_buffer
        self.target_update_steps = target_update_steps
        self.train_every = train_every
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min

        # Online network: trained every train_every steps via gradient descent
        self.online_net = QNetwork()
        # Target network: frozen copy, hard-updated every target_update_steps
        self.target_net = QNetwork()
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Counts total environment steps (used for train_every and target updates)
        self.total_steps = 0

    # ----------------------------------------------------------------
    # Action selection
    # ----------------------------------------------------------------

    def select_action(self, board, valid_moves):
        """Epsilon-greedy selection restricted to valid moves only."""
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        state_t = torch.FloatTensor(encode_state(board)).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t).squeeze(0).numpy()

        return max(valid_moves, key=lambda a: q_values[a])

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------

    def store(self, board, action, reward, next_board, done):
        """Push one transition into the replay buffer."""
        self.replay_buffer.push(
            encode_state(board), action, reward, encode_state(next_board), done
        )

    def train_step(self):
        """Increment step counter; run a gradient update if conditions are met.

        Conditions:
            1. Buffer has at least min_buffer transitions
            2. total_steps is a multiple of train_every

        Returns the loss value if a gradient update was performed, else None.
        """
        self.total_steps += 1

        if len(self.replay_buffer) < self.min_buffer:
            return None

        if self.total_steps % self.train_every != 0:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t      = torch.FloatTensor(states)
        actions_t     = torch.LongTensor(actions)
        rewards_t     = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t       = torch.FloatTensor(dones)

        # Q-values for the actions actually taken
        q_pred = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Targets: r + gamma * max_a Q_target(s', a'), zeroed out on terminal states
        with torch.no_grad():
            next_q_max = self.target_net(next_states_t).max(dim=1)[0]
            q_target = rewards_t + self.gamma * next_q_max * (1.0 - dones_t)

        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        # Hard-copy online -> target on schedule
        if self.total_steps % self.target_update_steps == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    # ----------------------------------------------------------------
    # Epsilon schedule
    # ----------------------------------------------------------------

    def update_epsilon(self, episode, total_episodes):
        """Linear decay from 1.0 to epsilon_min over first 80% of training."""
        decay_over = int(total_episodes * 0.8)
        if episode < decay_over:
            self.epsilon = 1.0 - (1.0 - self.epsilon_min) * (episode / decay_over)
        else:
            self.epsilon = self.epsilon_min
