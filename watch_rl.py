"""
watch_rl.py — Watch a trained DQN or tabular RL agent play 2048 in real time.

Usage:
    python watch_rl.py --agent dqn --model models/dqn_final.pth
    python watch_rl.py --agent dqn --model models/dqn_final.pth --delay 150
    python watch_rl.py --agent rl
"""

import argparse
import torch

from main import watch_agent


def main():
    parser = argparse.ArgumentParser(description="Watch a trained RL agent play 2048")
    parser.add_argument("--agent", choices=["dqn", "rl"], required=True,
                        help="Which agent to watch: dqn or rl (tabular Q-learning)")
    parser.add_argument("--model", type=str, default="models/dqn_final.pth",
                        help="Path to saved DQN model weights (default: models/dqn_final.pth)")
    parser.add_argument("--delay", type=int, default=300,
                        help="Milliseconds between agent moves (default: 300)")
    args = parser.parse_args()

    if args.agent == "dqn":
        from dqn_agent import DQNAgent
        agent = DQNAgent()
        agent.online_net.load_state_dict(torch.load(args.model, weights_only=True))
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        print(f"Loaded DQN model from {args.model}")
        watch_agent(agent, delay_ms=args.delay)

    elif args.agent == "rl":
        from rl_agent import QLearningAgent
        print("Note: tabular RL agent has no saved model — launching with untrained agent.")
        print("For best results, train first via run_rl.py and pass the agent object directly.")
        agent = QLearningAgent()
        watch_agent(agent, delay_ms=args.delay)


if __name__ == "__main__":
    main()
