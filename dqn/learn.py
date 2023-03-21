import random

import torch

from model.model import DQN, optimize_model
from model.util import ReplayMemory


def deep_q_learning(
    policy_net: DQN,  # Q_t
    target_net: DQN,  # Q_t+1
    optimizer: torch.optim,
    state: list,
    next_state: list,
    reward,
    action,
    device: str='cpu',
    GAMMA: float = 0.99,
    TAU: float = 0.005,
):
    state = torch.tensor(state, dtype=torch.float32, device=device)
    next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
    action = torch.tensor([action], dtype=torch.float32, device=device)
    reward = torch.tensor([reward], dtype=torch.float32, device=device)
    # Perform one step of the optimization (on the policy network)
    optimize_model(
        state, reward, optimizer, policy_net, target_net, device, GAMMA
    )
    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[
            key
        ] * TAU + target_net_state_dict[key] * (1 - TAU)
    target_net.load_state_dict(target_net_state_dict)
    return policy_net, target_net
