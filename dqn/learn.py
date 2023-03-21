import random

import torch

from model.model import DQN, optimize_model
from model.util import ReplayMemory


def deep_q_learning(
    policy_net: DQN,  # Q_t
    target_net: DQN,  # Q_t+1
    board: list,
    next_board: list,
    reward,
    action,
    device: str='cpu',
    GAMMA: float = 0.99,
    TAU: float = 0.005,
):
    # Perform one step of the optimization (on the policy network)
    policy_net = optimize_model(
        board, next_board, action, reward, policy_net, target_net, device, GAMMA
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
