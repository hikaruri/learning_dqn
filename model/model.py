from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 3)
        self.layer2 = nn.Linear(3, 3)
        self.layer3 = nn.Linear(3, n_actions)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        return self.layer3(x)


def optimize_model(board, next_board, reward, policy_net, target_net, device, GAMMA):
    state_action_value = torch.tensor(
        policy_net(torch.tensor(board.board, dtype=torch.float32)),
        requires_grad=True,
        dtype=torch.float32,
        device=device,
    )
    if next_board.winner is not None:
        next_state_value = 0
    with torch.no_grad():
        next_state_value = target_net(
            torch.tensor(next_board.board, dtype=torch.float32)
        ).argmax()
    # Compute the expected Q values
    expected_state_action_value = torch.tensor((next_state_value * GAMMA) + reward)
    # print(state_action_value)
    # print(expected_state_action_value)
    # Compute Huber loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_value, expected_state_action_value)
    optimizer = optim.AdamW(policy_net.parameters(), lr=1e-4, amsgrad=True)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(policy_net.state_dict()["layer2.weight"])
    return policy_net
