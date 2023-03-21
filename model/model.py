from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward")
)


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


def optimize_model(
    board, next_board, action, reward, policy_net, target_net, device, GAMMA
):
    state_action_pred = torch.tensor(
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
    expected_state_action_value = (next_state_value * GAMMA) + reward
    expected_pred = state_action_pred.clone()
    expected_pred[action] = expected_state_action_value
    if reward != 0:
        # print(action)
        # print(state_action_pred)
        # print(expected_pred)
        print(policy_net.state_dict()["layer2.weight"])
    criterion = nn.MSELoss()
    loss = criterion(state_action_pred, expected_pred)
    optimizer = optim.AdamW(policy_net.parameters(), lr=1e-4, amsgrad=True)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_net
