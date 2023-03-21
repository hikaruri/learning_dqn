from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        return self.layer3(x)


def optimize_model(state, reward, optimizer, policy_net, target_net, device, GAMMA):
    policy_net.train()
    state_action_value = torch.tensor(
        [policy_net(state).argmax()],
        requires_grad=True,
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        next_state_value = torch.tensor(
            [target_net(state).argmax()],
            dtype=torch.float32,
            device=device,
        )
    # Compute the expected Q values
    expected_state_action_value = (next_state_value * GAMMA) + reward

    # Compute Huber loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_value, expected_state_action_value)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

