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
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def optimize_model(
    board, next_board, action, reward, policy_net, target_net, device, GAMMA
):
    state_action_pred = policy_net(
        torch.tensor(board.board, dtype=torch.float32, device=device)
    ).unsqueeze(1)
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
    optimizer = optim.SGD(params=policy_net.parameters(), lr=1e-3, momentum=0.9)
    optimizer.zero_grad()
    loss = nn.MSELoss()(expected_pred, state_action_pred)
    loss.backward()
    optimizer.step()
    # print(policy_net.state_dict()["layer3.weight"])
    return policy_net


def optimize_model_for_gym(
    memory, optimizer, policy_net, target_net, device, BATCH_SIZE, GAMMA
):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()