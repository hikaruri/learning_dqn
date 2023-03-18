# -*- coding: utf-8 -*-
import math
import random
from itertools import count
import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers import RecordVideo

from model import DQN, optimize_model
from params import learn_params
from util import ReplayMemory


def select_action(
    env, state, policy_net, steps_done, device, EPS_END, EPS_START, EPS_DECAY
):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1), steps_done
    else:
        return (
            torch.tensor(
                [[env.action_space.sample()]], device=device, dtype=torch.long
            )
        ), steps_done


def main():
    BATCH_SIZE = learn_params["BATCH_SIZE"]
    GAMMA = learn_params["GAMMA"]
    EPS_START = learn_params["EPS_START"]
    EPS_END = learn_params["EPS_END"]
    EPS_DECAY = learn_params["EPS_DECAY"]
    TAU = learn_params["TAU"]
    LR = learn_params["LR"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_dir_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    env = RecordVideo(gym.make("MountainCar-v0", render_mode="rgb_array"), video_dir_name)
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0
    num_episodes = 2000

    episode_durations = []

    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action, steps_done = select_action(
                env,
                state,
                policy_net,
                steps_done,
                device,
                EPS_END,
                EPS_START,
                EPS_DECAY,
            )
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(
                memory, optimizer, policy_net, target_net, device, BATCH_SIZE, GAMMA
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

            if done:
                episode_durations.append(t + 1)
                break

    print("Complete")
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    plt.savefig("duration.png")


if __name__=="__main__":
    main()