import random

import torch
import torch.optim as optim

from model.model import DQN
from dqn.learn import deep_q_learning
from model.util import ReplayMemory


DRAW = 2


class PlayerHuman:
    def __init__(self, turn):
        self.name = "Human"
        self.myturn = turn

    def act(self, board):
        valid = False
        while not valid:
            try:
                act = input(
                    "Where would you like to place " + str(self.myturn) + " (1-9)? "
                )
                act = int(act)
                # if act >= 1 and act <= 9 and board.board[act-1]==EMPTY:
                if act >= 1 and act <= 9:
                    valid = True
                    return act - 1
                else:
                    print("That is not a valid move! Please try again.")
            except Exception as e:
                print(act + "is not a valid move! Please try again.")
        return act

    def getGameResult(self, board):
        if (
            board.winner is not None
            and board.winner != self.myturn
            and board.winner != DRAW
        ):
            print("I lost...")


class PlayerRandom:
    def __init__(self, turn):
        self.name = "Random"
        self.myturn = turn

    def act(self, board):
        acts = board.get_possible_pos()
        i = random.randrange(len(acts))
        return acts[i]

    def getGameResult(self, board):
        pass


class PlayerAlphaRandom:
    def __init__(self, turn, name="AlphaRandom"):
        self.name = name
        self.myturn = turn

    def getGameResult(self, winner):
        pass

    def act(self, board):
        acts = board.get_possible_pos()
        # see only next winnable act
        for act in acts:
            tempboard = board.clone()
            tempboard.move(act, self.myturn)
            # check if win
            if tempboard.winner == self.myturn:
                # print ("Check mate")
                return act
        i = random.randrange(len(acts))
        return acts[i]


class PlayerDQN:
    def __init__(self, turn, name="DQN", device="cpu"):
        self.name = name
        self.myturn = turn
        self.policy_net = DQN(9, 9).to(device)
        self.target_net = DQN(9, 9).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=1e-4, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.device = device
        self.train_flag = False

    def getGameResult(self, winner):
        pass

    def act(self, board):
        # acts = board.get_possible_pos()
        state = board.board
        action = select_action(state, self.policy_net, self.device)
        print(action)
        next_board = board.clone()
        next_board.move(action, self.myturn)
        next_state = next_board
        if self.train_flag:
            if board.winner == None or board.winner == 2:
                reward = 0
            elif board.winner == self.turn:
                reward = 1
            else:
                reward = -1
            self.policy_net, self.target_net = deep_q_learning(
                self.policy_net, 
                self.target_net,
                self.memory,
                self.optimizer,
                state,
                next_state,
                reward,
                action,)
        return action


def select_action(state: list, policy_net, device, eps=0.02):
    sample = random.random()
    if sample > eps:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float64, device=device).unsqueeze(0)
            print(policy_net(state))
            return int(policy_net(state))
    else:
        rnd = random.random()
        return int(rnd * 9 // 9) + 1