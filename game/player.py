import random

import torch
import torch.optim as optim

from model.model import DQN
from dqn.learn import deep_q_learning


DRAW = 2
EMPTY = 0


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
    def __init__(self, turn, model_path=None, eps=0.4, name="DQN", device="cpu"):
        self.name = name
        self.myturn = turn
        self.policy_net = DQN(9, 9).to(device)
        self.target_net = DQN(9, 9).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=1e-4)
        self.device = device
        self.train_flag = True
        self.eps = eps
        if model_path is not None:
            self.policy_net.load_state_dict(torch.load(model_path))
            # self.train_flag = False

    def save_model(self, model_path):
        torch.save(self.policy_net.state_dict(), model_path)

    def getGameResult(self, winner):
        pass

    def act(self, board):
        acts = board.get_possible_pos()
        state = board.board
        action = select_action(state, self.policy_net, self.device, acts, self.eps)
        invalid_count = 0
        while state[action] != EMPTY:
            self.policy_net, self.target_net = deep_q_learning(
                self.policy_net,
                self.target_net,
                self.optimizer,
                state,
                state,
                -1.5,
                action,
                self.device,
            )
            action = select_action(state, self.policy_net, self.device, acts, self.eps)
            invalid_count += 1
            if invalid_count > 10:
                # print("Exceed Pos Find" + str(board.board) + " with " + str(action))
                rnd = random.random()
                indices_num = len(acts)
                action = acts[int(rnd * indices_num // indices_num)]
        if self.eps > 0.0001:
            self.eps -= 0.0001
        # print(self.eps)
        next_board = board.clone()
        next_board.move(action, self.myturn)
        next_state = next_board.board
        if self.train_flag:
            if next_board.winner == None:
                reward = 0
            elif next_board.winner == self.myturn:
                reward = 1
            elif next_board.winner == DRAW:
                reward = 0
            else:
                reward = -1
            self.policy_net, self.target_net = deep_q_learning(
                self.policy_net,
                self.target_net,
                self.optimizer,
                state,
                next_state,
                reward,
                action,
                self.device,
            )
            # print(self.policy_net.state_dict()["layer2.weight"][0][12])
        return action


def select_action(state: list, policy_net, device, acts, eps=0.1):
    sample = random.random()
    if sample > eps:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=device)
            return int(policy_net(state).argmax())
    else:
        rnd = random.random()
        indices_num = len(acts)
        return acts[int(rnd * indices_num // indices_num)]
