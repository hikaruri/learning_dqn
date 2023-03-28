from game.env import TTT_GameOrganizer
from game.player import PlayerHuman, PlayerRandom, PlayerAlphaRandom, PlayerDQN


def Alpha_vs_Random():
    PLAYER_X = 1
    PLAYER_O = -1
    p1 = PlayerRandom(PLAYER_X)
    p2 = PlayerAlphaRandom(PLAYER_O)
    game = TTT_GameOrganizer(p1, p2, 1000, False, False)
    game.progress()


def Alpha_vs_DQN():
    PLAYER_X = 1
    PLAYER_O = -1
    p1 = PlayerAlphaRandom(PLAYER_X)
    p2 = PlayerDQN(PLAYER_O, eps=0.4)
    game = TTT_GameOrganizer(p1, p2, 10000, False, False)
    game.progress()
    p2.save_model("./model.cpt")


def Human_vs_DQN():
    PLAYER_X = 1
    PLAYER_O = -1
    p1 = PlayerHuman(PLAYER_O)
    p2 = PlayerDQN(PLAYER_X, model_path="./model.cpt", eps=0)
    game = TTT_GameOrganizer(p1, p2, 5, True)
    game.progress()


def DQN_vs_DQN():
    PLAYER_X = 1
    PLAYER_O = -1
    p1 = PlayerDQN(PLAYER_O, name="dqn1", model_path="model.cpt", eps=0.2)
    p2 = PlayerDQN(PLAYER_X, name="dqn2", eps=0.2)
    game = TTT_GameOrganizer(p1, p2, 10000, False, False)
    game.progress()
    p1.save_model("./model.cpt")


if __name__ == "__main__":
    # Alpha_vs_DQN()
    DQN_vs_DQN()
    Human_vs_DQN()
