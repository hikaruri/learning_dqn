from game.env import TTT_GameOrganizer
from game.player import PlayerHuman, PlayerRandom, PlayerAlphaRandom, PlayerDQN


def Alpha_vs_Random():
    PLAYER_X = 1
    PLAYER_O = -1
    p1 = PlayerRandom(PLAYER_X)
    p2 = PlayerAlphaRandom(PLAYER_O)
    game = TTT_GameOrganizer(p1, p2, 1000, False)
    game.progress()

def Alpha_vs_DQN():
    PLAYER_X = 1
    PLAYER_O = -1
    p1 = PlayerRandom(PLAYER_X)
    p2 = PlayerDQN(PLAYER_O)
    game = TTT_GameOrganizer(p1, p2, 1000, False)
    game.progress()

if __name__ == "__main__":
    Alpha_vs_DQN()