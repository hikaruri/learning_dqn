import random


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
