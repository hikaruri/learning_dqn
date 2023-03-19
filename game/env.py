import random


EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1
MARKS = {PLAYER_X: "X", PLAYER_O: "O", EMPTY: " "}
DRAW = 2


class TTTBoard:
    def __init__(self, board=None):
        if board == None:
            self.board = []
            for i in range(9):
                self.board.append(EMPTY)
        else:
            self.board = board
        self.winner = None

    def get_possible_pos(self):
        pos = []
        for i in range(9):
            if self.board[i] == EMPTY:
                pos.append(i)
        return pos

    def print_board(self):
        tempboard = []
        for i in self.board:
            tempboard.append(MARKS[i])
        row = " {} | {} | {} "
        hr = "\n-----------\n"
        print((row + hr + row + hr + row).format(*tempboard))

    def check_winner(self):
        win_cond = (
            (1, 2, 3),
            (4, 5, 6),
            (7, 8, 9),
            (1, 4, 7),
            (2, 5, 8),
            (3, 6, 9),
            (1, 5, 9),
            (3, 5, 7),
        )
        for each in win_cond:
            if (
                self.board[each[0] - 1]
                == self.board[each[1] - 1]
                == self.board[each[2] - 1]
            ):
                if self.board[each[0] - 1] != EMPTY:
                    self.winner = self.board[each[0] - 1]
                    return self.winner
        return None

    def check_draw(self):
        if len(self.get_possible_pos()) == 0 and self.winner is None:
            self.winner = DRAW
            return DRAW
        return None

    def move(self, pos, player):
        if self.board[pos] == EMPTY:
            self.board[pos] = player
        else:
            self.winner = -1 * player
        self.check_winner()
        self.check_draw()

    def clone(self):
        return TTTBoard(self.board.copy())

    def switch_player(self):
        if self.player_turn == self.player_x:
            self.player_turn = self.player_o
        else:
            self.player_turn = self.player_x


class TTT_GameOrganizer:

    act_turn = 0
    winner = None

    def __init__(self, px, po, nplay=1, showBoard=True, showResult=True, stat=100):
        self.player_x = px
        self.player_o = po
        self.nwon = {px.myturn: 0, po.myturn: 0, DRAW: 0}
        self.nplay = nplay
        self.players = (self.player_x, self.player_o)
        self.board = None
        self.disp = showBoard
        self.showResult = showResult
        self.player_turn = self.players[random.randrange(2)]
        self.nplayed = 0
        self.stat = stat

    def progress(self):
        while self.nplayed < self.nplay:
            self.board = TTTBoard()
            while self.board.winner == None:
                if self.disp:
                    print("Turn is " + self.player_turn.name)
                act = self.player_turn.act(self.board)
                self.board.move(act, self.player_turn.myturn)
                if self.disp:
                    self.board.print_board()

                if self.board.winner != None:
                    # notice every player that game ends
                    for i in self.players:
                        i.getGameResult(self.board)
                    if self.board.winner == DRAW:
                        if self.showResult:
                            print("Draw Game")
                    elif self.board.winner == self.player_turn.myturn:
                        out = "Winner : " + self.player_turn.name
                        if self.showResult:
                            print(out)
                    else:
                        print("Invalid Move!")
                    self.nwon[self.board.winner] += 1
                else:
                    self.switch_player()
                    # Notice other player that the game is going
                    self.player_turn.getGameResult(self.board)

            self.nplayed += 1
            if self.nplayed % self.stat == 0 or self.nplayed == self.nplay:
                print(
                    self.player_x.name
                    + ":"
                    + str(self.nwon[self.player_x.myturn])
                    + ","
                    + self.player_o.name
                    + ":"
                    + str(self.nwon[self.player_o.myturn])
                    + ",DRAW:"
                    + str(self.nwon[DRAW])
                )

    def switch_player(self):
        if self.player_turn == self.player_x:
            self.player_turn = self.player_o
        else:
            self.player_turn = self.player_x
