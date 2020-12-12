from Game import Game

class Connect4(Game):

    def __init__(self):
        self.restart()
        super().__init__(3,3,9,27)

    def restart(self):
        pass
