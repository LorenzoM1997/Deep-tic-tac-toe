import numpy as np

class Game:

    def __init__(self,num_rows, num_cols, action_space, obs_space):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.action_space = action_space
        self.obs_space = obs_space

    def board2array(self):
        new_board = np.zeros(self.obs_space)
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                val = self.board[row][col]
                new_board[(3 * self.num_cols) * row + 3 * col + val] = 1
        return new_board.reshape(1, self.obs_space)

    def restart(self, player=0):
        raise NotImplementedError()

    def resume(self, player, board, valid_moves, terminal):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def get_valid_moves(self):
        raise NotImplementedError()

    def get_player(self):
        raise NotImplementedError()

    def is_terminal(self):
        raise NotImplementedError()
