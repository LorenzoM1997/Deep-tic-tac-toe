import os
from Games import TicTacToe
from File_storage import *

EPSILON = 0.1
SANITY_CHECK = True
WEIGHTS_FILENAME = "/tmp/my_checkpoint"
cwd = os.getcwd()
cwd = cwd + '\\tensorflow_logs'

def init():
    # create game
    global game
    game = TicTacToe()

    # initialize Monte Carlo tree
    global mct
    mct = load_mct()
    if mct == []:
        mct.append(Node(game))
