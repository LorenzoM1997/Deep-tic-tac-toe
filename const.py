import os
from Games import TicTacToe
from File_storage import *

# if true, clear the tree between every simulation session. This helps reducing
# the bias, but it might take longer to converge
CLEAR_TREE_ON_ITERATION = True

EPSILON = 0.1

# if true, run 100 games against a random agent
EVALUATE_ON_ITERATION = True

# whether to perform a sanity check of the system by training one epoch the
# neural network.
SANITY_CHECK = False

# name of the file where the weights of the model are saved
WEIGHTS_FILENAME = "/tmp/my_checkpoint"

# the number of times we are doing the simulation, network training cycle.
N_ITERATION_MAIN = 10

# the number of roll-outs in the simulation
N_ROLLOUTS = 1000

cwd = os.getcwd()
cwd = cwd + '\\tensorflow_logs'

def init():
    # create game
    global game
    game = TicTacToe()

    # initialize Monte Carlo tree
    global mct
    mct = load_mct()
