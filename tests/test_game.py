import sys
import os
import random

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

from games.Connect4 import Connect4
from games.TicTacToe import TicTacToe

def assert_valid_game(game):
    game.restart()
    moves = game.get_valid_moves()
    assert len(moves) == game.action_space

    a = random.choice(game.get_valid_moves())
    r = game.step(a)
    assert r == 0

    arr = game.board2array()
    assert arr.shape == (1, game.obs_space)

    p = game.get_player()
    t = game.is_terminal()
    game.resume(game)
    a = random.choice(game.get_valid_moves())
    r = game.step(a)
    assert r == 0

def test_TicTacToe():
    game = TicTacToe()
    assert_valid_game(game)

def test_Connect4():
    game = Connect4()
    assert_valid_game(game)
