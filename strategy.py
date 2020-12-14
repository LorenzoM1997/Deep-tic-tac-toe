from math import log, sqrt

import const
import numpy as np
import random
import copy
from Node import Node

C_PUCT = 1

def check_new_node(game, mct, current_node, a):
    """
    this function check if it is the first time the player visited the current node
    if it is the first time, create all the child nodes
    and append them in the Monte Carlo Tree (mct)
    """
    if a not in current_node.Child_nodes.keys():
        # generate child nodes
        next_ix = len(mct)
        current_node.Child_nodes[a] = next_ix
        mct[next_ix] = Node(game.board2array())

def random_move(game, current_node):
    #check if it is a new node
    a = random.choice(game.get_valid_moves())
    return a

"""
Returns a predictions given a node, the current game and a model
It uses the value of the monte carlo tree search, and, when not present,
the neural network prediction
"""
def prediction_classic(game, mct, model, current_node):
    pred = model.policy(current_node.board).numpy()[0]

    # else find the best of the nodes
    for a in range(game.action_space):
        if a in current_node.Child_nodes.keys():
            c = current_node.Child_nodes[a]
            pred[a] = mct[c].Q(model)
            continue

        if not game.is_valid(a):
            pred[a] = -1.001

    return pred

"""
Choose the move with the highest value returned by the classic prediction
scheme.
"""
def choose_move_classic(game, mct, model, current_node):

    pred = prediction_classic(game, mct, model, current_node)
    return np.argmax(pred)

def prediction_UTC(game, mct, model, current_node):
    pred = model.policy(current_node.board).numpy()[0]

    for a in range(game.action_space):
        if a in current_node.Child_nodes.keys():
            c = current_node.Child_nodes[a]
            n_i = max(0.5, mct[c].N)
            exploration = C_PUCT * sqrt(log(current_node.N + 1) / n_i)
            pred[a] = mct[c].Q(model) + exploration
            continue

        if not game.is_valid(a):
            pred[a] = -1.001

    return pred

def choose_move_UCT(game, mct, model, current_node):

    pred = prediction_UTC(game, mct, model, current_node)
    return np.argmax(pred)

def predictions_after_rollouts(game, model, current_node, rollouts):

    # create new monte carlo tree with the current node as root node
    mct = {
        0: Node(current_node.board)
    }

    g = copy.deepcopy(game)

    for _ in range(rollouts):

        # create deep copy of the game and current node
        g.resume(game.player, np.copy(game.board), game.valid_moves.copy(), game.terminal)
        node = mct[0]
        node_list = [[0, g.player]]

        while not g.terminal:

            if g.player == game.player:
                a = choose_move_UCT(g, mct, model, node)
                r = g.step(a)
            else:
                a = choose_move_UCT(g, mct, model, node)
                r = -g.step(a)

            check_new_node(g, mct, node, a)
            current_ix = node.Child_nodes[a]
            node = mct[current_ix]
            node_list.append([current_ix, g.player])

        for n in node_list:
            if n[1] == game.player:
                mct[n[0]].update(-r)
            else:
                mct[n[0]].update(r)

    return prediction_classic(game, mct, model, mct[0])
