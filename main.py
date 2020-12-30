"""
Self-learning Tic Tac Toe
Made by Lorenzo Mambretti

Last Update: Dec 8 2020
"""
from File_storage import *
from nn import train_model, get_model, save_model
from strategy import check_new_node
from Node import Node

import random
import numpy as np
import progressbar
import const
import tensorflow as tf
import argparse
import strategy

def evaluation(game, episodes):
    print("evaluation() started")

    defeats = 0
    victories = 0
    for e in range(episodes):

        game.restart(player = e % 2)
        current_node = const.mct[0]

        while not game.terminal:
            # agent to evaluate
            if game.player == 0:
                pred = strategy.predictions_after_rollouts(
                    game, nnet, current_node, 100)
                a = np.argmax(pred)
                r = game.step(a)
            else:
                # random agent
                a = strategy.random_move(game, current_node)
                r = -game.step(a)

            check_new_node(game, const.mct, current_node, a)
            current_node = const.mct[current_node.Child_nodes[a]]

        if r == 1:
            victories += 1
        elif r == -1:
            defeats += 1

    return victories, defeats


def simulation(game, episodes):
    print("simulation() started")
    node_list = [[]]

    # progressbar
    bar = progressbar.ProgressBar(maxval=episodes, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    replay = episodes*[None]

    for e in range(episodes):

        if (e + 1) % (episodes/100) == 0:
            bar.update(e)

        game.restart(player = e % 2)             # empty board
        node_list.clear()
        current_node = const.mct[0]   # root of the tree is current node
        node_list.append([0,game.player])

        # while state not terminal
        while game.terminal == False:

            pred = strategy.predictions_after_rollouts(
                game, nnet, current_node, 200)
            if random.random() < const.EPSILON:
                a = strategy.random_move(game, current_node)
            else:
                a = np.argmax(pred)

            r = game.step(a)
            if game.player == 1:
                r *= -1

            check_new_node(game, const.mct, current_node, a)
            current_ix = current_node.Child_nodes[a]
            current_node = const.mct[current_ix]
            node_list[-1].append(pred)
            node_list.append([current_ix, game.player])
            #save state in node list

        #update all nodes
        for node in node_list:
            if node[1] == 0:
                node[1] = -r
            else:
                node[1] = r

        # append to replay memory
        node_list.pop()
        replay[e] = node_list.copy()

    bar.finish()
    return replay

def play():
    import gui

def extract_data(replay, game, model, min_visits = const.MIN_VISITS):
    print("extract_data(): start")

    n_nodes = len(replay) * game.obs_space
    data = np.empty((n_nodes, game.obs_space))
    policy_labels = np.empty((n_nodes, game.action_space))
    value_labels = np.empty((n_nodes, 1))

    i = 0
    for node_list in replay:

        for node in node_list:

            c = const.mct[node[0]]
            data[i,:] = c.board
            policy_labels[i,:] = node[2]
            value_labels[i,:] = node[1]

            i += 1

    data = data[:i,:]
    policy_labels = policy_labels[:i, :]
    value_labels = value_labels[:i, :]
    print("extract_data(): end")

    return data, policy_labels, value_labels

def train():
    # create neural network
    global nnet
    nnet = get_model()

    # SIMULATION: playing and updating Monte Carlo Tree
    if not bool(const.mct):
        # start with a simulation
        const.mct[0] = Node(const.game.board2array())

    for _ in range(const.N_ITERATION_MAIN):

        replay = simulation(const.game, const.N_ROLLOUTS)

        data, p_labels, v_labels = extract_data(replay, const.game, nnet)

        # train the network
        train_model(data, p_labels, v_labels, nnet, const.EPOCHS)

        # clear the monte carlo tree
        if (const.CLEAR_TREE_ON_ITERATION):
            const.game.restart()
            const.mcd = {
                0: Node(const.game.board2array())
            }

        if (const.EVALUATE_ON_ITERATION):
            w, l = evaluation(const.game, const.EVALUATION_EPISODES)
            t = const.EVALUATION_EPISODES - (w + l)
            print("won: ", w, " ties: ", t, " lost: ",l)

    # save model
    save_model(nnet)

    # save latest monte carlo tree
    save_mct(const.mct, const.WEIGHTS_PATH, const.GAME)


if __name__ == "__main__":
    const.init()

    parser = argparse.ArgumentParser(description='Train or play.')
    parser.add_argument('--play', dest='accumulate', action='store_const',
                       const=play, default=train,
                       help='play a const.game (default: train)')

    args = parser.parse_args()
    print(args.accumulate())
