"""
Self-learning Tic Tac Toe
Made by Lorenzo Mambretti

Last Update: 8/10/2018 11:38 AM (Lorenzo)
"""
import random
import os
import numpy as np
import progressbar
from File_storage import *
from nn import NN, train_model
from Games import *
import const
import math
import tensorflow as tf
import argparse

class Node:
    global nnet
    def __init__(self):
        self.N = 0
        self.V = 0
        self.Child_nodes = []
        self.board = const.game.board2array()

    def update(self,r):
        self.V = self.V + r
        self.N = self.N + 1

    def Q(self):
        c_puct = 0.2    #hyperparameter
        P = np.max(nnet.call(self.board.reshape(1,27)))

        if self.N == 0:
            return c_puct * P * math.sqrt(self.N)/(1 + self.N)
        else:
            if self.Child_nodes == []:
                Q = self.V
            else:
                Q = ((self.V * self.N) + P)/(self.N + 1)
            return Q / self.N

def check_new_node(current_node):
    """
    this function check if it is the first time the player visited the current node
    if it is the first time, create all the child nodes
    and append them in the Monte Carlo Tree (const.mct)
    """
    if current_node.N == 0:
        # generate child nodes
        for a in range(9):
            if const.game.is_valid(a) == True:
                current_node.Child_nodes.append(len(const.mct))
                const.mct.append(Node())
            else:
                current_node.Child_nodes.append(None)



def random_move(current_node):

    check_new_node(current_node) #check if it is a new node

    #random action
    a = random.randint(0,const.game.action_space - 1)
    while const.game.is_valid(a) == False:
        a = (a + 1) % const.game.action_space
    return a

def choose_move(current_node):

    # if is the first time you visit this node
    if current_node.N == 0:
        # generate child nodes
        for a in range(9):
            if const.game.is_valid(a) == True:
                current_node.Child_nodes.append(len(const.mct))
                const.mct.append(Node())
            else:
                current_node.Child_nodes.append(None)

        #random action
        a = random.randint(0,8)
        while const.game.is_valid(a) == False:
            a = (a + 1) % 9
        return a

    # if you already visited this node
    else:
        best_a = 0
        best_q = -2
        for c in current_node.Child_nodes:
            if c != None:
                if const.mct[c].Q() > best_q:
                    best_q = const.mct[c].Q()
                    best_a = current_node.Child_nodes.index(c)
                #print(const.mct[c].Q())
            #else:
                #print("None")
        return best_a

def simulation(episodes, TRAINING = False):
    node_list = [[]]

    # progressbar
    bar = progressbar.ProgressBar(maxval=episodes, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for e in range(episodes):

        if (e + 1) % (episodes/100) == 0:
            bar.update(e)

        player = e % 2          # choose player
        const.game.restart()             # empty board
        node_list.clear()
        current_node = const.mct[0]   # root of the tree is current node
        node_list.append([0,player])

        # while state not terminal
        while const.game.terminal == False:
            #choose move
            if player == 0:
                #if player 1 not random
                a = choose_move(current_node)
                r = const.game.step(a)
            else:
                #if player 2 epsilon-greedy
                if random.random() < const.EPSILON:
                    a = random_move(current_node)
                else:
                    a = choose_move(current_node)
                const.game.invert_board()
                r = - const.game.step(a)
                const.game.invert_board()

            current_node = const.mct[current_node.Child_nodes[a]]
            player = (player + 1) % 2
            node_list.append([const.mct.index(current_node),player])
            #save state in node list

        #update all nodes
        for node in node_list:
            if node[1] == 0:
                const.mct[node[0]].update(-r)
            else:
                const.mct[node[0]].update(r)

        # train neural network
        train(const.mct, nnet, 100, 2)

    bar.finish()

def play():
    import gui

def train():
    global nnet

    if const.SANITY_CHECK == True:
        if len(const.mct) > 1000:
            # sanity check
            print("Single batch overfit.")
            train_model(const.mct, nnet, 1, 10000)
    # SIMULATION: playing and updating Monte Carlo Tree
    print("Simulating episodes")
    if len(const.mct) < 30000:
        # const.mct is small, make a lot of simulations
        print("Simulation without neural network")
        simulation(95000)
        # TRAINING: neural network is trained while keeping playing
        print("Neural network training")
        simulation(5000, TRAINING = True)
    else:
        # TRAINING: neural network is trained on the Monte Carlo Tree
        print("Neural network training. This will take a while")
        for _ in range(10):
            train_model(const.mct, nnet, 10000,2)
    print("Simulation terminated.")
    # SAVE FILE
    try:
        model.save_weights(const.WEIGHTS_FILENAME)
        print("/tmp/model.ckpt saved correctly.")
    except:
        print("ERROR: an error has occured while saving the weights. The session will not be available when closing the program")

    save_mct(const.mct)


if __name__ == "__main__":
    const.init()

    # create neural network
    nnet = NN(0.0001, 64)
    try:
        nnet.load_weights(const.WEIGHTS_FILENAME)
    except:
        print(const.WEIGHTS_FILENAME, " not found")

    parser = argparse.ArgumentParser(description='Train or play.')
    parser.add_argument('--play', dest='accumulate', action='store_const',
                       const=play, default=train,
                       help='play a const.game (default: train)')

    args = parser.parse_args()
    print(args.accumulate())
