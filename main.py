"""
Self-learning Tic Tac Toe
Made by Lorenzo Mambretti

Last Update: 8/3/2018 5:55 PM (Lorenzo)
"""
import random
import numpy as np
import progressbar
from nn import NN
from Games import TicTacToe

class Node:    
    def __init__(self):
        self.N = 0
        self.V = 0
        self.Child_nodes = []
        
    def update(self,r):
        self.V = self.V + r
        self.N = self.N + 1

    def Q(self):
        if self.N == 0:
            return 0.05
        else:
            return self.V/self.N

def check_new_node(s, current_node):
    """
    this function check if it is the first time the player visited the current node
    if it is the first time, create all the child nodes
    and append them in the Monte Carlo Tree (mct)
    """
    if current_node.N == 0:
        # generate child nodes
        for a in range(9):
            if s.is_valid(a) == True:
                current_node.Child_nodes.append(len(mct))
                mct.append(Node())
            else:
                current_node.Child_nodes.append(None)

    

def random_move(s, current_node):

    global mct
    check_new_node(s, current_node) #check if it is a new node

    #random action
    a = random.randint(0,8)
    while s.is_valid(a) == False:
        a = (a + 1) % 9
    return a

def choose_move(s,current_node):
    
    global mct
    
    # if is the first time you visit this node
    if current_node.N == 0:
        # generate child nodes
        for a in range(9):
            if s.is_valid(a) == True:
                current_node.Child_nodes.append(len(mct))
                mct.append(Node())
            else:
                current_node.Child_nodes.append(None) 

        #random action
        a = random.randint(0,8)
        while s.is_valid(a) == False:
            a = (a + 1) % 9
        return a

    # if you already visited this node
    else:
        best_a = 0
        best_q = -2
        for c in current_node.Child_nodes:
            if c != None:
                if mct[c].Q() > best_q:
                    best_q = mct[c].Q()
                    best_a = current_node.Child_nodes.index(c)
                #print(mct[c].Q())
            #else:
                #print("None")
        return best_a

def training(episodes):
    global mct
    epsilon = 0.1
    node_list = [[]]

    # progressbar
    bar = progressbar.ProgressBar(maxval=episodes, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    for e in range(episodes):

        if (e + 1) % (episodes/100) == 0:
            bar.update(e)
        
        player = e % 2          # choose player
        s = TicTacToe()             # empty board
        node_list.clear()
        current_node = mct[0]   # root of the tree is current node
        node_list.append([0,player])

        # while state not terminal
        while s.terminal == False:
            #choose move
            if player == 0:
                #if player 1 not random
                a = choose_move(s, current_node)
                r = s.step(a)
            else:
                #if player 2 epsilon-greedy
                if random.random() < epsilon:
                    a = random_move(s, current_node)
                else:
                    a = choose_move(s, current_node)
                s.invert_board()
                r = - s.step(a)
                s.invert_board()

            current_node = mct[current_node.Child_nodes[a]]
            player = (player + 1) % 2
            node_list.append([mct.index(current_node),player])
            #save state in node list

        #update all nodes
        for node in node_list:
            if node[1] == 0:
                mct[node[0]].update(-r)
            else:
                mct[node[0]].update(r)
    bar.finish()
        
mct = []
mct.append(Node())

def play():
    global mct
    node_list = [[]]
    
    while True:
        player = random.randint(0,1)          # choose player
        s = TicTacToe()             # empty board
        node_list.clear()
        current_node = mct[0]   # root of the tree is current node

        # while state not terminal
        while s.terminal == False:
            s.render()
            #choose move
            if player == 0:
                #if player 1 not random
                a = choose_move(s, current_node)
                r = s.step(a)
            else:
                #if player 2 random
                a = int(input("Please enter your move: "))
                while(s.is_valid(a) == False):
                    a = int(input("Please enter a correct move: "))
                check_new_node(s, current_node)
                s.invert_board()
                r = - s.step(a)
                s.invert_board()

            current_node = mct[current_node.Child_nodes[a]]
            player = (player + 1) % 2
            node_list.append([mct.index(current_node),player])

        s.render()        
        if r == 0:
            print ("Tie")
        elif r == -1:
            print ("You won")
        else:
            print ("You lost")

        for node in node_list:
            if node[1] == 0:
                mct[node[0]].update(-r)
            else:
                mct[node[0]].update(r)
    
print("Simulating episodes")
training(100000)
print("Simulation terminated.")
print("Starting Training Neural Network")
print("Training terminated.")
print("Play new game.")
play()
