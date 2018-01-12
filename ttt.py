import random
import numpy as np
import tensorflow as tf

class State:
    board = np.zeros((3,3))
    terminal = False

def is_valid(action, state):
    if state.board[int(np.floor(action / 3))][action % 3] != 0:
        return False
    else:
        return True

def step(state, action, player):

    # insert
    state_ = State()
    state_.board = np.copy(state.board)
    row_index = int(np.floor(action / 3))
    col_index = action % 3
    state_.board[row_index][col_index] = player

    # undecided
    terminal = 1

    # to check for 3 in a row horizontal
    for row in range(3):
        for col in range(3):
            if(state_.board[row][col] != player):
                terminal = 0
        if(terminal == 1):
            state_.terminal = True
            return +1, state_
        else:
            terminal = 1

    # to check for 3 in a row vertical
    for col in range(3):
        for row in range(3):
            if(state_.board[row][col] != player):
                terminal = 0
        if(terminal == 1):
            state_.terminal = True
            return +1, state_
        else:
            terminal = 1

    # diagonal top-left to bottom-right
    for diag in range(3):
        if(state_.board[diag][diag] != player):
            terminal = 0
    if(terminal == 1):
        state_.terminal = True
        return +1, state_
    else:
        terminal = 1

    # diagonal bottom-left to top-right
    for diag in range(3):
        if(state_.board[2 - diag][diag] != player):
            terminal = 0
    if(terminal == 1):
        state_.terminal = True
        return +1, state_
    else:
        terminal = 1

    # checks if board is filled completely
    for row in range(3):
        for col in range(3):
            if(state_.board[row][col] == 0):
                terminal = 0

    if terminal == 1:
        state_.terminal = True
        return 0, state_
    
    reward = 0

    return reward, state_

def extract_policy(state, w1, b1, w2, b2):
    board = np.copy(state.board)
    policy = None
    best_q = None
    for action in range(9):
        if is_valid(action,state):
            if policy == None:
                policy = action
                best_q = compute_Q_value(state, policy, w1, b1, w2, b2)
            else:
                new_q = compute_Q_value(state, action, w1, b1, w2, b2)
                if new_q > best_q:
                    policy = action
                    best_q = new_q
    return policy

def compute_Q_value(state,action,w1,b1,w2,b2):
    # computes associated Q value based on NN function approximator
    q_board = np.copy(state.board)
    q_board = np.reshape(q_board, (1,9))
    action_board = np.zeros((1,9))
    action_board[0][action] = 1
    q_board = np.concatenate((q_board, action_board), axis=1)

    #NN forward propogation
    h1 = tf.tanh(tf.matmul(x, w1) + b1)
    y = tf.tanh(tf.matmul(h1, w2) + b2)
    q_value = sess.run(y, feed_dict = {x: q_board})
    return (q_value)

episodes = 10
n0 = 100.0
experience_replay = np.zeros((0,4))
W1 = tf.Variable(tf.random_uniform([18, 9]))
W2 = tf.Variable(tf.random_uniform([9, 1]))
b1 = tf.Variable(tf.random_uniform([9]))
b2 = tf.Variable(tf.random_uniform([1]))
x = tf.placeholder(tf.float32, [None, 18])

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for e in range(episodes):

    state = State()
    state.board = np.zeros((3,3))

    while (state.terminal == False):
        player = 1
        # select epsilon-greedy action
        epsilon = n0 / (n0 + e)
        if random.random() < epsilon:
            # take random action
            action = random.randint(0,8)
            while (is_valid(action,state) == False):
                action = random.randint(0,8)
        else:
            # take greedy action
            action = extract_policy(state, W1, b1, W2, b2)

        r, state_ = step(state, action, player)

        if (state_.terminal == False):
            player = 2
            if random.random() < epsilon:
                # take random action
                action2 = random.randint(0,8)
                while (is_valid(action2,state_) == False):
                    action2 = random.randint(0,8)
            else:
                # take greedy action
                action2 = extract_policy(state_, W1, b1, W2, b2)
            r, state_ = step(state_, action2, player)
            
            r = -r
        
        value = compute_Q_value(state,action,W1,b1,W2,b2)

        D0 = State()
        D0.board = np.copy(state.board)
        D0.terminal = state.terminal
        D1 = State()
        D1.board = np.copy(state_.board)
        D1.terminal = state_.terminal
        D = (D0, action, D1, r)
        experience_replay = np.append(experience_replay, [D], axis = 0)
        state.board = np.copy(state_.board)
        state.terminal = state_.terminal

