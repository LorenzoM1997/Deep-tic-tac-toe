"""
Self-learning Tic Tac Toe
Made by Lorenzo Mambretti and Hariharan Sezhiyan

Last Update: 1/16/2018 9:05 PM (Lorenzo)
    * tensorboard functionality
    * finalized structure 
    * improved train function
"""

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

def step(state, action):

    # insert
    state_ = State()
    state_.board = np.copy(state.board)
    row_index = int(np.floor(action / 3))
    col_index = action % 3
    state_.board[row_index][col_index] = 1

    # undecided
    terminal = 1

    # to check for 3 in a row horizontal
    for row in range(3):
        for col in range(3):
            if(state_.board[row][col] != 1):
                terminal = 0
        if(terminal == 1):
            state_.terminal = True
            return +1, state_
        else:
            terminal = 1

    # to check for 3 in a row vertical
    for col in range(3):
        for row in range(3):
            if(state_.board[row][col] != 1):
                terminal = 0
        if(terminal == 1):
            state_.terminal = True
            return +1, state_
        else:
            terminal = 1

    # diagonal top-left to bottom-right
    for diag in range(3):
        if(state_.board[diag][diag] != 1):
            terminal = 0
    if(terminal == 1):
        state_.terminal = True
        return +1, state_
    else:
        terminal = 1

    # diagonal bottom-left to top-right
    for diag in range(3):
        if(state_.board[2 - diag][diag] != 1):
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
                break
    if terminal == 1:
        state_.terminal = True

    return 0, state_

def save(W1, W2, B1, B2):
    np.save("weights", sess.run(W1))
    print("file weights.txt has beeen updated successfully")

def load():
    fin = np.load('weights.npy')
    return w1, w2, b1, b2

def extract_policy(state):
    policy = None
    for action in range(9):
        if is_valid(action,state):
            if policy == None:
                policy = action
                best_q = compute_Q_value(state, policy)
            else:
                new_q = compute_Q_value(state, action)
                if new_q > best_q:
                    policy = action
                    best_q = new_q
    return policy
    
def invert_board(state):
    state_ = State()
    state_.board = np.copy(state.board)
    state_.terminal = state.terminal
    for row in range(3):
        for col in range(3):
            if(state.board[row][col] == 1):
                state_.board[row][col] = 2
            elif(state.board[row][col] == 2):
                state_.board[row][col] = 1

    return state_

def play_game():
    while(true):
        start_nb = input("If you would like to move first, enter 1. Otherwise, enter 2. ")
        start = int(start_nb)
        state = State()
        state.board = np.zeros((3,3))
    
        # human moves first
        if(start == 1):
            while(state.terminal != True):
                move_nb = input("Please enter your move: ")
                while(is_valid(int(move_nb), state) == False):
                    move_nb = input("Please enter a correct move: ")
                r, state = step(state, int(move_nb))
                
                if(state.terminal != True):
                    action2 = extract_policy(state)
                else:
                    print("Game finished. You won.")
                    break
                state = invert_board(state)
                r, state = step(state, action2)
                state = invert_board(state)
                if(state.terminal == True):
                    print("Game finished. You lost.")
                    break
                print(state.board)
    
        # ai moves first
        else:
            while(state.terminal != True):
                action_pool = np.random.choice(9,9,replace = False)
                action2 = extract_policy(state)
                state = invert_board(state)
                r, state = step(state, action2)
                state = invert_board(state)
    
                if(state.terminal != True):
                    move_nb = input("Please enter your move: ")
                    while(is_valid(int(move_nb), state) == False):
                        move_nb = input("Please enter a correct move: ")
                else:
                    print("Game finished. You lost.")
                    break
                r, state = step(state, int(move_nb))
                if(state.terminal == True):
                    print("Game finished. You won.")
                    break
                print(state.board)


def compute_Q_value(state,action):
    # computes associated Q value based on NN function approximator
    q_board = np.copy(state.board)
    q_board = np.reshape(q_board, (1,9))
    action_board = np.zeros((1,9))
    action_board[0][action] = 1
    q_board = np.concatenate((q_board, action_board), axis=1)

    #NN forward propogation
    q_value = sess.run(y, feed_dict = {x: q_board})
    return (q_value)

def train(experience_replay):
    
    action_boards_ = np.zeros((576,9))
    for i in range(576):
        action_boards_[i][i % 9] = 1
    
    for i in range(100):

        mini_batch = experience_replay[np.random.choice(experience_replay.shape[0], 64), :]
        action_boards = np.zeros((64,9)) 

        batch = np.reshape(np.array([o.board for o in mini_batch[:,0]]), (64, 9))
        batch_ = np.reshape(np.array([o.board for o in mini_batch[:,3]]), (64, 9))
        for i in range(64):
            action_boards[i][mini_batch[i][1]] = 1
        batch = np.concatenate((batch, action_boards), axis=1)
        # batch will now be a 64 by 18 array
        
        batch_ = np.repeat(batch_, 9, axis=0)
        batch_ = np.concatenate((batch_,action_boards_), axis=1)
        # batch_ will now be a 576 by 18 array
        
        r = mini_batch[:,2] # is the list of all rewards within the mini_batch
        q_t = sess.run(q_target, feed_dict = {x_old: batch_})
        
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch, fixed_qt: q_t, reward : r})
        train_writer.add_summary(summary, i)
    
    return

# Q learner neural network
with tf.name_scope('Q-learner') as scope:
    x = tf.placeholder(tf.float32, [None, 18], name='x')
    with tf.name_scope('hidden_layer') as scope:
        W1 = tf.Variable(tf.random_normal([18, 9]), name='W1')
        b1 = tf.Variable(tf.random_normal([9]), name='b1')
        h1 = tf.tanh(tf.matmul(x, W1) + b1)
    with tf.name_scope('output_layer') as scope:
        W2 = tf.Variable(tf.random_normal([9, 1]), name='W2')
        b2 = tf.Variable(tf.random_normal([1]), name='b2')
        y = tf.tanh(tf.matmul(h1, W2) + b2)

# Q target neural network
with tf.name_scope('Q-target') as scope:
    x_old = tf.placeholder(tf.float32, [None, 18], name='x_old')
    with tf.name_scope('hidden_layer') as scope:
        W1_old = tf.Variable(tf.random_normal([18, 9]), name='W1_old')
        b1_old = tf.Variable(tf.random_normal([9]), name='b1_old')
        h1_old = tf.tanh(tf.matmul(x_old, W1_old) + b1_old, name='h1')
    with tf.name_scope('output_layer') as scope:
        W2_old = tf.Variable(tf.random_normal([9, 1]), name='W2_old')
        b2_old = tf.Variable(tf.random_normal([1]), name='b2_old')
        y_old = tf.tanh(tf.matmul(h1_old, W2_old) + b2_old, name='y_old')
    q_target = tf.reduce_max(tf.reshape(y_old,[64,9]), axis = 1, keep_dims = True,)

with tf.name_scope('loss') as scope:
    fixed_qt = tf.placeholder(tf.float32, [None, 1])
    reward = tf.placeholder(tf.float32, [None])
    gamma = tf.constant(0.9, name='gamma')
    squared_deltas = tf.square(reward + (gamma * fixed_qt) - y)
    loss = tf.reduce_mean(squared_deltas)
    
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()

sess = tf.InteractiveSession()
train_writer = tf.summary.FileWriter('C:/Users/Lorenzo Mambretti/AppData/Local/Programs/Python/Python35/Scripts/Output', sess.graph)
tf.global_variables_initializer().run()

episodes = 1000
n0 = 100.0
experience_replay = np.zeros((0,4))
count_experience = 0

for e in range(episodes):
    # print("episode ",e)
    state = State()
    epsilon = n0 / (n0 + e)
    
    if e % 2 == 1:
        # this is player 2's turn
            state = invert_board(state)
            if random.random() < epsilon:
                # take random action
                action_pool = np.random.choice(9,9, replace = False)
                for a in action_pool:
                    if is_valid(a, state):
                        action = a
                        break
            else:
                # take greedy action
                action = extract_policy(state)

            r, state = step(state, action)
            state = invert_board(state)
            r = -r 
    
    while (state.terminal == False):
        # this section is player 1's turn
        # select epsilon-greedy action
        if random.random() < epsilon:
            # take random action
            action_pool = np.random.choice(9,9, replace = False)
            for a in action_pool:
                if is_valid(a, state):
                    action = a
                    break
        else:
            # take greedy action
            action = extract_policy(state)

        r, state_ = step(state, action)

        if (state_.terminal == False):
            # this is player 2's turn
            state_ = invert_board(state_)
            if random.random() < epsilon:
                # take random action
                action_pool = np.random.choice(9,9, replace = False)
                for a in action_pool:
                    if is_valid(a, state_):
                        action2 = a
                        break
            else:
                # take greedy action
                action2 = extract_policy(state_)

            r, state_ = step(state_, action2)
            state_ = invert_board(state_)
            r = -r 

        D0 = State()
        D0.board = np.copy(state.board)
        D0.terminal = state.terminal
        D1 = State()
        D1.board = np.copy(state_.board)
        D1.terminal = state_.terminal
        D = (D0, action, r, D1)
        experience_replay = np.append(experience_replay, [D], axis = 0)
        state.board = np.copy(state_.board)
        state.terminal = state_.terminal

    if(len(experience_replay) >= 128 and (e % 10) == 0):
        print("Training")
        train(experience_replay)
        if((e % 50) == 0):
            print("Updating old network")
            tf.assign(W1_old, W1)
            tf.assign(W2_old, W2)
            tf.assign(b1_old, b1)
            tf.assign(b2_old, b2)

play_game()
