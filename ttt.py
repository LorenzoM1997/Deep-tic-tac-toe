import random
import numpy as np

class State:
    board = np.zeros((3,3))

def is_valid(action, state):
    if state.board[int(np.floor(action / 3))][action % 3] != 0:
        return False
    else:
        return True

def step(state, action, player):
    
    # insert
    state_ = State()
    for i in range(3): 
        for j in range(3):
            state_.board[i][j]  = state.board[i][j]
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
            return +1, "terminal"
        else:
            terminal = 1
    
    # to check for 3 in a row vertical
    for col in range(3):
        for row in range(3):
            if(state_.board[row][col] != player):
                terminal = 0
        if(terminal == 1):
            return +1, "terminal"
        else:
            terminal = 1
    
    # diagonal top-left to bottom-right
    for diag in range(3):
        if(state_.board[diag][diag] != player):
            terminal = 0
    if(terminal == 1):
        return +1, "terminal"
    else:
        terminal = 1
    
    # diagonal bottom-left to top-right
    for diag in range(3):
        if(state_.board[2 - diag][diag] != player):
            terminal = 0
    if(terminal == 1):
        return +1, "terminal"
    else:
        terminal = 1

    for row in range(3):
        for col in range(3):
            if(state_.board[row][col] == 0):
                terminal = 0
    if terminal == 1:
        return 0, "terminal"
    reward = 0
    
    return reward, state_

def extract_policy(state):
    action = random.randint(0,8)
    while(is_valid(action,state) == False):
        action = random.randint(0,8)
    return action
    
episodes = 1
n0 = 100.0
experience_replay = np.zeros((0,4))

for e in range(episodes):
    
    state = State()
    state.board = np.zeros((3,3))

    while (state != "terminal"):
        player = 1
        # select epsilon-greedy action
        epsilon = n0 / (n0 + e)
        if random.random() < epsilon:
            # take random action
            action_pool = np.random.choice(9,9, replace= False)
            for a in action_pool:
                if is_valid(a, state):
                    action = a
                    break
        else:
            # take greedy action
            action = extract_policy(state)

        r, state_ = step(state, action, player)

        if (state_ != "terminal"):
            # player 2 acts epsilon greedy
            print(state.board)
            print(state_.board)
            player = 2            
            if random.random() < epsilon:
                # take random action
                action_pool = np.random.choice(9,9, replace= False)
                for a in action_pool:
                    if is_valid(a, state_):
                        action2 = a
                        break
            else:
                # take greedy action
                action2 = extract_policy(state_)
            r, state_ = step(state_, action2, player)
            print(state_.board)
            r = -r

        D = (state, action, state_, r)
        experience_replay = np.append(experience_replay, [D])
        state = state_

