# Deep tic-tac-toe
A deep-q learner for solving the game of tic tac toe. Inspired by the deep-q learner used by Deep Mind to solve Atari games and described in the Lecture 6 of David Silver's reinforcement Learning course.

## Install
Needs numpy and tensorflow to work properly. The code is currently using Tensorflow 1.5, but it is likely to work also with less recent versions. You can use the native install.

    pip3 install --upgrade numpy
    pip3 install --upgrade tensorflow
    pip install progressbar

You can follow the instruction in the official Tensorflow website if you want to install it through Anaconda or if you prefer to use tensorflow-gpu.

## Structure and functionality
The algorithm uses experience replay and fixed Q-targets
* Take action according to epsilon-greedy policy
* Store transition (s, a, r, s_) in replay memory D
* Sample random mini-batch of transitions (s, a, r, s_) from D
* Compute Q-learning targets with respect to old, fixde parameters W_
* Optimize MSE between Q-network and Q-learning targets
* Using gradient descent to minimize error

In the file "ddd_ttt.py" a Double-DQN, also inspired by a Deepmind paper, has been implemented.

## Versions
In the file "dd_ttt.py" several improvement have been made in the algorithm. The player is now enclosed in a class and the algorithm has been modified to implement a Double-DQN. The loss reached is equivalent or lower than the one achieved by DQN. However, the best final policy achieved still belongs to DQN. This is the reason why both version are available at the moment.  In the future, we expect the Double-DQN to achieve better results.

The latest version is in the main.py, which uses a simple monte carlo tree search to explore the possibilities.

## Run
To run the Monte Carlo Tree Search algorithm, type in your terminal
    python main.py

To run the DQN algorithm, type in your terminal

    python3 ttt.py
To run the DDQN algorithm, type in your terminal

    python3 dd_ttt.py

The game will ask you to play a match immediately after the training has finished and the data has been saved.

## Results
The best result achieved for now was human-like performance after 100K episodes.

## Contact
Please write at lorenz.m97@gmail.com if you need more information.
