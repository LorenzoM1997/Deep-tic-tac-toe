# Deep tic-tac-toe
A deep-q learner for solving the game of tic tac toe. Inspired by the deep-q learner used by Deep Mind to solve Atari games and described in the Lecture 6 of David Silver's reinforcement Learning course.

## Install
Needs numpy and tensorflow to work properly. The code is currently using Tensorflow 1.5, but it is likely to work also with less recent versions. You can use the native install.

    pip3 install --upgrade numpy
    pip3 install --upgrade tensorflow

You can follow the instruction in the official Tensorflow website if you want to install it through Anaconda or if you prefer to use tensorflow-gpu.

## Structure and functionality
The algorithm uses experience replay and fixed Q-targets
* Take action according to epsilon-greedy policy
* Store transition (s, a, r, s_) in replay memory D
* Sample random mini-batch of transitions (s, a, r, s_) from D
* Compute Q-learning targets with respect to old, fixde parameters W_
* Optimize MSE between Q-network and Q-learning targets
* Using gradient descent to minimize error

## Results
The best result achieved for now was quasi-human performance after 60K episodes. Improved hyperparameters are likely to produce human performance in less or equal number of episodes.

## Contact
Please write at lorenz.m97@gmail.com if you need more information.
