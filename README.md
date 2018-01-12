# Deep tic-tac-toe
A deep-q learner for solving the game of tic tac toe. Inspired by the deep-q learner used by Deep Mind to solve Atari games and described in the Lecture 6 of David Silver's reinforcement Learning course.

## Structure and functionality
The algorithm uses experience replay and fixed Q-targets
* Take action according to epsilon-greedy policy
* Store transition (s, a, r, s_) in replay memory D
* Sample random mini-batch of transitions (s, a, r, s_) from D
* Compute Q-learning targets with respect to old, fixde parameters W_
* Optimize MSE between Q-network and Q-learning targets
* Using gradient descent to minimize error
