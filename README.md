# Deep Tic Tac Toe
Here is implemented a simplified version of Alpha Zero, reinforcement learning developed by Google Deep Mind for chess. Here it is re-purposed to play tic tac toe. Being much simpler than chess, it needs less time to train and it uses more heavily the Monte Carlo Tree search, and the neural network plays a minor role.

## Install
Needs numpy,tensorflow and progressbar to work properly. The code is currently using Tensorflow 1.9, but it is likely to work also with less recent versions. You can use the native pip install.

    pip3 install --upgrade numpy
    pip3 install --upgrade tensorflow
    pip install progressbar

You can follow the instruction in the official Tensorflow website if you want to install it through Anaconda. If you have a CUDA-compatible Nvidia GPU, I suggest to install the gpu version for better performances.

    pip install --upgrade tensorflow-gpu

## Structure and functionality
The algorithm implements a Monte Carlo Tree to explore the board. After an initial exploration using the monte carlo tree search, the network start training a deep dense neural network. The monte carlo tree explores the games always until the end, the training is therefore online.

The neural networks has 3 hidden layes, for a total of 99 neurons. It is trained using the Adam Optimizer provided by tensorflow with a learning rate of 0.00025, 200k iterations in total with batches randomly chosen from the nodes of the monte carlo tree.

Both the neural network and the Monte Carlo tree are saved locally during the execution, so it is faster on the next execution.

## Versions
The main version is contained in the file main.py, but there are two more versions. These two versions did not achieve the same results unfortunately, but they are technically good approaches. Further developments may solve the current problems and make them achieve better performances.

The file "ttt.py" implements a DQN, similar to the one described by the Lecture n.6 in David Silver course on Reinforcement Learning (you can find it on YouTube). In the file "dd_ttt.py" several improvement have been made in the algorithm. The player is now enclosed in a class and the algorithm has been modified to implement a Double-DQN. The loss reached is equivalent or lower than the one achieved by DQN. However, the best final policy achieved still belongs to DQN. This is the reason why both version are available at the moment.  In the future, we expect the Double-DQN to achieve better results.

## Run
Download and extract the repository to your local machine. After the installation, execute the following.
To train the program instead of playing, type in your terminal

    python main.py --train
    
To train the algorithm, just type

    python main.py
To run the DQN algorithm, type in your terminal

    python3 ttt.py
To run the DDQN algorithm, type in your terminal

    python3 dd_ttt.py

In the last two versions, the game will ask you to play a match immediately after the training has finished and the data has been saved. During training it is possible to visualize the tensorboard. Run the following in your terminal

    tensorboard --logdir tensorflow_logs

## Results
The best result achieved for now was human-like performance after 100K episodes, in the Monte Carlo tree version. The result is achieved independently by the use of the neural network, but the purpose of the neural network will be evident in more complex games.

## External resources
https://web.stanford.edu/~surag/posts/alphazero.html

## Contact
Please write at lmambretti@ucdavis.edu if you need more information or if you are interested in developing more. (cause I am!)
