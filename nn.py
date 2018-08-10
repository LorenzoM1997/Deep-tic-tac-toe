import tensorflow as tf
from tensorflow import keras
print("tensorflow: ",tf.__version__)
import numpy as np
import random
import const

class NN:
    """        
    INPUT LAYER
    27 neurons
    """
    x = tf.placeholder(tf.float32,[None, 27], name='x')
    
    """
    HIDDEN LAYER 1
    21 neurons
    tanh
    """
    with tf.name_scope('Hidden_layer_1') as scope:
        W1 = tf.Variable(tf.random_uniform([27,27], minval = -1, maxval = 1), name='W1')
        b1 = tf.Variable(tf.random_uniform([27], minval = -1, maxval = 1), name='b1')
        h1 = tf.tanh(tf.matmul(x, W1) + b1)
    """
    HIDDEN LAYER 2
    15 neurons
    tanh
    """
    with tf.name_scope('Hidden_layer_2') as scope:
        W2 = tf.Variable(tf.random_uniform([27,21], minval = -1, maxval = 1), name='W2')
        b2 = tf.Variable(tf.random_uniform([21], minval = -1, maxval = 1), name='b2')
        h2 = tf.tanh(tf.matmul(h1, W2) + b2)
    """
    HIDDEN LAYER 3
    15 neurons
    tanh
    """
    with tf.name_scope('Hidden_layer_3') as scope:
        W3 = tf.Variable(tf.random_uniform([21,15], minval = -1, maxval = 1), name='W3')
        b3 = tf.Variable(tf.random_uniform([15], minval = -1, maxval = 1), name='b3')
        h3 = tf.tanh(tf.matmul(h2, W3) + b3)
    """
    OUTPUT LAYER
    9 neurons
    tanh
    """
    with tf.name_scope('Output_layer') as scope:
        W4 = tf.Variable(tf.random_uniform([15,9], minval = -1, maxval = 1))
        b4 = tf.Variable(tf.random_uniform([9], minval = -1, maxval = 1))
        y_ = tf.tanh(tf.matmul(h3, W4) + b4)

    y = tf.placeholder(tf.float32,[None, 9], name="y")

    """
    loss = mean squared error
    optimizer: adam
    or try the fastai library optimizers
    """
    loss = tf.losses.mean_squared_error(y_,y)

    def __init__(self, lr = 0.00025, batch_size = 64):
        self.batch_size = batch_size
        self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)
        
        # summaries
        tf.summary.scalar('loss', self.loss)
        self.summaries = tf.summary.merge_all()

        # start training session
        self.sess = tf.InteractiveSession()
        self.train_writer = tf.summary.FileWriter(const.cwd, self.sess.graph)
        tf.global_variables_initializer().run()

    def train(self, mct, iterations, training_steps):

        # create batches
        input_batch = np.zeros((self.batch_size, 27))
        output_batch = np.zeros((self.batch_size, 9))
        action_matrix = np.zeros(9, dtype="int")
        
        for i in range(iterations):
            seed = random.randint(0, len(mct) - self.batch_size - 1)
            for b in range(self.batch_size):

                if not mct[seed + b].Child_nodes:
                    # this node is not useful for training if it's not visited
                    # don't count it and advance by 1 in the list
                    b = b - 1
                    # generate new point from where to look in the list
                    seed = random.randint(0, len(mct) - self.batch_size - 1)
                    
                else:     
                    input_batch[b] = mct[seed + b].board
                    for a in range(9):
                        if mct[seed + b].Child_nodes[a] != None:
                            action_matrix[a] = mct[mct[seed + b].Child_nodes[a]].Q()
                        else:
                            action_matrix[a] = -1
                    output_batch[b] = action_matrix

            for j in range(training_steps):
                summary, _ = self.sess.run([self.summaries, self.train_step],
                                      feed_dict={ self.x: input_batch,
                                                  self.y: output_batch})
                self.train_writer.add_summary(summary, i)
        print("loss: ",self.sess.run(self.loss, feed_dict={self.x: input_batch,
                                                           self.y: output_batch}))

    def run(self,input_data):
        """
        PARAMS
        input_data  a 27d representation of a single board

        RETURN
        v           a 9d float array with the q values of all the actions
        """

        v = self.sess.run(self.y_, feed_dict={ self.x: [input_data]})
        return v
