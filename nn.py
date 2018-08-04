class NN:

    def __init__(self,lr):
        self.lr = lr
        """define structure here
        
        ***
        INPUT LAYER
        27 neurons
        """
        self.x = tf.placeholder(tf.float32,[None, 27])
        
        """
        HIDDEN LAYER
        18 neurons
        tanh
        """
        self.W1 = tf.Variable(tf.zeros([27,18])) #FIXME random uniform
        self.b1 = tf.Variable(tf.zeros([18])) #FIXME random uniform
        self.h1 = tf.tanh(tf.matmul(x, W1) + self.b1)
        
        """
        OUTPUT LAYER
        9 neurons
        tanh
        """
        self.W2 = tf.Variable(tf.zeros([18,9])) #FIXME random uniform
        self.b2 = tf.Variable(tf.zeros([9])) #FIXME random uniform
        self.y_ = tf.tanh(tf.matmul(self.h1, self.W2) + self.b2)

        """
        loss = mean squared error
        optimizer: adam
        or try the fastai library optimizers
        """

    def train(self,data):
        print("Initialize training NN")

    def run(self,input_data):
        return None
