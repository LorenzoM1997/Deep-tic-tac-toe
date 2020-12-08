"""
Neural network
Made by Lorenzo Mambretti
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
print("tensorflow: ",tf.version.VERSION)
import numpy as np
import random
import const

class NN(Model):

    def __init__(self, lr = 0.00025, batch_size = 64):
        super(NN, self).__init__()
        self.d1 = Dense(21, activation='relu')
        self.d2 = Dense(15, activation='relu')
        self.d3 = Dense(9, activation='tanh')

        self.batch_size = batch_size

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)

"""
define here the training process and all necessary accessories"""
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

@tf.function
def train_step(model, data, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(data, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)

@tf.function
def test_step(model, data, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)

def train_model(mct, model, epochs, training_steps):

    # create batches
    input_batch = np.zeros((model.batch_size, 27))
    output_batch = np.zeros((model.batch_size, 9))
    action_matrix = np.zeros(9, dtype="int")

    for epoch in range(epochs):

        train_loss.reset_states()
        test_loss.reset_states()

        seed = random.randint(0, len(mct) - model.batch_size - 1)
        for b in range(model.batch_size):

            if not mct[seed + b].Child_nodes:
                # this node is not useful for training if it's not visited
                # don't count it and advance by 1 in the list
                b = b - 1
                # generate new point from where to look in the list
                seed = random.randint(0, len(mct) - model.batch_size - 1)

            else:
                input_batch[b] = mct[seed + b].board
                for a in range(9):
                    if mct[seed + b].Child_nodes[a] != None:
                        action_matrix[a] = mct[mct[seed + b].Child_nodes[a]].Q()
                    else:
                        action_matrix[a] = -1
                output_batch[b] = action_matrix

        train_step(model, input_batch, output_batch)
