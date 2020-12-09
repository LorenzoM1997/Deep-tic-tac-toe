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

    def __init__(self):
        super(NN, self).__init__()
        self.d1 = Dense(21, activation='relu')
        self.d2 = Dense(15, activation='relu')
        self.d3 = Dense(9, activation='tanh')

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

def train_model(data, labels, model, epochs, batch_size = 64):

    print("train_model() started")

    # create batches
    input_batch = np.empty((batch_size, 27))
    output_batch = np.empty((batch_size, 9))
    action_matrix = np.empty(9, dtype="int")

    train_dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    for epoch in range(epochs):

        train_loss.reset_states()
        test_loss.reset_states()

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            train_step(model, input_batch, output_batch)
