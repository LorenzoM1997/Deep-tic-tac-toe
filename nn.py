"""
Neural network
Made by Lorenzo Mambretti
"""

import tensorflow as tf
from models.ConnectNN import ConnectNN
from models.TicTacToeNN import TicTacToeNN
print("tensorflow: ",tf.version.VERSION)
import numpy as np
import os
import random
import const

def get_model(dir=const.WEIGHTS_PATH):
    if const.GAME == 'TIC_TAC_TOE':
        nnet = TicTacToeNN()
    elif const.GAME == 'CONNECT4':
        nnet = ConnectNN()

    filename = os.path.join(dir, "checkpoint_" + const.GAME)
    try:
        nnet.load_weights(filename)
        print("get_model(): Load neural network weights")
    except:
        print("get_model():", filename, " not found")

    return nnet

def save_model(model, dir=const.WEIGHTS_PATH):
    filename = os.path.join(dir, "checkpoint_" + const.GAME)
    model.save_weights(filename)

"""
define here the training process and all necessary accessories"""
mse = tf.keras.losses.MeanSquaredError()
cross_entropy = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_policy_loss = tf.keras.metrics.Mean(name='train_policy_loss')
train_value_loss = tf.keras.metrics.Mean(name='train_value_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

@tf.function
def train_step(model, data, p_labels, v_labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        p_pred = model.policy(data)
        v_pred = model.value(data)
        value_loss = mse(v_labels, v_pred)
        policy_loss = mse(p_labels, p_pred)
        loss = value_loss + policy_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_policy_loss(value_loss)
    train_value_loss(policy_loss)

@tf.function
def test_step(model, data, labels):
    predictions = model.policy(data)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)

def train_model(data, p_labels, v_labels, model, epochs, batch_size = 64):

    print("train_model() started")

    # create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((data, p_labels, v_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    for epoch in range(epochs):

        train_policy_loss.reset_states()
        train_value_loss.reset_states()
        test_loss.reset_states()

        for step, (x_batch_train, p_batch, v_batch) in enumerate(train_dataset):
            train_step(model, x_batch_train, p_batch, v_batch)

        print(
            f'Epoch {epoch + 1}, '
            f'Policy Loss: {train_policy_loss.result()}, '
            f'Value Loss: {train_value_loss.result()}, '
          )
