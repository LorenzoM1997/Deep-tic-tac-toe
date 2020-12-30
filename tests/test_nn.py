import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import numpy as np
import const
from models.ConnectNN import ConnectNN
from models.TicTacToeNN import TicTacToeNN
from nn import save_model, get_model, train_model

np.random.seed(0)

# Helper functions
# ----------------------------------------------------------------------------
def assert_valid_model(model, input_shape, output_shape):
    assert model is not None
    x = np.random.random(input_shape)
    yp = model.policy(x)
    assert yp.shape == output_shape
    assert np.max(yp) <= 1
    assert np.min(yp) >= -1
    yv = model.value(x)
    assert yv.shape == (output_shape[0],1)
    assert np.max(yp) <= 1
    assert np.min(yp) >= -1

def assert_valid_model_TTT(model):
    assert_valid_model(model, (10,27), (10,9))

def assert_valid_model_connect(model):
    assert_valid_model(model, (10,126), (10,7))

# Test functions
# ----------------------------------------------------------------------------
def test_TicTacToeNN():
    nnet = TicTacToeNN()
    assert_valid_model_TTT(nnet)

def test_ConnectNN():
    nnet = ConnectNN()
    assert_valid_model_connect(nnet)

def test_save_model(tmpdir):
    #check that the save_model function is working
    nnet = TicTacToeNN()
    save_model(nnet, tmpdir)
    assert_valid_model_TTT

def test_get_model():
    # try that get_model returns something valid
    # the aim of this test is not to check every individual model, there are
    # other tests for that
    nnet = get_model()
    if const.GAME == 'TIC_TAC_TOE':
        assert_valid_model_TTT(nnet)
    else:
        assert_valid_model_connect(nnet)

def test_get_saved_model(tmpdir):
    # verify that we can open a saved model
    nnet= get_model(tmpdir)
    save_model(nnet, tmpdir)
    nnet2 = get_model(tmpdir)
    assert nnet is not None

def test_train_model():
    n_entries = 10
    epochs = 10

    # get some random data
    nnet = TicTacToeNN()
    data = np.random.random((n_entries, 27))
    p_labels = np.random.random((n_entries, 9))
    v_labels = np.random.random((n_entries, 1))

    # train model
    train_model(data, p_labels, v_labels, nnet, epochs)
    assert_valid_model_TTT(nnet)
