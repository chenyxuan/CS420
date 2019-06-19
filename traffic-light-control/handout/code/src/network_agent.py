'''
@author: chenyxuan

Network agent from agent.py

'''

import numpy as np
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, Multiply, Add
from keras.models import Model, model_from_json, load_model
from keras.optimizers import RMSprop
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
import random
from keras.engine.topology import Layer
import os

from agent import Agent

class Selector(Layer):

    def __init__(self, select, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.select = select
        self.select_neuron = K.constant(value=self.select)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Selector, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.cast(K.equal(x, self.select_neuron), dtype="float32")

    def get_config(self):
        config = {"select": self.select}
        base_config = super(Selector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class NetworkAgent(Agent):

    @staticmethod
    def conv2d_bn(input_layer, index_layer,
                  filters = 16,
                  kernel_size = (3, 3),
                  strides = (1, 1)):
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3
        conv = Conv2D(filters = filters,
                      kernel_size = kernel_size,
                      strides = strides,
                      padding = 'same',
                      use_bias = False,
                      name = "conv{0}".format(index_layer))(input_layer)
        bn = BatchNormalization(axis = bn_axis, scale = False, name = "bn{0}".format(index_layer))(conv)
        act = Activation('relu', name="act{0}".format(index_layer))(bn)
        pooling = MaxPooling2D(pool_size=2)(act)
        x = Dropout(0.3)(pooling)
        return x

    @staticmethod
    def _cnn_network_structure(img_features):
        conv1 = NetworkAgent.conv2d_bn(img_features, 1, filters = 32, kernel_size = (8, 8), strides = (4, 4))
        conv2 = NetworkAgent.conv2d_bn(conv1, 2, filters = 16, kernel_size = (4, 4), strides = (2, 2))
        img_flatten = Flatten()(conv2)
        return img_flatten

    @staticmethod
    def _shared_network_structure(state_features, dense_d):
        hidden_1 = Dense(dense_d, activation = "sigmoid", name = "hidden_shared_1")(state_features)
        return hidden_1

    @staticmethod
    def _separate_network_structure(state_features, dense_d, num_actions, memo=""):
        hidden_1 = Dense(dense_d, activation = "sigmoid", name = "hidden_separate_branch_{0}_1".format(memo))(state_features)
        q_values = Dense(num_actions, activation = "linear", name = "q_values_separate_branch_{0}".format(memo))(hidden_1)
        return q_values

    @staticmethod
    def build_network_from_copy(network_copy):
        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure, custom_objects={"Selector": Selector})
        network.set_weights(network_weights)
        network.compile(optimizer = RMSprop(0.001),
                        loss = "mean_squared_error")
        return network


