 # -*- coding: utf-8 -*-

'''
@author: chenyxuan

Deep reinforcement learning agent

'''

import numpy as np
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, Multiply, Add
from keras.models import Model, model_from_json, load_model
from keras.optimizers import RMSprop
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping
import random
import os

from network_agent import NetworkAgent, Selector
from agent import State

class DeepLightAgent(NetworkAgent):

    feature_list = [
        'num_of_vehicles',
        'num_of_waiting_vehicles',
        'cur_phase'
    ]

    def __init__(self,
                 num_phases):
        super(DeepLightAgent, self).__init__(
            num_phases = num_phases
        )

        self.q_network = self.build_network()
        self.q_network_bar = self.build_network_from_copy(self.q_network)
        self.q_bar_outdated = 0

        self.memory = []
        return

    def build_network(self):

        dic_input_node = {}
        for feature_name in self.feature_list :
            dic_input_node[feature_name] = Input(shape = getattr(State,  "D_" + feature_name.upper()), name = "input_" + feature_name)

        dic_flatten_node = {}
        for feature_name in self.feature_list :
            if len(getattr(State, "D_" + feature_name.upper())) > 1 :
                dic_flatten_node[feature_name] = self._cnn_network_structure(dic_input_node[feature_name])
            else:
                dic_flatten_node[feature_name] = dic_input_node[feature_name]

        list_all_flatten_feature = []
        for feature_name in self.feature_list :
            list_all_flatten_feature.append(dic_flatten_node[feature_name])
        all_flatten_featrue = concatenate(list_all_flatten_feature, axis = 1, name = "all_flatten_featrue")

        shared_dense = self._shared_network_structure(all_flatten_featrue, 20)

        list_selected_q_values = []
        for phase in range(self.num_phases):
            locals()["q_values_{0}".format(phase)] = self._separate_network_structure(
                shared_dense, 20, self.num_phases, memo=phase)
            locals()["selector_{0}".format(phase)] = Selector(
                phase, name="selector_{0}".format(phase))(dic_input_node["cur_phase"])
            locals()["q_values_{0}_selected".format(phase)] = Multiply(name="multiply_{0}".format(phase))(
                [locals()["q_values_{0}".format(phase)],
                 locals()["selector_{0}".format(phase)]]
            )
            list_selected_q_values.append(locals()["q_values_{0}_selected".format(phase)])
        q_values = Add()(list_selected_q_values)

        network = Model(inputs = [dic_input_node[feature_name] for feature_name in self.feature_list], outputs = q_values)
        network.compile(optimizer = RMSprop(lr = 0.001), loss = "mean_squared_error")
#        network.summary()

        return network

    def remeber(self, state, action, reward, next_state):
        self.memory.append([state, action, reward, next_state])
        return

    def forget(self):
        if(len(self.memory) > 1000) :
            self.memory = self.memory[-1000:]
        return


    def train_network(self, Xs, Y):
        epochs = 3
        batch_size = min(20, len(Y))

        self.q_network.fit(Xs, Y, batch_size = batch_size, epochs = epochs,
                                  verbose = False, validation_split = 0.3)

    def _sample_memmory(self):
        sample_size = min(len(self.memory), 300)
        sampled_memory = random.sample(self.memory, sample_size)
        return sampled_memory

    def get_sample(self, memory_slices, dic_arrays, Y, is_pretrain):
        gamma = 0.9
        for memory_slice in memory_slices :
            state, action, reward, next_state = memory_slice
            for feature_name in self.feature_list :
                dic_arrays[feature_name].append(getattr(state, feature_name)[0])
            target = self.q_network.predict(self.convert_state_to_input(state))
            if is_pretrain :
                target[0][action] = reward / (1 - gamma)
            else :
                target[0][action] = reward + gamma * self._get_next_estimated_reward(next_state)
            Y.append(target[0])

        return dic_arrays, Y

    def update_network(self, current_time, is_pretrain):
        if(current_time < 200) :
            return

        dic_state_feature_arrays = {}
        for feature_name in self.feature_list :
            dic_state_feature_arrays[feature_name] = []
        Y = []

        sampled_memory = self._sample_memmory()
        dic_state_feature_arrays, Y = self.get_sample(sampled_memory, dic_state_feature_arrays, Y, is_pretrain)

        Xs = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in self.feature_list]
        Y = np.array(Y)

        self.q_bar_outdated += 1
        self.train_network(Xs, Y)
        self.forget()
        return

    def choose(self, state, certain=False):
        q_values = self.q_network.predict(self.convert_state_to_input(state))

        if (not certain) and random.random() <= 0.01 :
            action = random.randrange(len(q_values[0]))
        else:
            action = np.argmax(q_values[0])

        return action

    def convert_state_to_input(self, state):
        return [getattr(state, feature_name) for feature_name in self.feature_list]

    def _get_next_estimated_reward(self, next_state):
        next_estimated_reward = np.max(self.q_network_bar.predict(
            self.convert_state_to_input(next_state))[0])
        return next_estimated_reward

    def update_network_bar(self):
        if self.q_bar_outdated >= 20:
            self.q_network_bar = self.build_network_from_copy(self.q_network)
            self.q_bar_outdated = 0

    def load_model(self, name):
        self.q_network = load_model("model/dqn_{0}.h5".format(name), custom_objects = {'Selector': Selector})

    def save_model(self, name):
        self.q_network.save("model/dqn_{0}.h5".format(name))
