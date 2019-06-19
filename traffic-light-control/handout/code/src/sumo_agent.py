# -*- coding: utf-8 -*-

'''
@author: chenyxuan

For interaction.

'''

import numpy as np
from agent import State
import cityflow

class SumoAgent:

    def __init__(self):
        config = "data/2x2/config.json"
        self.eng = cityflow.Engine(config, thread_num=4)
        self.cur_phase = {
            'intersection_1_2': 0,
            'intersection_1_1': 0,
            'intersection_2_1': 0,
            'intersection_2_2': 0
        }
        self.lanes_dict = {
            'intersection_1_2': [
                "road_2_2_2",
                "road_1_3_3",
                "road_0_2_0",
                "road_1_1_1"
            ],
            'intersection_1_1': [
                "road_1_2_3",
                "road_0_1_0",
                "road_1_0_1",
                "road_2_1_2"
            ],
            'intersection_2_1': [
                "road_1_1_0",
                "road_2_0_1",
                "road_3_1_2",
                "road_2_2_3"
            ],
            'intersection_2_2': [
                "road_2_1_1",
                "road_3_2_2",
                "road_2_3_3",
                "road_1_2_0"
            ]

        }

        return

    def get_state(self, lane_list, cur_phase):
        lane_vehicle_count = self.eng.get_lane_vehicle_count()  # return a dict, {lane_id: lane_count, ...}
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()  # return a dict, {lane_id: lane_waiting_count, ...}

        num_of_vehicles = []
        num_of_waiting_vehicles = []

        for lane in lane_list :
            for i in range(3) :
                sublane = (lane + "_{0}".format(i))
                num_of_vehicles.append(lane_vehicle_count[sublane])
                num_of_waiting_vehicles.append(lane_waiting_vehicle_count[sublane])

        return State(
            num_of_vehicles = np.reshape(np.array(num_of_vehicles), newshape = (1, 12)),
            num_of_waiting_vehicles = np.reshape(np.array(num_of_waiting_vehicles), newshape = (1, 12)),
            cur_phase = np.reshape(np.array([cur_phase]), newshape = (1, 1))
        )

    def get_observation(self):
        state = {}
        for key, value in self.lanes_dict.items() :
            state[key] = self.get_state(value, self.cur_phase[key])
        return state

    def calcu_reward(self, lanes, last_state, cur_state):
        res = 0
        for _lane in lanes :
            for i in range(3) :
                lane = _lane + "_{0}".format(i)
                res += cur_state[1][lane] * (-0.25)
                vehicle_leaving = 0
                for vehicle in last_state[2][lane] :
                    if not vehicle in cur_state[2][lane] :
                        vehicle_leaving += 1
                res += vehicle_leaving * 1
        return res

    def take_action(self, action_dict):
        reward = {}

        last_lane_vehicle_count = self.eng.get_lane_vehicle_count()  # return a dict, {lane_id: lane_count, ...}
        last_lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()  # return a dict, {lane_id: lane_waiting_count, ...}
        last_lane_vehicles = self.eng.get_lane_vehicles()  # return a dict, {lane_id: [vehicle1_id, vehicle2_id, ...], ...}
        last_vehicle_speed = self.eng.get_vehicle_speed()  # return a dict, {vehicle_id: vehicle_speed, ...}
        last_vehicle_distance = self.eng.get_vehicle_distance()  # return a dict, {vehicle_id: vehicle_distance, ...}
        last_state = [last_lane_vehicle_count, last_lane_waiting_vehicle_count, last_lane_vehicles, last_vehicle_speed, last_vehicle_distance]

        for key, value in action_dict.items() :
            self.eng.set_tl_phase(key, value)
            self.cur_phase[key] = value
        self.eng.next_step()

        cur_lane_vehicle_count = self.eng.get_lane_vehicle_count()  # return a dict, {lane_id: lane_count, ...}
        cur_lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()  # return a dict, {lane_id: lane_waiting_count, ...}
        cur_lane_vehicles = self.eng.get_lane_vehicles()  # return a dict, {lane_id: [vehicle1_id, vehicle2_id, ...], ...}
        cur_vehicle_speed = self.eng.get_vehicle_speed()  # return a dict, {vehicle_id: vehicle_speed, ...}
        cur_vehicle_distance = self.eng.get_vehicle_distance()  # return a dict, {vehicle_id: vehicle_distance, ...}
        cur_state = [cur_lane_vehicle_count, cur_lane_waiting_vehicle_count, cur_lane_vehicles, cur_vehicle_speed, cur_vehicle_distance]

        for key, value in action_dict.items() :
            reward[key] = self.calcu_reward(self.lanes_dict[key], last_state, cur_state)
        return reward

    def get_time(self):
        return self.eng.get_current_time()

    def get_score(self):
        return self.eng.get_score()