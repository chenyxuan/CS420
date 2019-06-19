# -*- coding: utf-8 -*-

'''
@author: chenyxuan

Control agent, mainly choosing actions

'''

class State(object):
	D_NUM_OF_VEHICLES = (12,)
	D_NUM_OF_WAITING_VEHICLES = (12,)
	D_CUR_PHASE = (1,)

	def __init__(self,
		num_of_vehicles,
		num_of_waiting_vehicles,
		cur_phase):

		self.num_of_vehicles = num_of_vehicles
		self.num_of_waiting_vehicles = num_of_waiting_vehicles
		self.cur_phase = cur_phase

		return

class Agent(object):

	def __init__(self, num_phases):
		self.num_phases = num_phases
		self.memory = []
		return
