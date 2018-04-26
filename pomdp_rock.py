import numpy as np
from bs import *
from bel_mdp import belief_mdp
import sys

class Rock:

	def __init__(self, grid_h, grid_w, rocks, robot_pos):

		#The grid where the robot moves
		self.grid = np.zeros((grid_h, grid_w))

		self.state = (robot_pos, rocks)

		self.actions = [('North', None), ('South', None), ('East', None), ('West', None), ('Sample', None)] +
			[('Check', i) for i in range(len(rocks))]

		self.disc = 0.99

	#obs: observation, state: vector[state], action: action -> np.array[float]
	def obs_prob(self, obs, state, action):
		return [0]

	#Gets all possible states from current state
	def all_states(self, state):
		return [0]

	#state1: vector[state], state2: vector[state], action: action -> np.array[float]
	def transition(self, state1, state2, action):
		return [0]
	




