import numpy as np
from bs import *
from bel_mdp import belief_mdp
import time


class TigerModel:
	def __init__(self):

		# Spec of initial tiger pomdp:
		self.states = ['SL', 'SR']
		self.actions = ['L', 'LI', 'R']
		self.observations = ['TL', 'TR']

		# Discount Factor:
		self.disc = .9

		#transition probability matrix state x state x action
		self.transition_arr = np.array([[[.5, 1.0, .5], [.5, 0.0, .5]], [[.5, 0.0, .5], [.5, 1.0, .5]]])

		#state x action rewards
		self.rewards = np.array([[-100.0, -1.0, 10.0], [10.0,-1.0, -100.0]])

		#observation probabilities with observation x state x action
		self.obs_prob_arr = np.array([[[.5, .85, .5], [.5, .15, .5]], [[.5, .15, .5], [.5, .85, .5]]])

	# Transition probabilities (SxS'xA). Probability of going from State1 to State2 given action A
	# state1: vector[state], state2:vector[state], action:int -> np.array[float]
	def transition(self, state1, state2, action):
		rows, cols = np.ix_(self.state_indices(state1),self.state_indices(state2))
		return self.transition_arr[rows, cols, action]

	# Reward function (SxA)
	# state: vector[state], action:int -> vector[float]
	def reward(self, state, action):
		return self.rewards[self.state_indices(state), action]

	# Possible Observations
	# index: int -> observation
	def observe(self, index):
		return self.observations[index]

	#obs: observation -> int
	def observation_index(self, obs):
		return self.observations.index(obs)

	#state: state -> int
	def state_index(self, state):
		return self.states.index(state)

	#obs: vector[observation] -> vector[int]
	def observation_indices(self, obs):
		obs_indices = []
		for ob in obs:
			obs_indices.append(self.observation_index(ob))
		return np.array(obs_indices)

	#states: vector[states] -> vector[int]
	def state_indices(self, states):
		s_indices = []
		for state in states:
			s_indices.append(self.state_index(state))
		return np.array(s_indices)

	#vector[observation]
	def all_obs(self):
		return self.observations

	#vector[state]
	def all_states(self):
		return self.states

	# Observation probabilities (OxSxA). Probability of observing o, given state s' and action a
	# obs:vector[observation], state:vector[state], action:int -> np.array[float]
	def obs_prob(self, obs, state, action):
		obs_inds = self.observation_indices(obs)
		state_inds = self.state_indices(state)
		
		return self.obs_prob_arr[obs_inds, state_inds, action]

	# A function to state whether the state of the game should  be reset given an action
	# Int -> Boolean

	def reset(self, action): action != 1




tm = TigerModel()

model = belief_mdp(tm)

def run_model():

	first_state_ind = np.random.binomial(1,.5)
	state = np.array([tm.states[first_state_ind]])
	print('Tiger Location:')
	print(state)
	print()
	next_act = 1
	bel = np.array([.5,.5])

	while next_act == 1:
		next_act = solve(.1, tm.disc, 10, model, bel, state)
		print('next action:')
		print(tm.actions[next_act])
		print()


		obs_ind = np.random.binomial(1,.15) if state == 'SL' else np.random.binomial(1,.85)
		obs = tm.observe(obs_ind)
		bel, _ = model.bel_update(bel, next_act, obs)


	return (state, next_act)


correct = 0.0
total = 0.0

for i in range(10):
	state, next_act = run_model()
	action = tm.actions[next_act]
	print('iteration:', i)
	print(state, action)
	print()
	correct += (state == 'SL' and action == 'R') or (state == 'SR' and action == 'L')
	total += 1

print(correct/total)


