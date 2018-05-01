#import torch
#from torch.autograd import Variable

#import pyro
#import pyro.distributions as dist

import numpy as np
import time

# A class which given the parameters for a POMDP creates a corresponding
# belief MDP
class belief_mdp:

	def __init__(self, pomdp):
		self.pomdp = pomdp

	# Given a current belief, an index corresponding to an action, and an observation
	# return the updated belief

	# belief: np.array, action: action, obs:observation -> new_bel:np.array[float]
	def bel_update(self, belief, action, obs, prev_ask):

		# calculate the unnormalized new belief according to update equation(found on wikipedia POMDP page)
		new_bel = (self.pomdp.obs_prob(obs, self.pomdp.all_states(prev_ask),action)*
			np.dot(self.pomdp.transition(self.pomdp.all_states(prev_ask),self.pomdp.all_states(prev_ask),action), belief))

		# normalize the distribution and save the normalizer for the next part
		normalizer = np.sum(new_bel)
		new_bel = new_bel/normalizer

		return new_bel

	# Samples the model for an observation to update the belief state and returns the new belief and state
	#belief: np.array, action: action, state: state -> vector[float], state
	def bel_sampler(self, belief, action, state):
		if self.pomdp.reset(action):
			return [], 'complete'
		else:
			obs, state = self.pomdp.sample_obs(action, state)
			new_bel = self.bel_update(belief, action, obs, state)
			return new_bel, state

	# Returns the reward expected from an action given a belief
	# belief: vector[float], action: action -> float
	def reward_func(self, belief, action): 
		return np.dot(belief, self.pomdp.reward(self.pomdp.all_states((None,None)),action))
