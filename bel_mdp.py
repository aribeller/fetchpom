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

	# belief: np.array, action: action, obs:observation -> new_bel:np.array
	def bel_update(self, belief, action, obs, prev_ask):

		# calculate the unnormalized new belief according to update equation(found on wikipedia POMDP page)
		new_bel = (self.pomdp.obs_prob(obs, self.pomdp.all_states(prev_ask),action)*
			np.dot(self.pomdp.transition(self.pomdp.all_states(prev_ask),self.pomdp.all_states(prev_ask),action), belief))

		# normalize the distribution and save the normalizer for the next part
		normalizer = np.sum(new_bel)
		new_bel = new_bel/normalizer

		return new_bel



	#belief: np.array, action: action, state: state
	def bel_sampler(self, belief, action, state):
		if self.pomdp.reset(action):
			belief = self.pomdp.init_bel()
			all_states = self.pomdp.all_states((None,None))
			index = np.random.randint(len(all_states))
			state = all_states[index]
			return belief, state
		else:
			obs, state = self.pomdp.sample_obs(action, state)
			new_bel = self.bel_update(belief, action, obs, state)
			return new_bel, state


	# def bel_sampler(self, belief, action, state):
		# if self.pomdp.reset(action):
		# 	state_ind = pyro.sample('state', dist.Bernoulli(Variable(torch.FloatTensor([.5]))))
		# 	new_bel = np.array([.5,.5])
		# 	return new_bel
		# else:
		# 	obs = pyro.sample('obs', 
		# 		dist.Categorical(Variable(torch.FloatTensor(self.pomdp.obs_prob(self.pomdp.all_obs(),state,action))), 
		# 		self.pomdp.all_obs()))
		# 	new_bel, _ = self.bel_update(belief, action, obs)

		# return new_bel


	def reward_func(self, belief, action): 
		return np.dot(belief, self.pomdp.reward(self.pomdp.all_states((None,None)),action))







