import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist

import numpy as np
import time

# A class which given the parameters for a POMDP creates a corresponding
# belief MDP
class belief_mdp:

	def __init__(self, states, actions, transition, reward, observe, all_obs, obs_prob, disc):
		self.states = states
		self.actions = actions
		self.transition = transition
		self.reward = reward
		self.observe = observe
		self.all_obs = all_obs
		self.obs_prob = obs_prob
		self.disc = disc


		# belief:np.array, action_ind:int, obs_ind:int -> new_bel:np.array
	def bel_update(self, belief, action_ind, obs):

		# calculate the unnormalized new belief according to update equation(found on wikipedia POMDP page)
		new_bel = self.obs_prob(obs,np.arange(len(self.states)),action_ind)*np.dot(self.transition(np.arange(len(self.states)),np.arange(len(self.states)),action_ind), belief)

		# normalize the distribution and save the normalizer for the next part
		normalizer = np.sum(new_bel)
		return new_bel/normalizer, normalizer

	# def initial_state(self):
	# 	return ((pyro.sample('state',
	# 	 dist.Categorical(
	# 	 	Variable(torch.FloatTensor([1/len(self.states) for s in self.states]))
	# 	 	,self.states))), [1/len(self.states) for s in self.states])

	def bel_sampler(self, belief, action_ind, state_ind):
		# print('action_ind')
		# print(action_ind)
		# print(type(action_ind))
		# print()
		# time.sleep(1)
		if action_ind != 1:
			state_ind = pyro.sample('state', dist.Bernoulli(Variable(torch.FloatTensor([.5]))))
			new_bel = np.array([.5,.5])
			return new_bel
		else:
			print('listen')
			# obs_allind = np.arange(2)
			obs = pyro.sample('obs', dist.Categorical(Variable(torch.FloatTensor(self.obs_prob(self.all_obs(),state_ind,action_ind))), self.all_obs()))
			# obs_ind = self.observe(obs)
			# print('prior_bel')
			# print(belief)
			# print()
			new_bel, _ = self.bel_update(belief, action_ind, obs)
			print('new_bel in scope:')
			print(new_bel)
			print()
			# print('new_bel')
			# print(new_bel)
			# print()

		print('new_bel out scope:')
		print(new_bel)
		print()
		return new_bel




	# A function to check whether the belief check is close enough to the next one
	# def close_enough(self, check, post_bel, tol): return 1 if all([abs(a-b) < tol for a,b in zip(check,post_bel)]) else 0


	def transition_prob(self, prior_bel, action, post_bel, normalizer, tol):
		total_prob = 0

		for obs in range(len(observe)):
			check = bel_update(prior_bel, action, obs)
			total_prob += close_enough(check, post_bel, tol) * normalizer

		return total_prob

	def reward_func(self, belief, action_ind): return np.dot(belief, self.reward(np.arange(len(self.states)),action_ind))







