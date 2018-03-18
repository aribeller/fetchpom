import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist

# A class which given the parameters for a POMDP creates a corresponding
# belief MDP
class belief_mdp:

	def __init__(self, states, actions, transition, reward, observe, obs_prob, disc):
		self.states = states
		self.actions = actions
		self.transition = transition
		self.reward = reward
		self.observe = observe
		self.obs_prob = obs_prob
		self.disc = disc


	def bel_update(self, belief, action, observation):
		new_bel = []
		# for each possible next state
		for next_state in range(len(belief)):
			state_sum = 0
			# for each possible current state
			for curr_state in range(len(states)):
				# sum T(next|curr,act)*bel(curr)
				state_sum += self.transition[curr_state][next_state][action]*belief[curr_state]

			# belief confidence for this state is O(next|obs,act)*state_sum
			bel_val = self.obs_prob[observation][next_state][action]*state_sum
			new_bel.append(bel_val)

		# normalize the distribution and save the normalizer for the next part
		normalizer = sum(new_bel)
		return [bel/normalizer for bel in new_bel], normalizer

	def initial_state(self):
		return ((pyro.sample('state',
		 dist.Categorical(
		 	Variable(torch.FloatTensor([1/len(self.states) for s in self.states]))
		 	,self.states))), [1/len(self.states) for s in self.states])

	def bel_sampler(self, belief, action_ind, state_ind):
		obs = pyro.sample('obs', dist.Categorical(Variable(torch.FloatTensor(self.obs_prob[:,state_ind,action_ind])), self.states))
		obs_ind = self.observe.index(obs)
		new_bel = bel_update(belief, action_ind, obs_ind)
		return new_bel

	# A function to check whether the belief check is close enough to the next one
	def close_enough(check, post_bel, tol): return 1 if all([abs(a-b) < tol for a,b in zip(check,post_bel)]) else 0


	def transition_prob(prior_bel, action, post_bel, normalizer, tol):
		total_prob = 0

		for obs in range(len(observe)):
			check = bel_update(prior_bel, action, obs)
			total_prob += close_enough(check, post_bel, tol) * normalizer

		return total_prob







