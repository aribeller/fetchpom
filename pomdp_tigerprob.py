import numpy as np
from bs import *
from bel_mdp import belief_mdp

# Spec of initial tiger pomdp:
states = ['SL', 'SR']
actions = ['L', 'LI', 'R']

# Discount Factor:
disc = .9

# Transition probabilities (SxS'xA). Probability of going from State1 to State2 given action A
# State1 and State2 are vectors
def transition(state1, state2, action):
	transition_arr = np.array([[[.5, 1.0, .5], [.5, 0.0, .5]], [[.5, 0.0, .5], [.5, 1.0, .5]]])
	return transition_arr[state1, state2, action]

# Reward function (SxA)
# state is a vector
def reward(state, action):
	rewards = np.array([[-100.0, -1.0, 10.0], [10.0,-1.0, -100.0]])
	return rewards[state, action]

# Possible Observations
def observe(index):
	observations = ['TL', 'TR']
	return observations[index]

def observation_index(obs):
	observations = ['TL', 'TR']
	return observations.index(observation)

def all_obs():
	observations = ['TL', 'TR']
	return observations

# Observation probabilities (OxSxA). Probability of observing o, given state s' and action a
# obs:vector[observation], state:vector[state], action:int -> np.array[float]
def obs_prob(obs, state, action):
	obs_index = observation_index(obs)
	obs_prob_arr = np.array([[[.5, .85, .5], [.5, .15, .5]], [[.5, .15, .5], [.5, .85, .5]]])
	return obs_prob_arr[obs_index, state, action]






model = belief_mdp(states, actions, transition, reward, observe, obs_prob, disc)

def run_model():

	first_state_ind = np.random.binomial(1,.5)
	state = states[first_state_ind]
	print('Tiger Location:')
	print(state)
	print()
	next_act = 1
	bel = np.array([.5,.5])

	while next_act == 1:
		next_act = solve(.1, model.disc, 10, model, bel, state)
		print('next action:')
		print(actions[next_act])
		print()

		bel, _ = model.bel_update(bel, next_act, np.random.binomial(1,.15) if state == 'SL' else np.random.binomial(1,.85))

	return (state, next_act)


correct = 0.0
total = 0.0

for i in range(1):
	state, next_act = run_model()
	action = actions[next_act]
	print('iteration:', i)
	print(state, action)
	print()
	correct += (state == 'SL' and action == 'R') or (state == 'SR' and action == 'L')
	total += 1

print(correct/total)


