import numpy as np

# Spec of initial tiger pomdp:
states = ['SL', 'SR']
actions = ['L', 'LI', 'R']

# Transition probabilities (SxS'xA). Probability of going from State1 to State2 given action A
transition = np.array([[[.5, 1.0, .5], [.5, 0.0, .5]], [[.5, 0.0, .5], [.5, 1.0, .5]]])

# Reward function (SxA)
reward = np.array([[-100.0, -1.0, 10.0], [10.0,-1.0, -100.0]])

# Possible Observations
observe = ['TL', 'TR']

# Observation probabilities (OxSxA). Probability of observing o, given state s' and action a
obs_prob = np.array([[[.5, .85, .5], [.5, .15, .5]], [[.5, .15, .5], [.5, .85, .5]]])

# Discount Factor:
disc = .9



from bs import *
from bel_mdp import belief_mdp



model = belief_mdp(states, actions, transition, reward, observe, obs_prob, disc)

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

