import math
import numpy as np
import pomdp_tigerprob as hobbes

def expectedQ(depth, width, disc, model, state):
	qs = []
	for action in model.possible_actions:
		qval = model.reward_func(state, action)
		new_states = model.transition(state, action)
		for new_state in new_states:
			qval += 1/width*disc*expectedV(depth - 1, width, disc, model, new_state)
		qs.append(qval)
	return qs

def expectedV(depth, width, disc, model, state):
	return np.argmax(np.array(expectedQ(depth, width, disc, model, state)))

def solve(epsilon, gamma, rmax, model, state):
	vmax = rmax/(1 - gamma)
	lamb = (epsilon*(1 - gamma)**2)/4
	h = int(math.ceil(math.log(lamb/vmax, gamma)))
	c = vmax**2/(lamb**2)*(2*h*math.log(len(model.all_actions())*h*vmax**2/(lamb**2)) + math.log(rmax/lamb))
	qs = expectedQ(h, c, gamma, model, state)
	return np.argmax(np.array(qs))