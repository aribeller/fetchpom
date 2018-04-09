import math
import numpy as np
import time

def expectedQ(depth, width, disc, bmdp, belief, state):
	if depth == 0:
		return [(action, 0.0) for action in bmdp.pomdp.actions]
	else:
		qs = []
		for action in bmdp.pomdp.actions:
			qval = bmdp.reward_func(belief, action)
			new_beliefs = []
			new_states = []
			for k in range(width):
				new_bel, new_state = bmdp.bel_sampler(belief, action, state)
				new_beliefs.append(new_bel)
				new_states.append(new_state)
			for i in range(len(new_beliefs)):
				qval += 1/width*disc*np.sum(expectedV(depth - 1, width, disc, bmdp, new_beliefs[i], new_states[i]))
			qs.append((action, qval))
		return qs

def expectedV(depth, width, disc, bmdp, belief, state):
	temp = np.array([q[1] for q in expectedQ(depth, width, disc, bmdp, belief, state)])
	return np.max(temp)

def solve(epsilon, gamma, rmax, bmdp, belief, state):
	vmax = rmax/(1 - gamma)
	lamb = (epsilon*(1 - gamma)**2)/4
	h = 2
	# h = int(math.ceil(math.log(lamb/vmax, gamma)))
	c = 10
	# c = int(vmax**2/(lamb**2)*(2*h*math.log(len(bmdp.actions)*h*vmax**2/(lamb**2)) + math.log(rmax/lamb)))
	qs = expectedQ(h, c, gamma, bmdp, belief, state)
	print('qs')
	print(qs)
	print()
	return qs[np.argmax(np.array([q[1] for q in qs]))][0]



