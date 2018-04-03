import math
import numpy as np
import time

def expectedQ(depth, width, disc, model, belief, state):
	# print('depth')
	# print(depth)
	# print()
	if depth == 0:
		return np.zeros(len(model.pomdp.actions))
	else:
		qs = []
		for action in model.pomdp.actions:
			# print('action')
			# print(range(len(model.pomdp.actions)))
			# print(action)
			# print()
			qval = model.reward_func(belief, action)
			new_beliefs = []
			for k in range(width):
				new_bel = model.bel_sampler(belief, action, state)
				# print('new_bel')
				# print(new_bel)
				# print()
				new_beliefs.append(new_bel)
			for new_belief in new_beliefs:
				qval += 1/width*disc*np.sum(expectedV(depth - 1, width, disc, model, new_belief, state))
			qs.append((action, qval))
		return qs

def expectedV(depth, width, disc, model, belief, state):
	return np.max(np.array([q[1] for q in expectedQ(depth, width, disc, model, belief, state)]))

def solve(epsilon, gamma, rmax, model, belief, state):
	vmax = rmax/(1 - gamma)
	lamb = (epsilon*(1 - gamma)**2)/4
	h = 3
	# h = int(math.ceil(math.log(lamb/vmax, gamma)))
	c = 8
	# c = int(vmax**2/(lamb**2)*(2*h*math.log(len(model.actions)*h*vmax**2/(lamb**2)) + math.log(rmax/lamb)))
	qs = expectedQ(h, c, gamma, model, belief, state)
	print('qs')
	print(qs)
	print()
	return qs[np.argmax(np.array([q[1] for q in qs]))][0]



