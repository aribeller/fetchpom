# Spec of initial tiger pomdp:
states = ['SL', 'SR']
actions = ['L', 'LI', 'R']

# Transition probabilities (SxS'xA). Probability of going from State1 to State2 given action A
transition = [[[.5, 1.0, .5], [.5, 0, .5]], [[.5, 0, .5], [.5, 1, .5]]]

# Reward function (SxA)
reward = [[-100.0, -1.0, 10.0], [10.0,-1.0, -100.0]]

# Possible Observations
observe = ['TL', 'TR']

# Observation probabilities (OxSxA). Probability of observing o, given state s' and action a
obs_prob = [[[.5, .85, .5], [.5, .15, .5]], [[.5, .15, .5], [.5, .85, .5]]]

# Discount Factor:
disc = .9

# A function to update the belief state
def bel_update(belief, action, observation):
	new_bel = []
	# for each possible next state
	for next_state in range(len(belief)):
		state_sum = 0
		# for each possible current state
		for curr_state in range(len(states)):
			# sum T(next|curr,act)*bel(curr)
			state_sum += transition[curr_state][next_state][action]*belief[curr_state]

		# belief confidence for this state is O(next|obs,act)*state_sum
		bel_val = obs_prob[observation][next_state][action]*state_sum
		new_bel.append(bel_val)

	# normalize the distribution and save the normalizer for the next part
	normalizer = sum(new_bel)
	return [bel/normalizer for bel in new_bel], normalizer

# A function to check whether the belief check is close enough to the next one
def close_enough(check, post_bel, tol): return 1 if all([abs(a-b) < tol for a,b in zip(check,post_bel)]) else 0


def transition_prob(prior_bel, action, post_bel, normalizer, tol):
	total_prob = 0

	for obs in range(len(observe)):
		check = bel_update(prior_bel, action, obs)
		total_prob += close_enough(check, post_bel, tol) * normalizer

	return total_prob


def reward_func(belief, action):
	r = 0

	for i in range(len(states)):
		r += belief(i)*reward[i][action]

	return r







