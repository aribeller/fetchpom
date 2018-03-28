import numpy as np

# NUM_OBJS = 6

# #state x is equal to i_d = floor(x/7) and i_r = x % 7 where index 6 is null
# states = [i for i in range((NUM_OBJS + 1)**2)]
# #action x for x up to and including NUM_OBJS - 1 is pick object x
# #action x for NUM_OBS to 2*NUM_OBJS - 1 is point to object x - NUM_OBJS
# actions = [i for i in range(NUM_OBJS*2 + 1)]

# #reward function SxA
# reward = np.zeros((len(states), len(actions)))

# #define the rewards
# for i in range((NUM_OBJS + 1)**2):
# 	i_d = math.floor(i/7)
# 	i_r = i % 7
# 	if i_d == 6:
# 		continue
# 	if i_r == 6:
# 		i_r = 'null'
# 	for j in range(NUM_OBJS*2 + 1):
# 		a = None
# 		if j <= NUM_OBJS - 1:
# 			a = ('pick', j)
# 		elif j <= 2*NUM_OBJS - 1:
# 			a = ('point', j - NUM_OBJS)
# 		else:
# 			a = ('wait', 0)
		
# 		if a[0] == 'pick':
# 			if a[1] == i_d:
# 				#correct
# 				reward[i, j] = 10
# 			else:
# 				#incorrect
# 				reward[i, j] = -12.5
# 		elif a[0] == 'point':
# 			reward[i, j] = -6
# 		elif a[0] == 'wait':
# 			reward = -1


class fetch:
	# items: list[string]
	# vocab: dict[string -> set[string]]
	def __init__(self, items, vocab):

		# List of items for picking
		self.items = items
		# Dictionary of item to its associated words
		self.vocab = vocab
		# vocab size
		self.v_size = sum([len(v) for v in self.vocab.values()])
		# All possible states as tuples. i is actual state and j is previous ask 
		self.state = [(i,j) for i in range(len(items)) for j in range(len(items)+1)]
		# All possible actions (pick for each item, point for each item, and wait)
		self.actions = [('pick', i) for i in range(len(items))] + [('point', j) for j in range(len(items))] + [('wait', len(items))]
		# Affirmative response words
		self.affirm = {'yes', 'yeah', 'sure', 'yup'}
		# Negative response words
		self.negative = {'no', 'nope', 'other', 'not'}
		# Response words
		self.response = self.affirm | self.negative
		# Smoothing parameter
		self.smooth = 0.2
		# prob of utterance
		self.utter_prob = .95


	# Transition probabilities (SxS'xA). Probability of going from State1 to State2 given action A
	# state1: vector[state], state2:vector[state], action:int -> np.array[float]

	# This is inefficient as is. Right now instantiates the entire squared
	# state space when really only a small portion of those transitions are possible
	def transition(self, state1, state2, action):
		out = np.zeros((len(state1), len(state2)))

		for i in range(len(state1)):
			for j in range(len(state2)):
				curr_state = state1[i]
				next_state = state2[j]

				checks == True

				checks = checks and curr_state[0] == next_state[0] and next_state[1] == action[1]

				if checks:
					out[i][j] = 1.0

		return out


	# Reward function
	# state: vector[state], action:int -> vector[float]
	def reward(self, state, action):
		rewards = []

		act = action[0]
		item = action[1]

		for s in state:
			if act == 'pick' and item == s[0]:
				rewards.append(10.0)
			elif act == 'pick' and item != state[0]:
				rewards.append(-12.5)
			elif act == 'point':
				rewards.append(-6.0)
			else:
				rewards.append(-1.0)

		return np.array(rewards)

	# Observation probabilities (OxSxA). Probability of observing o, given state s' and action a
	# obs:vector[observation], state:vector[state], action:int -> np.array[float]

	# Should there really be a vector of observations? I think we only ever consider one?
	def obs_prob(self, obs, state, action):
		out = []

		words = obs.split()
		base = {word for word in words if word not in self.response}
		resp = {word for word in words if word in self.response}

		for s in state:
			out.append(prob_base(words, base)*prob_resp(words,resp))

		return np.array(out)

	# A helper to calculate the base probability of an utterance
	# list[string], state -> float
	def prob_base(self, base, state):
		obj = state[0]

		if len(base) == 0:
			return 1.0 - self.utter_prob
		else:
			acc = self.utter_prob
			for word in base:
				acc = acc * (((1.0 if word in self.vocab[obj] else 0.0) + self.smooth)/
					(len(self.vocab[obj]) + self.smooth*self.v_size))

			return acc


	# A helper to calculate the response probability of an utterance
	# list[string], state -> float
	def prob_resp(self, resp, state):
		obj = state[0]
		r_obj = state[1]

		cond_prob = [[.5,.5],[.99,.01],[.01,.99]]

		if r_obj == len(items):
			cond = 0
		elif obj == r_obj:
			cond = 1
		else:
			cond = 2

		if len(resp) == 0:
			return 1.0 - self.utter_prob
		else:
			acc = self.utter_prob
			for word in resp:
				if word in self.affirm:
					sentiment = 0
				else:
					sentiment = 1
				acc = acc*cond_prob[cond][sentiment]
			return acc



			






