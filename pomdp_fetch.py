import numpy as np
from bs import *
from bel_mdp import belief_mdp
import sys
from collections import defaultdict

# States are represented as 2-tuples where first index is actual object desired
# and second index is object the program asked about (None if no previous object)

# Actions are 2-tuples where first index is the type of action ('pick', 'point', 'wait')
# and the second index is the object selected or None if action is 'wait' 

class Fetch:
	# items: list[string]
	# vocab: dict[string -> set[string]]
	def __init__(self, items, vocab, item_names):

		# List of items for picking
		self.items = items
		# List of item names
		self.item_names = item_names
		# Dictionary of item to a dictionary of words associated to that item and a count for each word
		self.vocab = vocab
		# All the word types in the vocab + *UNK*
		self.all_words = list({word for v in self.vocab.values() for word in v} | {'*UNK*'})
		# vocab size (plus one for *UNK*)
		self.v_size = len(self.all_words)
		# All possible states as tuples. i is actual state and j is previous ask 
		self.states = [(i,j) for i in range(len(items)) for j in range(len(items)+1)]
		# All possible actions (pick for each item, point for each item, and wait)
		self.actions = [('pick', i) for i in range(len(items))] + [('point', j) for j in range(len(items))] + [('wait', len(items))]
		# Affirmative response words
		self.affirm = {'yes', 'yeah', 'sure', 'yup'}
		# Negative response words
		self.negative = {'no', 'nope', 'other', 'not'}
		# Left response words
		self.left = {'left'}
		# Right response words
		self.right = {'right'}
		# Position words
		self.positional = self.left | self.right

		# self.filter_cond = {'rightof', 'leftof', 'right', 'left'}
		self.filter_2arg = {'rightof', 'leftof', 'near'}

		self.base_mask = {'rightof': [0.0, 0.0, 0.8], 'leftof': [0.8, 0.0, 0.0], 'near': [0.8, 0.0, 0.8]}

		self.filter_1arg = {'right', 'left'}

		self.filter_cond = self.filter_2arg | self.filter_1arg

		# Response words
		self.response = self.affirm | self.negative
		# Smoothing parameter
		self.smooth = 0.2
		# prob of utterance
		self.utter_prob = .95
		# discout factor
		self.disc = 0.99

		# value of the previous utterance
		# self.prev = None


	# Second pass at the transition function. We can vastly simplify by representing
	# the belief as a distribution over what the agent is uncertain about.

	# Transition probabilities (SxS'xA). Probability of going from State1 to State2 given action A
	# state1: vector[state], state2:vector[state], action:action -> np.array[float]
	def transition(self, state1, state2, action):
		out = np.zeros((len(state1), len(state2)))

		# iterate across each from state then each to state
		for i in range(len(state1)):
			for j in range(len(state2)):
				curr_state = state1[i]
				next_state = state2[j]

				# No transition should effect state
				if curr_state[0] == next_state[0]:
					out[i][j] = 1.0

		return out


	# Reward function
	# state: vector[state], action:action -> vector[float]
	def reward(self, state, action):
		rewards = []

		act = action[0]
		item = action[1]

		# could probably do this with numpy ops instead of a loop
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
	# obs:observation, state:vector[state], action:action -> np.array[float]

	# Should there really be a vector of observations? I think we only ever consider one?
	def obs_prob(self, obs, state, action):
		out = []

		words = obs.split()
		base = [word for word in words if word not in self.response and word not in self.positional]
		resp = [word for word in words if word in self.response and word not in self.positional]
		# pos = [word for word in words if word in self.positional]
		# pos = []

		# for i in range(len(words)):
		# 	word = words[i]
		# 	if word in self.positional:
		# 		if i == len(words) - 1:
		# 			pos.append(word)
		# 		else:
		# 			if words[i + 1] != 'of':
		# 				pos.append(word)

		for s in state:
			out.append(self.prob_base(base, s)*self.prob_resp(resp,s))#*self.prob_pos(pos, s))

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
				acc = acc * self.unigram(obj,word)

			return acc


	# A helper to calculate the response probability of an utterance
	# list[string], state -> float
	def prob_resp(self, resp, state):
		obj = state[0]
		r_obj = state[1]

		cond_prob = [[.5,.5],[.99,.01],[.01,.99]]

		if r_obj == len(self.items):
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

	# pos: vector[string], state: state -> float
	# def prob_pos(self, pos, state):
	# 	obj = state[0]
	# 	r_obj = state[1]
	# 	lr = 1 if obj < len(self.items)/2 else 2
	# 	cond_prob = [[0.5, 0.5], [0.95, 0.05], [0.05, 0.95]]
	# 	cond = 0
	# 	if r_obj == len(self.items):
	# 		cond = 0
	# 	else:
	# 		cond = lr
	# 	if len(pos) == 0:
	# 		return 1.0 - self.utter_prob
	# 	else:
	# 		acc = self.utter_prob
	# 		for word in pos:
	# 			if word in self.left:
	# 				acc *= cond_prob[cond][0]
	# 			else:
	# 				acc *= cond_prob[cond][1]
	# 		return acc



	#Returns all possible states conditioned on the partially observable part of the state
	# int -> vector[state]
	def all_states(self, state): return [(i, state[1]) for i in self.items]

	# Returns whether the model should reset the game after this action, and resets if so
	# action: action -> boolean
	def reset(self, action):
		if action[0] == 'pick':
			self.prev = None
			return True
		else:
			return False

	# Returns the initial belief about the possible states
	# None -> vector[float]
	def init_bel(self): return np.array([1/len(self.items) for _ in self.items])

	# Returns a sampled observation based on the action that was taken from the given state
	# action:action, state: state -> observation
	def sample_obs(self, action, state):
		# print(state)
		sample = ''
		if action[0] != 'wait':
			# word = self.sample_base(state)
			# L = np.random.choice(['aff', 'neg'])
			# if L == 'aff':
			# 	resp = np.random.choice(list(self.affirm))
			# else:
			# 	resp = np.random.choice(list(self.affirm))

			# sample = word + ' ' + resp

			if action[0] == 'point':
				state = (state[0], action[1])
				#self.prev = action[1]
				word = self.sample_base(state)
				# If the pointed objected matches the desired object, randomly sample
				# from the affirmative array with 0.99 probability and 0.01 negative
				# and vice versa if they do not match
				# probs = [0.99 if action[1] == state[0] else 0.01, 0.01 if action[1] == state[0] else 0.99]
				# print(probs)
				L = np.random.choice(['aff', 'neg'], 
						p=[0.99 if action[1] == state[0] else 0.01,
						 0.01 if action[1] == state[0] else 0.99])

				if L == 'aff':
					resp = np.random.choice(list(self.affirm))
				else:
					resp = np.random.choice(list(self.negative))

				sample = word + ' ' + resp

			#Positional here
			# if np.random.rand() < 0.3:
			# 	pos_arg = np.random.choice(['left', 'right'],
			# 			p=[0.95 if state[0] < len(self.items)/2 else 0.05,
			# 			0.05 if state[0] < len(self.items)/2 else 0.95])
			# 	if pos_arg == 'left':
			# 		sample += ' ' + np.random.choice(list(self.left))
			# 	else:
			# 		sample += ' ' + np.random.choice(list(self.right))


		return sample, state

	# Helper to get a base utterance conditioned on the state
	# state:state -> string
	def sample_base(self, state):
		# print(state)
		return np.random.choice(self.all_words, p=[self.unigram(state[0],word) for word in self.all_words])

	# Helper to get the unigram probability of a word given an object
	# obj: item, word: string -> float
	def unigram(self, obj, word):
		# print(obj)
		return (self.vocab[obj][word] + self.smooth)/(sum(self.vocab[obj].values()) + self.smooth*self.v_size)

	#obj: int, index of the object, words: vector[string] -> float
	def obj_prob(self, obj, words):
		# cur_max = None
		# max_val = -1
		# for obj in self.items:
		# 	cur_prob = 1
		# 	for word in words:
		# 		cur_prob *= self.unigram(obj, word)
		# 	if cur_max is None or cur_prob > max_val:
		# 		cur_max = obj
		# 		max_val = cur_prob
		# return cur_max
		acc = 1
		for word in words:
			acc *= self.unigram(obj, word)
		return acc


	def two_arg(self, obs, keyword, bel):
		# for keyword in self.filter_cond:
		if keyword in obs:
			parts = obs.split(keyword)
			words = []
			for part in parts:
				words.append(part.split())
			#rows are left object on the phrase
			#cols are right object on the phrase
			p_mat = np.zeros((len(self.items), len(self.items)))
			for i in range(len(self.items)):
				for j in range(len(self.items)):
					if i != j:
						p_mat[i, j] = self.obj_prob(self.items[i], words[0])*self.obj_prob(self.items[j], words[1])
			"""if keyword == 'rightof':
				for col in range(len(self.items)):
					mask = np.zeros(len(self.items))
					for index in range(col + 1, len(self.items)):
						mask[index] = 0.8**index
					p_mat[:, col] = p_mat[:, col] * mask
				obj_ind = np.argmax(np.amax(p_mat, axis=1)) # this index is the index of the object we want to raise the belief of
				bel[obj_ind] += 3
				return bel/np.sum(bel)
			elif keyword == 'leftof':
				for col in range(len(self.items)):
					mask = np.zeros(len(self.items))
					for index in range(col):
						mask[index] = 0.8**((col-1)-index)
					p_mat[:, col] = p_mat[:, col] * mask
				obj_ind = np.argmax(np.amax(p_mat, axis=1))
				bel[obj_ind] += 3
				return bel/np.sum(bel)	
			"""
			for col in range(len(self.items)):
				mask = np.zeros(len(self.items))
				for index in range(len(self.items)):
					if index < col:
						mask[index] = self.base_mask[keyword][0]**abs(index - col)
					elif index == col:
						mask[index] = self.base_mask[keyword][1]
					elif index > col:
						mask[index] = self.base_mask[keyword][2]**abs(index - col)
				p_mat[:, col] = p_mat[:, col] * mask
			obj_ind = np.argmax(np.amax(p_mat, axis=1))
			bel[obj_ind] += 3
			return bel/np.sum(bel)	

	def one_arg(self, bel, keyword, threshold):
		e_bel = enumerate(bel)
		srt = sorted(e_bel, key=lambda x: x[1], reverse=True)

		total = 0.0
		ind = 0
		context = []
		while total < threshold:

			context.append(srt[ind])
			total += srt[ind][1]
			ind += 1

		if keyword == 'left':
			con_left = sorted(context, key=lambda x: x[0])
			for i in range(len(con_left)):
				x,y = con_left[i]
				con_left[i] = (x,y+.01**i)
			for x,y in con_left:
				bel[x] = y

		elif keyword == 'right':
			con_right = sorted(context, key=lambda x: x[0], reverse=True)
			for i in range(len(con_right)):
				x,y = con_right[i]
				con_right[i] = (x,y+.01**i)
			for x,y in con_right:
				bel[x] = y

		return bel/np.sum(bel)






	# bel: vector[float], obs: observation -> vector[float]
	def bel_bayes(self, bel, obs):
		obs = obs.replace('right of', 'rightof')
		obs = obs.replace('left of', 'leftof')

		for keyword in self.filter_cond:
			if keyword in obs and keyword in self.filter_2arg:
				bel = self.two_arg(obs, keyword, bel)
			elif keyword in obs and keyword in self.filter_1arg:
				bel = self.one_arg(bel, keyword, 0.7)
				# elif keyword == 'right':
				# 	mask = [.9**((len(self.items) - 1) - i) for i in range(len(self.items))]
				# 	new_bel = bel*mask
				# 	return new_bel/np.sum(new_bel)
				# elif keyword == 'left':
				# 	mask = [.9**(i) for i in range(len(self.items))]
				# 	new_bel = bel*mask
				# 	return new_bel/np.sum(new_bel)
		return bel







def run_model_auto(model, fm):
	choice = np.random.randint(len(fm.items))
	state = np.array([(choice, len(fm.items))])
	print('User Choice:')
	print(state[0])
	print()
	next_act = None # how does this even start?
	#what is the belief array for this mdp?
	bel = np.array([1/len(fm.items) for _ in range(len(fm.items))])
	while next_act is None or next_act[0] != 'pick':
		next_act = solve(0.1, fm.disc, 10, model, bel, state)	

def run_model(model, fm):
	print('Items:')
	print(fm.item_names)
	print()


	first_obs = input('Please describe the item you want:\n')
	bel = model.bel_update(np.array([1/len(fm.items) for _ in range(len(fm.items))]), ('wait',None), first_obs, (None,None))
	bel = fm.bel_bayes(bel, first_obs)
	# bel_unnorm = bayes_filter*bel
	# bel = bel_unnorm/sum(bel_unnorm)
	print('first belief update')
	print(bel)
	print()
	next_act = ('wait', None)

	while next_act is None or next_act[0] != 'pick':
		item = np.random.choice(fm.items, p=bel)
		# print('random item')
		# print(item)
		# print()
		next_act = solve(0.1, fm.disc, 10, model, bel, (np.random.choice(fm.items, p=bel),None))
		print(next_act)
		prev = next_act[1]
		if next_act[0] == 'point':
			print('Is the object you want ' + fm.item_names[next_act[1]] + '?\n')
			obs = input('Response?\n').lower()
			bel = model.bel_update(bel, next_act, obs, (None,prev))
			bel = fm.bel_bayes(bel, obs)
			print('successive belief updates')
			print(bel)
			print()
			#bel_unnorm = bayes_filter*bel
			#bel = bel_unnorm/sum(bel_unnorm)
		elif next_act[0] == 'wait':
			obs = input('Describe to me which object you want:\n')
			bel = model.bel_update(bel, next_act, obs, (None,prev))
			bel = fm.bel_bayes(bel, obs)
			#bel_unnorm = bayes_filter*bel
			#bel = bel_unnorm/sum(bel_unnorm)
	chosen = next_act[1]
	answer = input('Is the ' + fm.item_names[chosen] + ' your item?\nPlease answer "yes" or "no"\n')
	if answer == 'yes':
		print('Correct prediction')
		return 1
	else:
		print('Incorrect prediction')
		return -1

def main():
	if len(sys.argv) == 1:
		print('Please input object vocab')
		return
	if len(sys.argv) == 2 or len(sys.argv) == 3:
		vocab = dict()
		items = None
		item_names = []
		# Open the vocab document
		with open(sys.argv[1]) as f:
			lines = f.readlines()
			for i, line in enumerate(lines):
				# Split the line
				words = line.split()
				word_count = defaultdict(int)
				# get a word count dictionary for the particular item
				for word in words[1:]:
					word_count[word] += 1
				# Add that item/dictionary pairing to the vocab
				vocab[i] = word_count
				# Append the item name to the list of item names
				item_names.append(words[0])
			items = [i for i in range(len(lines))]
		fm = Fetch(items, vocab, item_names)
		model = belief_mdp(fm)
		if len(sys.argv) == 2:
			run_model_auto(model, fm)
		elif sys.argv[2] == 'manual':
			run_model(model, fm)
		
if __name__ == "__main__":
	main()






