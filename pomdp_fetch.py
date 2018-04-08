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
		# Response words
		self.response = self.affirm | self.negative
		# Smoothing parameter
		self.smooth = 0.2
		# prob of utterance
		self.utter_prob = .95
		# discout factor
		self.disc = 0.99

		# value of the previous utterance
		self.prev = None


	# Transition probabilities (SxS'xA). Probability of going from State1 to State2 given action A
	# state1: vector[state], state2:vector[state], action:action -> np.array[float]

	# This is inefficient as is. Right now instantiates the entire squared
	# state space when really only a small portion of those transitions are possible
	# def transition(self, state1, state2, action):
	# 	# initialize a matrix for transition from each state to each given state
	# 	out = np.zeros((len(state1), len(state2)))

	# 	# iterate across each from state then each to state
	# 	for i in range(len(state1)):
	# 		for j in range(len(state2)):
	# 			curr_state = state1[i]
	# 			next_state = state2[j]

	# 			# No transition should effect state
	# 			checks = curr_state[0] == next_state[0]

	# 			# If the transition involves a point action previous action should update
	# 			if action[0] == 'point':
	# 				checks = checks and next_state[1] == action[1]
	# 			# Otherwise previous action should stay the same
	# 			else:
	# 				checks = checks and curr_state[1] == next_state[1]

	# 			# If an entry meets all the checks update it's value
	# 			if checks:
	# 				out[i][j] = 1.0

	# 	return out


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
	# obs:vector[observation], state:vector[state], action:action -> np.array[float]

	# Should there really be a vector of observations? I think we only ever consider one?
	def obs_prob(self, obs, state, action):
		out = []

		words = obs.split()
		base = [word for word in words if word not in self.response]
		resp = [word for word in words if word in self.response]

		for s in state:
			out.append(self.prob_base(base, state)*self.prob_resp(resp,state))

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
				acc = acc * unigram(obj,word)

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

	def all_states(self): return [(i, self.prev) for i in self.items]

	def reset(self, action):
		if action[0] == 'pick':
			self.prev = None
			return True
		else:
			return False

	def init_bel(self): return np.array([1/len(self.items) for _ in self.items])

	def sample_obs(self, action, state):
		sample = ''
		if action[0] == 'wait':
			sample = self.sample_base(state)
		elif action[0] == 'point':
			word = self.sample_base(state)
			# If the pointed objected matches the desired object, randomly sample
			# from the affirmative array with 0.99 probability and 0.01 negative
			# and vice versa if they do not match
			resp = np.random.choice(np.random.choice([self.affirm, self.negative], 
					p=[0.99 if action[1] == state[0] else 0.01, 
					 0.01 if action[1] == state[0] else 0.99]))

			sample = word + ' ' + resp

		return sample




	def sample_base(self, state):
		return np.random.choice(self.all_words, [self.unigram(state[0],word) for word in self.all_words])

	def unigram(self, obj, word):
		print(obj)
		return (self.vocab[obj][word] + self.smooth)/(len(self.vocab[obj]) + self.smooth*self.v_size)



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
	# I don't think this should be provided. 
	choice = input('Choose an object (the AI will not know this):\n')
	cont = True
	while cont:
		try:
			choice = fm.item_names.index(choice)
			cont = False
		except:
			print('Please type an object name exactly')
	state = np.array([(choice, len(fm.items) + 1)])
	print('User Choice:')
	print(state[0][0])
	print()
	next_act = None # how does this even start?
	#what is the belief array for this mdp?
	bel = np.array([1/len(fm.items) for _ in range(len(fm.items))])
	while next_act is None or next_act[0] != 'pick':
		next_act = solve(0.1, fm.disc, 10, model, bel, state)	
		if next_act[0] == 'point':
			fm.prev = next_act[1]
			print('Is the object you want ' + fm.item_names[next_act[1]] + '?')
			obs = input('Response?')
			bel, _ = model.bel_update(bel, next_act, obs)
		elif next_act[0] == 'wait':
			obs = input('Describe to me which object you want')
			bel, _ = model.bel_update(bel, next_act, obs) 
	if next_act[1] == state[0]:
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






