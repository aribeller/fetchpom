import numpy as np

NUM_OBJS = 6

#state x is equal to i_d = floor(x/7) and i_r = x % 7 where index 6 is null
states = [i for i in range((NUM_OBJS + 1)**2)]
#action x for x up to and including NUM_OBJS - 1 is pick object x
#action x for NUM_OBS to 2*NUM_OBJS - 1 is point to object x - NUM_OBJS
actions = [i for i in range(NUM_OBJS*2 + 1)]

#reward function SxA
reward = np.zeros((len(states), len(actions)))

#define the rewards
for i in range((NUM_OBJS + 1)**2):
	i_d = math.floor(i/7)
	i_r = i % 7
	if i_d == 6:
		continue
	if i_r == 6:
		i_r = 'null'
	for j in range(NUM_OBJS*2 + 1):
		a = None
		if j <= NUM_OBJS - 1:
			a = ('pick', j)
		elif j <= 2*NUM_OBJS - 1:
			a = ('point', j - NUM_OBJS)
		else:
			a = ('wait', 0)
		
		if a[0] == 'pick':
			if a[1] == i_d:
				#correct
				reward[i, j] = 10
			else:
				#incorrect
				reward[i, j] = -12.5
		elif a[0] == 'point':
			reward[i, j] = -6
		elif a[0] == 'wait':
			reward = -1



