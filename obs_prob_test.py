from collections import defaultdict

with open('setup1.txt') as f:
	file = [line.split() for line in f.readlines()]


  # vocab = {line[0]:line[1:] for line in file}

vocab = {}
words = set()

for line in file:
	word_counter = defaultdict(int)
	vocab[line[0]] = word_counter
	for word in line[1:]:
		word_counter[word] += 1
		words.add(word)


# v_size = sum([sum(x.values()) for x in vocab.values()])
# v_size = sum(len(x.keys()) for x in vocab.values())
v_size = len(words)

def unigram(word, obj):
	return (vocab[obj][word] + .2)/(sum(vocab[obj].values()) + .2*v_size)
	# return (vocab[word])/(v_size)


# words = [word for L in vocab.values() for word in L.keys()]

prob_check = sum(unigram(w,'black_bowl_2') for w in words)
print(prob_check)