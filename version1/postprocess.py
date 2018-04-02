import re
from collections import Counter
import sys
import os

DICT = Counter()

def createDict(file):
	text = open(file).read()
	words = re.findall(r'\w+', text.lower())
	# dictionary with count/frequency of word
	DICT = Counter(words)
	return DICT

def prob(word):
	# probability of word
	N = sum(DICT.values())
	return DICT[word]/N

def oneStep(word):
	# edits of distance 1 from the original word
	letters = 'abcdefghijklmnopqrstuvwxyz'
	splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
	deletes = [L + R[1:] for L, R in splits if R]
	transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
	replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
	inserts = [L + c + R for L, R in splits for c in letters]
	return set(deletes + transposes + replaces + inserts)

def twoStep(word):
	# edits of distance 2 from the original word
	return set(w2 for w1 in oneStep(word) for w2 in oneStep(w1))

def known(words):
	return set(w for w in words if w in DICT)

def candidates(word):
	return known([word]) or known(oneStep(word)) or known(twoStep(word)) or [word]

def correction(word):
	return max(candidates(word), key=prob)

if __name__ == "__main__":
	DICT = createDict('big.txt')
	# if (len(sys.argv) == 1):
	# 	file = input('filename: ')
	# else:
	# 	file = sys.argv[1]

	INPUT_DIR = "outputText"
	OUTPUT_DIR = "output_final"


	if not os.path.exists(OUTPUT_DIR):
		os.makedirs(OUTPUT_DIR)

	files_in_directory = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]

	for i, file in enumerate(files_in_directory):
		if ("acc" in file):
			continue
		input_path = INPUT_DIR + "/" + file
		print("Correcting file", input_path)
		text = open(input_path).read()
		words = re.findall(r'\w+', text.lower())
		output_path = OUTPUT_DIR + "/{:1}".format(file)
		output = open(output_path, "w")
		for word in words:
			word = word.lower()
			output.write(correction(word) + " ")
		print("Correction in {:1}".format(output_path))