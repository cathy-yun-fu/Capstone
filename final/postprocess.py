import re
from collections import Counter
import sys
import os

text = open('big.txt').read()
words = re.findall(r'\w+', text.lower())
# dictionary with count/frequency of word
DICT = Counter(words)
num_to_char = {'0':'o', '1':'l', '9':'a', '6':'c', '5':'s'}

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
	letters = 'abcdefghijklmnopqrstuvwxyz '
	splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
	deletes = [L + R[1:] for L, R in splits if R]
	replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
	inserts = [L + c + R for L, R in splits for c in letters]
	return set(deletes + replaces + inserts)

def twoStep(word):
	# edits of distance 2 from the original word
	return set(w2 for w1 in oneStep(word) for w2 in oneStep(w1))

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False

def replaceNums(word):
	if not is_number(word):
		for i in range(len(word)):
			word = word[:i] + num_to_char.get(word[i], word[i]) + word[i+1:]
	return word

def known(words):
	return set(w for w in words if w in DICT)

def candidates(word):
	word = replaceNums(word)
	return known([word]) or known(oneStep(word)) or known(twoStep(word)) or [word]

def correction(word):
	return max(candidates(word), key=prob)

def spellcheck(input_path, output_path, verbose):
	# DICT = createDict('big.txt')
	text = open(input_path).read()
	words = re.findall(r'\w+', text.lower())
	output = open(output_path, "w")
	corrected = ""
	for word in words:
		word = word.lower()
		cor_word = correction(word)
		output.write(cor_word + " ")
		corrected += cor_word + " "
	output.close()
	if verbose:
		print("Correction:", corrected)
	else:
		print("Correction saved in {:1}".format(output_path))

if __name__ == "__main__":
	INPUT_DIR = "output"
	OUTPUT_DIR = "output_final"
	files_in_directory = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]

	for i, file in enumerate(files_in_directory):
		input_path = INPUT_DIR + "/" + file
		output_path = OUTPUT_DIR + "/" + file
		spellcheck(input_path, output_path)