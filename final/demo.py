import os
import sys
import cv2
import os
import shutil
import glob
from keras.models import model_from_yaml
from os.path import basename
import pickle
import re
import argparse
from PIL import Image
from collections import Counter

# our files
import preprocess as prep
import character_segmentation as seg
import character_recognition as rec
import postprocess as post
import test_model as rnn

# paths
benchmark_dir = "BENCHMARKS/"
input_dir = "INPUT/"
output_dir = "OUTPUT/"
pre_process_path = "PREPROCESS/"
char_recognition_path = "CNN/"
rnn_path = "RNN/"
test_dir = 'test_data/'

# starting intro
def start_program():
	print("╔╗░╔╗░░░░░░░╔╗░░░░░░░╔╗░░░░░░░░╔═══╗╔════╗░░░░╔╗░")
	print("║║░║║░░░░░░░║║░░░░░░╔╝╚╗░░░░░░░║╔═╗║║╔╗╔╗║░░░╔╝╚╗")
	print("║╚═╝╠══╦═╗╔═╝╠╗╔╗╔╦═╬╗╔╬╦═╗╔══╗╚╝╔╝║╚╝║║╠╩═╦╗╠╗╔╝")
	print("║╔═╗║╔╗║╔╗╣╔╗║╚╝╚╝║╔╬╣║╠╣╔╗╣╔╗║╔═╝╔╝░░║║║║═╬╬╬╣║░")
	print("║║░║║╔╗║║║║╚╝╠╗╔╗╔╣║║║╚╣║║║║╚╝║║║╚═╗░░║║║║═╬╬╬╣╚╗")
	print("╚╝░╚╩╝╚╩╝╚╩══╝╚╝╚╝╚╝╚╩═╩╩╝╚╩═╗║╚═══╝░░╚╝╚══╩╝╚╩═╝")
	print("░░░░░░░░░░░░░░░░░░░░░░░░░░░╔═╝║░░░░░░░░░░░░░░░░░░")
	print("░░░░░░░░░░░░░░░░░░░░░░░░░░░╚══╝░░░░░░░░░░░░░░░░░░")

# preprocess
def preprocess(run_benchmark, verbose):
	print("Pre-processing")
	if not os.path.exists(pre_process_path):
		os.makedirs(pre_process_path)
	else:
		# clean up dest folder before running
		files = [f for f in os.listdir(pre_process_path) if os.path.isfile(os.path.join(pre_process_path, f))]
		for file in files:
			path = pre_process_path + file
			os.remove(path) 
	src = input_dir
	if run_benchmark:
		src = benchmark_dir
	files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]

	for file in files:
		if file.endswith(".jpg") or file.endswith(".png"):
			path = src + file
			filename = file.split('.')[0]
			out = pre_process_path + filename + ".png"
			prep.preprocess(path, out, verbose)

# char segmentation
def segment(verbose):
	print("Segmenting into characters")
	# clean up folder before running
	folders = [f for f in os.listdir(pre_process_path) if os.path.isdir(os.path.join(pre_process_path, f))]
	for f in folders:
		path = pre_process_path + f
		shutil.rmtree(path)

	seg.run_character_segementation_module(pre_process_path, verbose)

# char recognition
def recognition(args, verbose):
	# clean up dest folder before running
	files = [f for f in os.listdir(char_recognition_path) if os.path.isfile(os.path.join(char_recognition_path, f))]
	for file in files:
		path = char_recognition_path + file
		os.remove(path)
	
	print("Identifying characters")
	rec.char_recognition(args, pre_process_path + '*', char_recognition_path, verbose)

# post process w/rnn
def postprocess_rnn(verbose):
	print("Post-process step 1: RNN prediction")
	if not os.path.exists(rnn_path):
		os.makedirs(rnn_path)
	else:
		# clean up dest folder before running
		files = [f for f in os.listdir(rnn_path) if os.path.isfile(os.path.join(rnn_path, f))]
		for file in files:
			path = rnn_path + file
			os.remove(path)
	
	files_in_directory = [f for f in os.listdir(char_recognition_path) if os.path.isfile(os.path.join(char_recognition_path, f))]
	for i, file in enumerate(files_in_directory):
		input_path = char_recognition_path + file
		output_path = rnn_path + file
		rnn.predict(input_path, output_path, verbose)

# post process w/std spellchecker
def postprocess_std(verbose):
	print("Post-process step 2: spellchecker")
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	else:
		# clean up dest folder before running
		files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
		for file in files:
			path = output_dir + file
			os.remove(path)
	
	files_in_directory = [f for f in os.listdir(char_recognition_path) if os.path.isfile(os.path.join(char_recognition_path, f))]
	for i, file in enumerate(files_in_directory):
		input_path_1 = char_recognition_path + file
		output_path_1 = output_dir + file
		post.spellcheck(input_path_1, output_path_1, verbose)
		input_path_2 = rnn_path + file
		output_path_2 = output_dir + file.split('.')[0] + "_rnn.txt"
		post.spellcheck(input_path_2, output_path_2, verbose)

if __name__ == '__main__':
	# arguments for recognition
	parser = argparse.ArgumentParser(usage='A script for predicting the image with model.yaml')
	parser.add_argument('--test', action='store_true', default=False, help='use test folder')
	parser.add_argument('--letter', action='store_true', default=False, help='use letter model')
	parser.add_argument('--balanced', action='store_true', default=False, help='use balanced model with numbers')
	parser.add_argument('--benchmarks', action='store_true', default=False, help='run on benchmark directory')
	parser.add_argument('--verbose', action='store_true', default=False, help='show middle steps')
	args = parser.parse_args()
	
	start_program()

	run_benchmark = False
	if args.benchmarks:
		print("Running benchmarks")
		run_benchmark = True
	preprocess(run_benchmark, args.verbose)
	segment(args.verbose)
	recognition(args, args.verbose)
	postprocess_rnn(args.verbose)
	postprocess_std(args.verbose)

	print("Done")
