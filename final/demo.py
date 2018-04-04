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

# paths
input_dir = "INPUT/"
output_dir = "OUTPUT/"
pre_process_path = "PREPROCESS/"
char_recognition_path = "CNN/"
test_dir = 'test_data/'
MAX_STEP = 4

# preprocess
def preprocess():
	print("Pre-processing image")
	if not os.path.exists(pre_process_path):
		os.makedirs(pre_process_path)
	else:
		# clean up folder before running
		files = [f for f in os.listdir(pre_process_path) if os.path.isfile(os.path.join(pre_process_path, f))]
		for file in files:
			path = pre_process_path + file
			os.remove(path) 
			
	files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

	for file in files:
		path = input_dir + file
		filename = file.split('.')[0]
		out = pre_process_path + filename + ".png"
		prep.preprocess(path, out)

# char segmentation
def segment():
	print("Segmenting into characters")
	# clean up folder before running
	folders = [f for f in os.listdir(pre_process_path) if os.path.isdir(os.path.join(pre_process_path, f))]
	for f in folders:
		path = pre_process_path + f
		shutil.rmtree(path)

	seg.run_character_segementation_module(pre_process_path)

# char recognition
def recognition(args):
	print("Identifying characters")
	rec.char_recognition(args, pre_process_path + '*', char_recognition_path)

# post process
def postprocess():
	print("Post-processing text")
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	files_in_directory = [f for f in os.listdir(char_recognition_path) if os.path.isfile(os.path.join(char_recognition_path, f))]
	for i, file in enumerate(files_in_directory[:-1]):
		input_path = char_recognition_path + file
		output_path = output_dir + file
		post.spellcheck(input_path, output_path)

if __name__ == '__main__':
	# arguments for recognition
	parser = argparse.ArgumentParser(usage='A script for predicting the image with model.yaml')
	parser.add_argument('--test', action='store_true', default=False, help='use test folder')
	parser.add_argument('--letter', action='store_true', default=False, help='use letter model')
	parser.add_argument('--balanced', action='store_true', default=False, help='use balanced model with numbers')
	args = parser.parse_args()
	
	preprocess()
	segment()
	recognition(args)
	postprocess()

	print("Done")
