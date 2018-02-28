import urllib.request as urllib2 
import getpass
import os
import sys
import cv2
import os

from PIL import Image

# our files
import preprocess_iam as iam
import parse_data as parse
import character_segmentation_v1 as seg

mypath_iam = "../IAMData/data/"	
pre_process_path = "../ROOT_DIR/"

# download step
def Step1():
	if not os.path.exists(mypath_iam):
		os.makedirs(mypath_iam)
	
	print("Parsing database")
	imDict = parse.create_dict()

	print("Downloading data")
	# Set up authentication for database
	url = "http://www.fki.inf.unibe.ch/DBs/iamDB/data/"
	user = input('username: ')
	password = getpass.getpass('password: ')
	password_manager = urllib2.HTTPPasswordMgrWithDefaultRealm()
	password_manager.add_password(None, url, user, password)
	auth_manager = urllib2.HTTPBasicAuthHandler(password_manager)
	opener = urllib2.build_opener(auth_manager)

	imgIDs = parse.get_data(imDict, url, opener)

	print("Getting targets")
	parse.get_target(imgIDs, url, opener)

	for file in imgIDs:
		path = mypath_iam + file + ".png"
		im = Image.open(path)
		im = im.convert("RGB")
		im.save(mypath_iam + file + ".jpg")

# preprocess step
def Step2():
	print("Pre-processing images")
	if not os.path.exists(pre_process_path):
		os.makedirs(pre_process_path)
	else:
		files = [f for f in os.listdir(pre_process_path) if os.path.isfile(os.path.join(pre_process_path, f))]
		for file in files: # clean up folder before running
			path = pre_process_path + file
			os.remove(path) 
			
	onlyfiles = [f for f in os.listdir(mypath_iam) if os.path.isfile(os.path.join(mypath_iam, f))]

	print (onlyfiles)

	for file in onlyfiles:
		if (file.endswith(".jpg")):
			filename, ext = os.path.splitext(file)
			path = mypath_iam + file
			out = pre_process_path + filename + ".png"
			iam.preprocess(path,out)

def Step3():
	seg.run_character_segementation_module(pre_process_path)

def run_all(step):
	try:
		step = int(step)
	except:
		print("Error: input is not an integer")
		return

	if (step <= 1):
		Step1()
	if (step <=2):
		Step2()
	if (step <=3):
		Step3()
	else:
		print("Invalid step number")
	
if __name__ == "__main__":
	if (len(sys.argv) == 1):
		run_all(0)
	else:
		# run with a number corresponding to step would like to start from
		run_all(sys.argv[1])