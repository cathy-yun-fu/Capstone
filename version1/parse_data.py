import os
from collections import defaultdict
import urllib.request as urllib2
from bs4 import BeautifulSoup

def create_dict():
	# Create dict mapping images to their writer
	imDict = defaultdict(list)

	with open('../IAMData/forms.txt', 'r') as forms:
		for line in forms:
			if line[0] == '#':
				continue
			words = line.split(" ")
			formID = words[0]
			writerID = words[1]
			imDict[writerID].append(formID)

	return imDict

def get_data(imDict, baseURL, opener):
	# Download images
	if not os.path.exists('../IAMData/data/'):
		os.makedirs('../IAMData/data/')

	urllib2.install_opener(opener)
	url = baseURL + "forms/"
	ids = []
	with open('../IAMData/writers.txt', 'r') as temp:
		writers = [line.rstrip('\n') for line in temp]
		for writer in writers:
			for ID in imDict[writer]:
				ids.append(ID)
				name = ID + ".png"
				print("Downloading", url+name)
				img = urllib2.urlopen(url+name)
				with open("../IAMData/data/"+name, 'wb') as file:
					file.write(img.read())
	return ids

def get_target(ids, baseURL, opener):
	# Get groundtruth info
	if not os.path.exists('../IAMData/target/'):
		os.makedirs('../IAMData/target/')

	urllib2.install_opener(opener)
	url = baseURL + "xml/"
	for ID in ids:
		print("Getting gt for", ID)
		xml = urllib2.urlopen(url+ID+".xml")
		soup = BeautifulSoup(xml.read(), "lxml")
		text = ""
		for line in soup.find_all("line"):
			text = text + line['text'] + '\n'
		with open("../IAMData/target/"+ID+".txt", 'w+') as file:
			file.write(text)
