import os
from collections import defaultdict
import urllib.request as urllib2
from bs4 import BeautifulSoup
import getpass

def create_dict():
	# Create dict mapping images to their writer
	imDict = defaultdict(list)

	with open('forms.txt', 'r') as forms:
		for line in forms:
			if line[0] == '#':
				continue
			words = line.split(" ")
			formID = words[0]
			writerID = words[1]
			imDict[writerID].append(formID)

	return imDict

def get_paragraph(ids, baseURL, opener):
	# Get groundtruth info
	if not os.path.exists('textfile/'):
		os.makedirs('textfile/')

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

if __name__=='__main__':
	print("Getting paragraph IDs")
	ids = set()
	with open('forms.txt', 'r') as forms:
			for line in forms:
				if line[0] == '#':
					continue
				words = line.split(" ")
				formID = words[0]
				ids.add(formID)
	print("Number of paragraphs:", len(ids))
	print("Downloading")
	# Set up authentication for database
	url = "http://www.fki.inf.unibe.ch/DBs/iamDB/data/"
	user = input('username: ')
	password = getpass.getpass('password: ')
	password_manager = urllib2.HTTPPasswordMgrWithDefaultRealm()
	password_manager.add_password(None, url, user, password)
	auth_manager = urllib2.HTTPBasicAuthHandler(password_manager)
	opener = urllib2.build_opener(auth_manager)
	get_paragraph(ids, url, opener)