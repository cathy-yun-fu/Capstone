import preprocess_iam as iam
import parse_data as parse
import urllib.request as urllib2 
import getpass

if __name__ == "__main__":
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

	print("Pre-processing images")
	for image in imgIDs:
		iam.preprocess("../IAMData/data/"+image)