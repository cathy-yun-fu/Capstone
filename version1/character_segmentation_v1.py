import os
import math
import numpy as np
import re # alphaneumeric order

from skimage.measure import label, regionprops
from skimage.color import rgb2gray
from skimage.transform import rotate
from skimage import io
from skimage.viewer import ImageViewer

# CIRCULARITY_THRESHOLD = 0.2
CIRCULARITY_THRESHOLD = 1
END_THRESH = 10
FRAC_H = 0.5

# image.shape[1] / FONT_SIZE_FACTOR (height / FONT_SIZE_FACTOR)
FONT_SIZE_FACTOR = 3
# FONT_SIZE_FACTOR = 4


def create_crop_box(box,numRow,numCol):
	# bounding box: (min_row, min_col, max_row, max_col)
	if (box[0] <= 10):
		r1 = 0
	else:
		r1 = box[0]-10

	if (box[1] <= 10):
		c1 = 0
	else:
		c1 = box[1]-10

	if (box[2] >= numRow - 10):
		r2 = numRow
	else:
		r2 = box[2]+10

	if (box[3] >= numCol - 10):
		c2 = numCol
	else:
		c2 = box[3]+10

	return r1,r2,c1,c2


def split_into_letters(filename, output_dir):
	test_image = io.imread(filename)
	test_image = rotate(test_image, -90, resize=True)

	if len(test_image.shape) > 2 and test_image.shape[2] > 1:
		test_image = rgb2gray(test_image)

	# viewer = ImageViewer(test_image)
	# viewer.show()
	# print (test_image.shape)

	# white on black ; white: 1, black: 0
	# ==== does not invert =====
	if (test_image.max() > 1.0): 
		binary_image = np.where(test_image<180,0.0, 1.0) # 70%
	else:
		binary_image = np.where(test_image<0.7, 0.0, 1.0) # 70%
 
	label_image = label(binary_image)
	num = label_image.max()

	# print (num)

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	letter_end = 0
	total_h = 0

	for i in range(1,num+1):
		maskedImage = np.where(label_image==i,1.0,0)
		binaryImage = np.where(label_image==i,1,0)
		regionprops_data = regionprops(binaryImage)
		
		circularities = math.pow(regionprops_data[0].perimeter,2)/(4*math.pi*regionprops_data[0].area)
		if circularities <= 1:
			continue

		# bounding box: (min_row, min_col, max_row, max_col)
		r1,r2,c1,c2 = create_crop_box(regionprops_data[0].bbox, binary_image.shape[0],binary_image.shape[1])
		if r2 < letter_end + END_THRESH and r2 > letter_end - END_THRESH:
			continue
		letter_end = r2

		if i == num:
			avg_h = total_h/(num-1)
			total_h = 0
			if c2 - c1 < FRAC_H * avg_h:
				continue

		total_h += c2 - c1

		maskedImage = maskedImage[r1:r2,c1:c2]
		col1 = c1
		col2 = c2
		# print(i)
		# print(regionprops_data[0].bbox)
		# print("{:1} {:2} {:3} {:4}".format(r1,r2,c1,c2))
		
		maskedImage = rotate(maskedImage, 90, resize=True)

		# viewer = ImageViewer(maskedImage)
		# viewer.show()

		fileName = output_dir + "/Letter{:1}.jpg".format(i)
		io.imsave(fileName,maskedImage)


# Step 2
def split_into_words(filename, output_dir):
	test_image = io.imread(filename)
	test_image = rotate(test_image, -90, resize=True)
	print (filename)

	# viewer = ImageViewer(test_image)
	# viewer.show()
	# print (test_image.shape)

	if len(test_image.shape) > 2 and test_image.shape[2] > 1:
		test_image = rgb2gray(test_image)

	# ==== does not invert =====
	if (test_image.max() > 1.0): 
		binary_image = np.where(test_image<180,0.0, 1.0) # 70%
	else:
		binary_image = np.where(test_image<0.7, 0.0, 1.0) # 70%
 
	# viewer = ImageViewer(binary_image)
	# viewer.show()

	label_image = label(binary_image)
	num_connected_regions = label_image.max() # total number of connected regions

	regionprops_data = regionprops(label_image)

	remove_indices = []
	for i in range(num_connected_regions):
		circularities = math.pow(regionprops_data[i].perimeter,2)/(4*math.pi*regionprops_data[i].area)
		# print("i: {:1}, circularities: {:2}".format(i,circularities))
		if (circularities <= CIRCULARITY_THRESHOLD):
			remove_indices.append(i)

	# print (remove_indices)
	num_connected_regions = num_connected_regions - len(remove_indices)

	for i in sorted(remove_indices, reverse=True):
		del regionprops_data[i]

	rows=[]
	rows.append(0)

	fontSize = int(binary_image.shape[1]/FONT_SIZE_FACTOR) # param - low cuz current method of cropping leaves alot of white space above and below actual chars
	print ("fontSize calculated: {:1}".format(fontSize))

	# fontSize = 20

	for i in range(num_connected_regions-1):
		data_1 = regionprops_data[i]
		data_2 = regionprops_data[i+1]
		# print("centroids:")
		# print(data_1.centroid)

		# bounding box: (min_row, min_col, max_row, max_col)
		if ((data_2.bbox[0] - data_1.bbox[2]) > fontSize):
			val = int((data_2.bbox[0] + data_1.bbox[2])/2)
			rows.append(val)
			
	rows.append(binary_image.shape[0])
	# print(rows)

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	for i in range(len(rows)-1):
		image = binary_image[rows[i]:rows[i+1],:]
		image = rotate(image,90,resize=True)
		# viewer = ImageViewer(image)
		# viewer.show()
		fileName = output_dir + "/Img{:1}.jpg".format(i)
		io.imsave(fileName,image)


# Step 1
def split_rows(filename, output_dir):
	test_image = io.imread(filename)

	if len(test_image.shape) > 2 and test_image.shape[2] > 1:
		test_image = rgb2gray(test_image)

	# viewer = ImageViewer(test_image)
	# viewer.show()
	# print (test_image.shape) # temp

	# white on black ; white: 1, black: 0
	# ==== inverts =====
	if (test_image.max() > 1.0): 
		binary_image = np.where(test_image<180, 1.0, 0.0) # 70%
	else:
		binary_image = np.where(test_image<0.7, 1.0, 0.0) # 70%

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	print("Input image shape: %s" % (test_image.shape,))
	flattened_image = binary_image.sum(axis=1) # horizontal (sum of each row)
	
	# print (max(flattened_image))
	# print(min(flattened_image))
	print("Flattened image shape: %s" % (flattened_image.shape,))

	top_row = -1
	bottom_row = -1
	row_centers = []
	n = 0
	for i in range(flattened_image.shape[0]):
		if flattened_image[i] <= 0:
			if top_row < 0:
				continue
			else:
				bottom_row = (i)
				width = int(bottom_row - top_row)
				# print(width)
				
				if (width <= 5):
					top_row = -1
					bottom_row = -1
					continue 

				row_centers.append(int(bottom_row - top_row))
				
				image = binary_image[top_row:bottom_row,:]
				# print(str(top_row) + "  " + str(bottom_row))

				# viewer = ImageViewer(image)
				# viewer.show()
				fileName = output_dir + "/Row{:1}.jpg".format(n)
				n = n +1;
				io.imsave(fileName,image)

				top_row = -1
				bottom_row = -1
		elif flattened_image[i] > 0:
			# if (flattened_image[i] < 10):
			# 	print("value: " + str(flattened_image[i]))
			if top_row <0:
				top_row = i
			else:
				continue

def alphanumeric_sort(list):
    """ Sorts the given iterable in the way that is expected.
    Required arguments:
    l -- The iterable to be sorted.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(list, key=alphanum_key)

def run_character_segementation_module(base_directory):
	if not os.path.exists(base_directory):
		print("ERROR MESSAGE!")
		raise IOError("Character Segmentation Module: Base Directory '{:1}' does not exist".format(base_directory))
	
	files_in_directory = [f for f in os.listdir(base_directory) if os.path.isfile(os.path.join(base_directory, f))]
	# folders_in_directory = [f for f in os.listdir(base_directory) if not os.path.isfile(os.path.join(base_directory, f))]

	row_directories = [] 
	# doesn't need to be sorted, each file should be a paragraph
	print("================== Splitting paragraphs into rows ======================")
	for i, file in enumerate(files_in_directory):
		if file.endswith(".jpg") or file.endswith(".png"):
			name = file.split(".");
			input_path = os.path.join(base_directory,file)
			output_dir = os.path.join(base_directory,"{:1}".format(name[0]))
			split_rows(input_path, output_dir)
			row_directories.append(output_dir)

	word_directories = []
	print("================== Splitting rows into words ======================")
	for directory in row_directories:
		files_in_directory = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
		files_in_directory = alphanumeric_sort(files_in_directory)
		# rows
		for i, file in enumerate(files_in_directory):
			input_path = os.path.join(directory,file)
			output_dir = os.path.join(directory,"Row{:1}".format(i))
			split_into_words(input_path,output_dir)
			word_directories.append(output_dir)

	print("================== Splitting words into letters ======================")
	for directory in word_directories:
		files_in_directory = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
		files_in_directory = alphanumeric_sort(files_in_directory)
		# rows
		for i, file in enumerate(files_in_directory):
			input_path = os.path.join(directory,file)
			output_dir = os.path.join(directory,"Word{:1}".format(i))
			split_into_letters(input_path,output_dir)

run_character_segementation_module("../ROOT_DIR/")
