import sys
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from operator import itemgetter

def preprocess(img_id):
	img = cv2.imread(img_id + ".png", 0)

	# Shadow removal
	struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))
	# Remove text from the image, leaving a background
	closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, struct)
	# Remove the background from the original image
	clean_img = np.divide(img, closing*1.0)
	# Save as intermediate step
	fig = plt.imshow(clean_img, 'gray')
	plt.axis('off')
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)
	plt.savefig(img_id + "_1", bbox_inches='tight', pad_inches = 0, dpi=300)

	# Otsu's thresholding
	clean_img = cv2.imread(img_id + "_1.png", 0)
	ret, thresh_img = cv2.threshold(clean_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# Detect edges
	im_edges = cv2.Canny(thresh_img, 100, 200)

	# Fill in the text using dilations
	struct = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
	im2 = cv2.dilate(im_edges, struct, iterations = 15)

	# Find connected components in the image
	contours, hierarchy = cv2.findContours(im2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	c_info = []
	total_area = 0
	for c in contours:
		area = cv2.contourArea(c)
		total_area = total_area + area
		x,y,w,h = cv2.boundingRect(c)
		c_info.append({
	        'x1': x,
	        'y1': y,
	        'x2': x + w - 1,
	        'y2': y + h - 1,
	        'area': area
	    })

	# Sort components by size
	c_info.sort(key=itemgetter('area'), reverse=True)
	h, w = im2.shape
	page_size = h*w
	c = c_info[0]
	crop = c['x1'], c['y1'], c['x2'], c['y2']
	box_size = (c['x2'] - c['x1']) * (c['y2'] - c['y1'])
	area = c['area']
	# Calculate F1 score
	recall = 1.0 * area / total_area
	prec = 1 - 1.0 * box_size / page_size
	f1 = 2 * (prec * recall) / (prec + recall)

	# Find optimal bounding box
	for c in c_info[1:]:
		x1, y1, x2, y2 = c['x1'], c['y1'], c['x2'], c['y2']
		new_crop = min(x1, crop[0]), min(y1, crop[1]), max(x2, crop[2]), max(y2, crop[3])
		new_size = (new_crop[2] - new_crop[0]) * (new_crop[3] - new_crop[1])
		new_area = area + c['area']
		r = 1.0 * new_area / total_area
		p = 1 - 1.0 * new_size / page_size
		new_f1 = 2 * (p * r) / (p + r)
		if new_f1 > f1:
			crop = new_crop
			box_size = new_size
			area = new_area
			f1 = new_f1

	# Crop image to text
	x1, y1, x2, y2 = crop
	crop = clean_img[y1:y2+1, x1:x2+1]

	# Save the final image
	fig = plt.imshow(crop, cmap='gray')
	plt.axis('off')
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)
	plt.savefig(img_id + "_1", bbox_inches='tight', pad_inches = 0, dpi=300)


if __name__ == "__main__":
	assert len(sys.argv) == 2, "No input given."
	image = sys.argv[1]
	print("Pre-processing image " + image)
	preprocess(image)
	print("Done")
