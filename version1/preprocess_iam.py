import sys
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from operator import itemgetter

# added outpath so didn't overwrite file
def preprocess(img_url, out_url):
	img = cv2.imread(img_url, 0)

	# Otsu's thresholding
	ret, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# Detect edges
	im_edges = cv2.Canny(thresh_img, 50, 100)

	# Find straight, horizontal lines
	lines = cv2.HoughLinesP(im_edges, 1, np.pi/180.0, 50, np.array([]), 100, 2)
	h,w = im_edges.shape
	y_top = 0
	y_bot = h-1
	half = (h-1)/2
	for line in lines:
		for x1, y1, x2, y2 in line:
			if y1 == y2:
				if y1 < half and y1 > y_top:
					y_top = y1
				if y1 > half and y1 < y_bot:
					y_bot = y1

	# Crop out database text
	x1, y1, x2, y2 = 0, y_top, w-1, y_bot
	crop1 = thresh_img[y1:y2+1, x1:x2+1]

	# Detect edges again
	im_edges = cv2.Canny(crop1, 100, 200)
	# Fill in the text using dilations
	struct = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
	im2 = cv2.dilate(im_edges, struct, iterations = 30)

	# Find connected components in the image
	im2, contours, hierarchy = cv2.findContours(im2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

	# Crop again to remove borders
	x1, y1, x2, y2 = crop
	crop2 = crop1[y1:y2+1, x1:x2+1]

	# Save the final image
	fig = plt.imshow(crop2, cmap='gray')
	plt.axis('off')
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)
	plt.savefig(out_url, bbox_inches='tight', pad_inches = 0, dpi=300) # only can save .png format


if __name__ == "__main__":
	assert len(sys.argv) == 2, "No input given."
	image = sys.argv[1]
	print("Pre-processing image " + image)
	preprocess(image,"_"+image)
	print("Done")
