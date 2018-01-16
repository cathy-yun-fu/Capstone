import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from operator import itemgetter

img = cv2.imread('clean_otsus2.png')
# Detect edges in image
im_edges = cv2.Canny(img, 100, 200)
# Fill in the text using dilations
struct = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
im2 = cv2.dilate(im_edges, struct, iterations = 15)

plt.imshow(im2, cmap = 'gray')
plt.show()

# Find connected components in the image
contours, hierarchy = cv2.findContours(im2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
c_info = []
total_area = 0
for c in contours:
	area = cv2.contourArea(c)
	total_area = total_area + area
	x,y,w,h = cv2.boundingRect(c)
	# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	# plt.imshow(img)
	#plt.show()
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

x1, y1, x2, y2 = crop
cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0),2)
plt.imshow(img)
plt.show()

# Crop image
cropped = img[y1:y2+1, x1:x2+1, :]
fig = plt.imshow(cropped)
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig('output', bbox_inches='tight', pad_inches = 0, dpi=300)