import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

img = cv2.imread('input.png')
# Detect edges
im_edges = cv2.Canny(img, 50, 100)
# Find straight, horizontal lines
lines = cv2.HoughLinesP(im_edges, 1, np.pi/180.0, 50, np.array([]), 100, 2)
h,w = im_edges.shape
y_top = 0
y_bot = h-1
half = (h-1)/2
for x1, y1, x2, y2 in lines[0]:
	if y1 == y2:
		if y1 < half and y1 > y_top:
			y_top = y1
		if y1 > half and y1 < y_bot:
			y_bot = y1
		cv2.line(img, (x1,y1),(x2,y2),(255,0,0),5)
plt.imshow(img, cmap='gray')
plt.show()

# Crop image
x1, y1, x2, y2 = 0, y_top, w-1, y_bot
cropped = img[y1:y2+1, x1:x2+1, :]
fig = plt.imshow(cropped)
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig('crop1', bbox_inches='tight', pad_inches = 0, dpi=300)
