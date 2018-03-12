import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

img = cv2.imread("pos20.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
inv = cv2.bitwise_not(gray)

# x,y coords of all white pixels
coords = np.column_stack(np.where(inv > 0))
rect = cv2.minAreaRect(coords)
angle = rect[-1]
if angle < -45:
	angle = -(90 + angle)
else:
	angle = -angle

# rotate the image to deskew it
(h_img, w_img) = img.shape[:2]
(h, w) = np.int0(rect[1])
center = (rect[0][1], rect[0][0])

M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(img, M, (w_img, h_img),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

plt.imshow(rotated)
plt.show()