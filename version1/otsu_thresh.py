import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('clean2.png', 0)

img_clean = img
cv2.fastNlMeansDenoising(img, img_clean, 15, 31, 11)

# Otsu's thresholding
ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret, th1 = cv2.threshold(img_clean, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

images = [(th, "clean_otsus2"), (th1, "cleaner")]

for image, name in images:
	fig = plt.imshow(image, 'gray')
	plt.axis('off')
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)
	plt.savefig(name, bbox_inches='tight', pad_inches = 0, dpi=300)