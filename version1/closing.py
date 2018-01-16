import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# Read in the image
img = cv2.imread('img.jpg', 0)
# Create a struct that is larger than the text but smaller than the shadow
struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))
# Remove text from the image, leaving a background
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, struct)
# Remove the background from the original image
clean_img = np.divide(img, closing*1.0)
# Save the images
images = [("background2", closing), ("clean2", clean_img)]
for name, image in images:
	fig = plt.imshow(image, 'gray')
	plt.axis('off')
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)
	plt.savefig(name, bbox_inches='tight', pad_inches = 0, dpi=300)