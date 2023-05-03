import cv2
import numpy as np

# read the image
img = cv2.imread("jiraf.png")

# convert image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# invert the blurred image
inverted_image = cv2.bitwise_not(gray_image)

# apply Gaussian blur 
blurred_image = cv2.GaussianBlur(inverted_image, (21,21), 0)

#invert the blur image again
inverted_blur_image = cv2.bitwise_not(blurred_image)

# apply divide operation to grayscale image and inverted blurred image
sketch_image = cv2.divide(gray_image, inverted_blur_image, scale=256.0)

# adjust contrast
sketch_image = cv2.addWeighted(sketch_image, 1.0, np.zeros_like(sketch_image), 0, 0)

# save the final image
cv2.imwrite("Output_img.jpg", sketch_image)
