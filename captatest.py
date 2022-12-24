# import the necessary packages
import argparse
import imutils
import cv2
# construct the argument parse and parse the arguments
'''
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="file1.png")
args = vars(ap.parse_args())
'''
# load the image, convert it to grayscale, blur it slightly,
# and threshold it
image = cv2.imread('file1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 230, 70, cv2.THRESH_BINARY)[1]

cv2.imshow('Contours', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()