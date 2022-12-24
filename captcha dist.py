import math
import numpy as np
import cv2
from numpy import where
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import unique

from sklearn.cluster import Birch
buffer = 5

        
    

tolerance = 4
img_path = "file1.png"
img = cv2.imread(img_path)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv, (0, 0, 10), (255, 50, 255))


# Build mask of non black pixels.
nzmask = cv2.inRange(hsv, (0, 0, 100), (255, 255, 255))

# Erode the mask - all pixels around a black pixels should not be masked.
nzmask = cv2.erode(nzmask, np.ones((3,3)))

mask = mask & nzmask

new_img = img.copy()
new_img[np.where(mask)] = 255

edges = cv2.Canny(new_img, 10, 40)
	
cv2.imwrite('result1.jpg', edges)
numpydata = np.asarray(edges)
white_points = []

for y, row in enumerate(numpydata):
    for x, ele in enumerate(row):
        if int(ele) == 255:
            white_points.append([x, y])

# define dataset
X = np.array(white_points)
#print(X.tolist())
# define the model
model = Birch(threshold=0.01, n_clusters=7)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
new_images = []
for cluster in clusters:
 # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
 # create scatter of these samples
    #pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    xlist = X[row_ix, 0].tolist()[0]
    ylist = X[row_ix, 1].tolist()[0]
    
    element_image = numpydata[min(ylist) - buffer:max(ylist) + buffer, min(xlist) - buffer:max(xlist) + buffer]
    
    new_images.append(element_image)
histogram_list = []
for i in new_images:

    histogram_list.append(cv2.calcHist([i], [0],
                            None, [256], [0, 256]))
diffrance_list = []

diffrance = []
for ele in histogram_list:
    base_image = ele
    lowest = 'x'
    for h in histogram_list:
        
        i = 0
        c1 = 0
        while i<len(base_image) and i<len(h):
            c1+=(base_image[i]-h[i])**2
            i+= 1
        c2 = c2**(1 / 2)

  


''' 
# Euclidean Distance between data1 and test
i = 0
while i<len(histogram) and i<len(histogram1):
    c1+=(histogram[i]-histogram1[i])**2
    i+= 1
c1 = c1**(1 / 2)
  
 
# Euclidean Distance between data2 and test
i = 0
while i<len(histogram) and i<len(histogram2):
    c2+=(histogram[i]-histogram2[i])**2
    i+= 1
c2 = c2**(1 / 2)
'''
if(diffrance[1]<diffrance[2]):
    print("data1.jpg is more similar to test.jpg as compare to data2.jpg")
else:
    print("data2.jpg is more similar to test.jpg as compare to data1.jpg")
# show the plot
#pyplot.show()
