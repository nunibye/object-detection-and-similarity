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
img_path = "file2.png"
img = cv2.imread(img_path)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv, (0, 0, 10), (255, 30, 255))


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
    xlist = X[row_ix, 0].tolist()[0]
    ylist = X[row_ix, 1].tolist()[0]
    
    element_image = numpydata[min(ylist) - buffer:max(ylist) + buffer, min(xlist) - buffer:max(xlist) + buffer]
    cv2.imwrite(f'ele{cluster}.jpg', element_image)
    new_images.append(element_image)
histogram_list = []
for i in new_images:

    histogram_list.append(cv2.calcHist([i], [0], None, [256], [0, 256]))
diffrence_list = []


for count1, ele in enumerate(histogram_list):
    print (count1 + 1, '/', len(histogram_list))
    base_image = ele
    miniumum_distance = None
    for count2, h in enumerate(histogram_list):
        
        i = 0
        c1 = 0
        while i<len(base_image) and i<len(h):
            c1+=(base_image[i]-h[i])**2
            i+= 1
        c1 = float(c1**(1 / 2))
        if c1 != 0.0:
            if miniumum_distance is None:
                miniumum_distance = (count2, c1)
            else:
                
                if c1 < miniumum_distance[1] and c1 != 0.0:
                    miniumum_distance = (count2, c1)
        print(c1)
    print()
    print(miniumum_distance)
    diffrence_list.append(miniumum_distance)
    print('\n')
print(diffrence_list)
#find most similar

minimum = diffrence_list[0][1]
minimum_index = (0, diffrence_list[0][0])
for i in range(len(diffrence_list)):
    if diffrence_list[i][1] < minimum:
       minimum = diffrence_list[i][1]
       minimum_index = (i, diffrence_list[i][0])
print(minimum_index)


