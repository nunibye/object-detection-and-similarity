import math
import numpy as np
import cv2

def FindNearbyPoints(point_set, point, tolerance, searched):
    set_of_found = []
    searched.append(point)
    print(searched)
    print(len(point_set))
    for i in point_set:
        if math.fabs(point[1] - i[1]) <= tolerance and math.fabs(point[0] - i[0]) <= tolerance:
            set_of_found.append(i)
            point_set.remove(i)
            print(set_of_found)
            
    if len(set_of_found) > 0:
        farthest1 = set_of_found[0]
    else:
        return set_of_found, 'Done'
    for i in set_of_found:
        
        if math.dist(i, point) > math.dist(i, farthest1):
            farthest1 = i
    if farthest1 in searched or len(point_set) == 0:
        return set_of_found, 'Done'
    else:
        return (set_of_found, point_set, farthest1, searched)

        
    

tolerance = 4
img_path = "file1.png"
img = cv2.imread(img_path)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv, (0, 0, 10), (255, 60, 255))


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
        if int(ele / 255) == 1:
            white_points.append((x, y))



max_white_point = white_points[0]
current_shape =[]
points = white_points
searched = []
var = FindNearbyPoints(points, max_white_point, tolerance, list(searched))

while var[1] != 'Done':
    var = FindNearbyPoints(var[1], var[2], tolerance, list(var[3]))
    print('tes', var)
    for i in var[0]:
        #print(i)
        current_shape.append(i)
print(current_shape)
'''
minx = min(i[0] for i in current_shape)
maxx = max(i[0] for i in current_shape)
miny = min(i[1] for i in current_shape)
maxy = max(i[1] for i in current_shape)
print(minx, maxx, miny, maxy)
'''

