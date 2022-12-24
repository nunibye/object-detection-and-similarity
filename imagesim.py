import cv2
  
     
# test image
image1 = cv2.imread('ele4.jpg')
#gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
histogram = cv2.calcHist([image1], [0],
                         None, [256], [0, 256])
  
# data1 image
image2 = cv2.imread('ele5.jpg')
#gray_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
histogram1 = cv2.calcHist([image2], [0],
                          None, [256], [0, 256])
  
# data2 image
image3 = cv2.imread('ele7.jpg')
#gray_image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
histogram2 = cv2.calcHist([image3], [0],
                          None, [256], [0, 256])
  
  
c1, c2 = 0, 0
  
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
  
if(c1<c2):
    print("data1.jpg is more similar to test.jpg as compare to data2.jpg")
else:
    print("data2.jpg is more similar to test.jpg as compare to data1.jpg")