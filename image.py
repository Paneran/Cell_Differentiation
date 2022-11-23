import cv2
import numpy as np
import matplotlib.pyplot as plt

# actual 423
file = "A1_CH1"
tag = "_2"
dir = "C:/Leia Research/20221018_C2C12 DifferentiationExperiment/20221018_ImmLtf_D4/ImmLtf_D4_A1_Control/ImmLtf_D4_A1_control_01/"+ file + ".tif"
ret, img = cv2.imreadmulti(dir)
# Denoising
img[0][:,:, 1] = 0
img[0][:,:, 2] = 0
#denoisedImg = cv2.fastNlMeansDenoising(img[0], cv2.IMREAD_GRAYSCALE)
denoisedImg = img[0]
denoisedImg = cv2.cvtColor(denoisedImg, cv2.COLOR_BGR2GRAY)
# Threshold (binary image)
# thresh – threshold value.
# maxval – maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
# type – thresholding type
th, threshedImg = cv2.threshold(denoisedImg, 0, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU) # src, thresh, maxval, type

# Perform morphological transformations using an erosion and dilation as basic operations
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
# morphImg = cv2.morphologyEx(threshedImg, cv2.MORPH_OPEN, kernel)
morphImg = threshedImg

# Find and draw contours
contours, hierarchy = cv2.findContours(morphImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contoursImg = cv2.cvtColor(morphImg, cv2.COLOR_GRAY2RGB)
cv2.drawContours(contoursImg, contours, -1, (255,100,0), 3)

print(contours[0])
start = 0
end = 1000
step = 200

# Find the areas of each contour
num_contours = len(contours)
con_areas = np.zeros(num_contours)
for i in np.arange(num_contours):
    con_areas[i] = cv2.contourArea(contours[i])

# The picture frame is counted as a contour. remove it. 
con_areas[np.argmax(con_areas)] = 0

# put each contour in bins. Large bins to capture the real normal distribution
length = int((end-start)//step)
bins = [[]]*length
for i in np.arange(num_contours):
    index = int(con_areas[i]//step)
    if index >= length:
        index = length-1
    bins[index] = np.append(bins[index],con_areas[i])

# Find the most popular bin
m = 0
for i in np.arange(len(bins)):
    if len(bins[i]) > len(bins[m]):
        m = i

# Calculate average cell area and the estimated cell count
print(np.average(bins[m]))
print(np.sum(con_areas)//np.average(bins[m]))

# generate the histogram
plt.hist(con_areas, np.arange(start, end, step))
plt.show()

# write data. 
cv2.imwrite("C:/Leia Research/20221018_C2C12 DifferentiationExperiment/20221018_ImmLtf_D4/ImmLtf_D4_A1_Control/ImmLtf_D4_A1_control_01/" + file +"_result" + tag + ".tif", contoursImg)
textFile = open("results/results.txt","a")
textFile.write(file + "_result" + " Dots number: {}".format(num_contours) + "\n")
textFile.close()