import cv2
import numpy as np
import random
# Read image
img = cv2.imread('mask.png', cv2.IMREAD_COLOR)
# Convert the image to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the edges in the image using canny detector
edges = cv2.Canny(gray, 20, 250)
# cv2.imshow('edges',edges)
# cv2.waitKey(0)
# Detect points that form a line
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=10, maxLineGap=100)

print(lines.shape[0])
intensity_value = range(0,256)
grad = np.zeros(lines.shape[0])
mag = np.zeros_like(grad)
for index, line in enumerate(lines):
    x1, y1, x2, y2 = line[0]
    grad[index] = abs((y1 - y2)/(x1 - x2))
    mag[index] = (x1-x2)**2 + (y1-y2)**2

    R = random.choice(intensity_value)
    G = random.choice(intensity_value)
    B = random.choice(intensity_value)
    img = cv2.circle(img,(x1,y1),6,(R,G,B),3)
    img = cv2.circle(img,(x2,y2),6,(R,G,B),3)
    #img = cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

# Draw lines on the image
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
# Show result
print(grad,mag)
print(lines)
# cv2.imwrite("Result Image.jpg",img)
#
# cv2.imshow("Result Image", img)
# cv2.waitKey(0)
# [[[258 344 299 164]]
#
#  [[332 338 334 168]]
#
#  [[335  78 336  19]]
#
#  [[314  78 326  19]]
#
#  [[273 273 298 165]]
#
#  [[333 163 334 247]]
#
#  [[301 163 329 161]]]