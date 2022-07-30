import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.image as mpimg
from skimage import data, io
from skimage.color import rgb2gray
import cv2
from skimage import img_as_ubyte

#reads the image
skimage_image = rgb2gray(io.imread("example_image/05142020_1.jpeg"))

cv_image = img_as_ubyte(skimage_image)
cv_image = cv2.resize(cv_image, (960, 540))

mouseX = 0
mouseY = 0

#first (#2) number is the threshold number
#second (#3 number is the end number to assign the numbers that pass it
ret, thresh = cv2.threshold(cv_image, 180, 255, cv2.THRESH_BINARY)


#contours is a list of all the contours
#each contour is Numpy array of x and y coordinates of points on the outline
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print("numberofcontours" + str(len(contours) - 1))

# def count_contours():

goodlist = np.array([])
contourlist = np.array([])

#for every contour
for i in range(len(contours) - 1):
    mask = np.full_like(cv_image, 255)  # Create mask where white is what we want, black otherwise
    
    #Its first argument is source image, second argument is the contours which should be passed as a Python list, third argument is index of contours 
    # (useful when drawing individual contour. To draw all contours, pass -1) and remaining arguments are color, thickness etc.
    cv2.fillPoly(mask, pts =[contours[i + 1]], color=(0,0,0))
    out = np.full_like(cv_image, 255)  # Extract out the object and place into output image


    #TODO ask ms oneal what this means
    out[mask == 0] = cv_image[mask == 0]


    # Now crop
    (y, x) = np.where(mask == 0)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = out[topy:bottomy + 1, topx:bottomx + 1]
    sumblack = 0
    times = 0
    for k in range(len(out)):
        for j in range(len(out[k])):
            sumblack = sumblack + out[k, j]
            times = times + 1
    
    #if the contour is big enough
    if (times > 300):
        print("checkpt1")
        print(sumblack/times)
        goodlist = np.append(goodlist, [int(i + 1)])
        contourlist = np.append(contourlist, [sumblack/times])
    # print(str(sumblack / times))
    # print(str(times))
        
    # # Show the output image
    # cv2.imshow('Output', out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
drawcontours = []
print("length" + str(len(goodlist)))
for k in range(len(goodlist)):
    print("checkpoint2")
    print(goodlist[k])
    drawcontours.append(contours[int(goodlist[k])])

#ignore the first contour
print(len(drawcontours))
print(len(contours))

#For cv2.getTrackbarPos() function, first argument is the trackbar name, second one is the window name to which it is attached, 
# third argument is the default value, fourth one is the maximum value and fifth one is the callback function which is executed everytime trackbar value changes. 
# The callback function always has a default argument which is the trackbar position.
def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(cv_image,(x,y),100,(255,0,0),-1)
        mouseX,mouseY = x,y
cv2.namedWindow('image')
# cv2.createTrackbar('Threshold', "main", 225, 255, None)
cv2.setMouseCallback('image',draw_circle)
cv2.drawContours(cv_image, drawcontours, -1, (0, 245, 0), 3)
print("fin")
# cv2.imshow('img', cv_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

while(1):
    cv2.imshow('image',cv_image)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print (mouseX,mouseY)

#TODO
#shape recognition and outlining. 
#applying a value

#OUTLINING 
#https://stackoverflow.com/questions/61166260/python-opencv-outline-black-pixels-in-a-blank-white-canvas

#CONTOURS DOCS
#https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html

#HOW TO DRAW TEXT OVER IMAGE
#https: // www.geeksforgeeks.org / python - opencv - cv2 - puttext - method /
