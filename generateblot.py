import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
from skimage import data, io
from skimage.color import rgb2gray, gray2rgb
import cv2
from skimage import img_as_ubyte
from skimage.transform import rotate
import random



def testgetvalidcontourslikeversion3withthehopeofseeingtheirsizes():
    input_image = io.imread("example_image/01242020_1.jpeg")
    input_image = rgb2gray(input_image)
    cv_image = img_as_ubyte(input_image)
    print(cv_image.shape)
    thresh = cv2.adaptiveThreshold(cv_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,101,10)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #for every contour

    contour_values = np.array([])
    valid_contours = np.array([])

    for i in range(len(contours) - 1):
        mask = np.full_like(cv_image, 255)  # Create mask where white is what we want, black otherwise
        
        #Its first argument is source image, second argument is the contours which should be passed as a Python list, third argument is index of contours 
        # (useful when drawing individual contour. To draw all contours, pass -1) and remaining arguments are color, thickness etc.
        cv2.fillPoly(mask, pts =[contours[i + 1]], color=(0,0,0))
        out = np.full_like(cv_image, 255)  # Extract out the object and place into output image


        out[mask == 0] = cv_image[mask == 0]


        # Now crop
        (y, x) = np.where(mask == 0)
        (topy, topx) = (np.min(y), np.min(x))
        (bottomy, bottomx) = (np.max(y), np.max(x))
        out = out[topy:bottomy + 1, topx:bottomx + 1]
        if (out.size > 100):
            print(out.shape)
            valid_contours = np.append(valid_contours, contours[i])
            contour_values = np.append(contour_values, 10)


        cv2.drawContours(cv_image, contours, -1, (0, 245, 0), 3)

def testcropimagesandplaceindownload():

    input_image = io.imread("example_image/08262020_1.jpeg")
    input_image = rgb2gray(input_image)
    cv_image = img_as_ubyte(input_image)
    print(cv_image.shape)
    thresh = cv2.adaptiveThreshold(cv_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,101,30)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #for every contour

    contour_values = np.array([])
    valid_contours = np.array([])

    for i in range(len(contours) - 1):
        mask = np.full_like(cv_image, 255)  # Create mask where white is what we want, black otherwise
        
        #Its first argument is source image, second argument is the contours which should be passed as a Python list, third argument is index of contours 
        # (useful when drawing individual contour. To draw all contours, pass -1) and remaining arguments are color, thickness etc.
        cv2.fillPoly(mask, pts =[contours[i + 1]], color=(0,0,0))
        out = np.full_like(cv_image, 255)  # Extract out the object and place into output image
        out[mask == 0] = cv_image[mask == 0]

        # Now crop
        (y, x) = np.where(mask == 0)
        (topy, topx) = (np.min(y), np.min(x))
        (bottomy, bottomx) = (np.max(y), np.max(x))
        out = out[topy:bottomy + 1, topx:bottomx + 1]

        #TODO
        #Figure out a way here to screen for all the short and long ones. 
        if (out.size > 100 and out.size < 100000):
            # cv2.imshow('test', out)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite('C:\Repos\western_blot_script\pblotdownloads\pblot ' + str(i) + '.jpeg', out)
            print(out.shape)
            valid_contours = np.append(valid_contours, contours[i])
            contour_values = np.append(contour_values, 10)

def generateblot():

    #generate an all black array of floats
    array = np.full((40, 40),1.0)
    
    print(array.shape)

    
    #somehow generate it in a circular way so that the outside is always lighter than the inside and that it forms random shapes

    a = 10
    b = 35
    c = 10
    d = 35
    
    #generate inner blot
    for i in range(a + 5,b - 5):
        for j in range(c + 5, d - 5):
            value = random.uniform(0.2, 0.0)

            array[i,j] = value
    #generate outer blot

    for i in range
    #generate buffer




    # #for the height values between 20 and 30
    # for i in range(10,35):
    #     #for the horizontal values between 10 and 15
    #     for j in range(6,18):
    #         value = random.uniform(0.6, 0.0)
    #         array[i,j] = value

    print(array)
    print(array.shape)
    #has to be floats
    cv2.imshow("image", array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(array)    

# testcropimagesandplaceindownload()
generateblot()