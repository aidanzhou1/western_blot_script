import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
from numpy.core.numeric import outer
from skimage import data, io
from skimage.color import rgb2gray, gray2rgb
import cv2
from skimage import img_as_ubyte
from skimage.transform import rotate
import random
import math


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

def dimensionScale(integer):
    scale = 1
    while (integer > 10):
        integer = round(integer / 10)
        scale += 1
    return (scale)

def generateblot(a,b,c,d, grayscale_range_max, grayscale_range_min, canvas_size):

    
    #generate an all black array of floats
    #array has to a np 32 array (meaning it is an array of floats) because otherwise grayscale will all round to 1
    array = np.full((canvas_size, canvas_size),1.0)

    #scale (i.e. tens vs. hundredths vs. thousandths) of the blot being generated
    #used to calibrate how much I need to increment the horizontal buffer by to make it still look round
    blot_scale = dimensionScale(d - c)

    #gap between the top and bottom endpoints (a,b) and the y values where the curve starts and where the rectangle ends
    buffer_vertical = int(a/2) + random.randint(1 , 10)
    #since buffer is added to top and subtracted from bottom, buffer has to be bigger than difference between top (a) and bottom (b)
    if (buffer_vertical > (int(b-a)/2)):
        buffer_vertical = int(1 * int((b-a)/2))

    
    #generate inner blot (rectangle)
    for y in range(a + buffer_vertical,b - buffer_vertical):
        for x in range(c, d):

            value = random.uniform(grayscale_range_max,grayscale_range_min)
            array[y,x] = value

    #generate outer blot
    
    #horizontal buffer used to create the curve. you draw less and less as you go towards the end points so that it curves and closes
    buffer_horizontal_changing = 0

    #going from top of rectangle upwards towards a (why it's reversed -- getting smaller)
    for y in reversed(range(a, a + buffer_vertical)):
        for x in range(c + buffer_horizontal_changing,d - buffer_horizontal_changing):
            value = random.uniform(grayscale_range_max,grayscale_range_min)
            array[y,x] = value
        
        #since horizontal buffer subtracted from both sides, it has to be always less than half the width
        if (buffer_horizontal_changing < int((d-c) / 2)):

            #increment is how much to increment the horizontal buffer by 
            increment = random.randrange(0,blot_scale)
            #when the y value is less than halfway (or near the top) of the curve, the horizontal buffer increments by more to make the curve flatter
            if (y < a + int(((a + buffer_vertical) - a)/2)):
                increment = random.randrange(int(blot_scale/2), blot_scale)
            buffer_horizontal_changing += increment
    
    #same as above but going from bottom of rectangle downwards towards b (no need for reverse)
    #^BUT changes to variables and directions of operations (i.e. which substracts what)
    buffer_horizontal_changing = 0
    for y in range (b-buffer_vertical, b):
        for x in range(c + buffer_horizontal_changing,d - buffer_horizontal_changing):
            value = random.uniform(grayscale_range_max,grayscale_range_min)
            array[y,x] = value
        if (buffer_horizontal_changing < int((d-c) / 2)):
            increment = random.randrange(0,blot_scale)

            #when the y value is more than halfway (or near the bottom) of the curve, the horizontal buffer increments by more to make the curve flatter
            if (y > (b - buffer_vertical) +  int((b - (b - buffer_vertical))/2)):
                increment = random.randrange(int(blot_scale/2), blot_scale)
            buffer_horizontal_changing += increment

    #at this moment array has to still be a float otherwise it will all be 1
    # cv2.imshow("image", array)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return (array)

def calculateBlotValue(array):
    sum = 0
    timesthrough = 0
    #array is a 2d array of ints from 0 to 255
    for y in range(len(array)):
        for x in range(len(array[y])):
            
            pixel_value = array[y,x]
            #filte rout the white ones because they aren't part of the blot
            if (pixel_value < 255):
                sum += pixel_value
                timesthrough += 1
    
    return (sum/timesthrough)


def singleRun(a, b, c, d, grayscale_range_max, grayscale_range_min):


    # #somehow generate it in a circular way so that the outside is always lighter than the inside and that it forms random shapes
    # a = 50 #vertical top
    # b = 105 #vertical bottom
    # c = 20 #horizontal left
    # d = 120 #horizontal right

    # #lightest value of blot
    # grayscale_range_max = 0.21
    # #darkest value of blot
    # grayscale_range_min = 0.1

    canvas_size = 150



    for i in range(1,2):
        array =  generateblot(a,b,c,d,grayscale_range_max, grayscale_range_min, canvas_size)



        #convert from floating point numbers to ints. 
        #this essentially converts the array from a 32-bit floating point to an 8bit int or 16bit int (basically makes it an int array)
        #https://stackoverflow.com/questions/37026582/saving-an-image-with-imwrite-in-opencv-writes-all-black-but-imshow-shows-correct/37027314
        array1 = array * 255

        #find the blot value of each one to a T
        average_value = calculateBlotValue(array1)
        print(average_value)
        cv2.imwrite('generated_blots2/' + str( round(average_value,6) ) + '.png', array1)
    

def main():

    for a in range(15,20):
        for b in range (45, 50):
            for c in range(15,20):
                for d in range(135,140):
                    for grayscale_range_max in range(20,25):
                        for grayscale_range_min in range(25,30):
                            print(str(a * 2) + "," +str(b * 2) + "," +str(c) + "," +str (d) + "," +str (grayscale_range_max * 2/100) + "," + str (grayscale_range_min /100) + ")")
                            singleRun(a * 2,b * 2,c,d,((grayscale_range_max * 2)/100), (grayscale_range_min/100))


main()