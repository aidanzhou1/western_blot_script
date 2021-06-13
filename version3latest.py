import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.image as mpimg
from skimage import data, io
from skimage.viewer import ImageViewer
from skimage.color import rgb2gray, gray2rgb
import cv2
from skimage import img_as_ubyte
from skimage.transform import rotate
from matplotlib import pyplot as plt
import os
from os import path




'''
Takes file path of image, and converts it into 2D array. 
Opens Matplotlib plot viewer with an event listener for mouse click. 
'''
def pickImage(file_path, orientation, save_image):
    print('pickImage()')

    #read and convert to grayscale
    skimage_image = io.imread('input_images/' + file_path)
    skimage_image = rgb2gray(skimage_image)
    print(skimage_image.shape)
    original_image = skimage_image

    #rotate if necessary
    if (orientation == "vertical"):
        skimage_image = rotate(skimage_image, 90, resize=True)     
        original_image = rotate(original_image, 90, resize=True)     

    #create figure and include image as subplot
    ax = plt.gca()
    fig = plt.gcf()
    ax.imshow(skimage_image)
    fig.canvas.mpl_connect('button_press_event', lambda event:onclick(event,skimage_image, original_image, save_image, file_path))
    plt.show()


'''
Callback for event listener. Takes x,y position of mouse click and then calls main function. 
Coordinates of mouse click used to identify background color.
'''
def onclick(event, skimage_image, original_image, save_image, file_path):
    print('onclick()' + str([event.xdata, event.ydata]))

    #close the viewer
    plt.close('all')

    #call main function
    main(int(event.xdata), int(event.ydata), skimage_image, original_image, save_image, file_path)


'''
Main function. 
'''
def main(y, x, skimage_image, original_image, save_image, file_path):
    print('main()')

    #initial thresholding manually using sci-kit image. Removes any pixels lighter than the background color the user selected.
    skimage_image = skimageFilter(y, x, skimage_image)

    #converts from skimage to opencv (different formats)
    cv_image = skimageToCV(skimage_image)
    original_image = skimageToCV(original_image)

    # performs adaptive thresholding
    thresh = threshold(cv_image)

    # thresh = cannyedge(cv_image)

    #find contours of 
    contours = findContours(thresh)

    goodlist, valuelist = countContours(cv_image, contours)

    drawcontours = drawContourList(goodlist, contours)

    original_imagergb = gray2rgb(original_image)

    cv2.drawContours(original_imagergb, drawcontours, -1, (0, 245, 0), 3)
    
    for k in range(len(valuelist)):
        # Window name in which image is displayed 
        window_name = 'img'

        # font 
        font = cv2.FONT_HERSHEY_SIMPLEX 

        # org 
        org = (50, 50) 

        # fontScale 
        fontScale = 0.6

        # Blue color in BGR 
        color = (0, 0, 255) 

        # Line thickness of 2 px 
        thickness = 1

        # Using cv2.putText() method 
        original_imagergb = cv2.putText(original_imagergb, str(round(valuelist[k], 1)), (drawcontours[k][0][0,0],drawcontours[k][0][0,1]), font, fontScale, color, thickness, cv2.LINE_AA) 


    #circle from where the thing skimage bg source is drawn
    original_imagergb = cv2.circle(original_imagergb, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=-1)

    #resize the image
    original_imagergb = cv2.resize(original_imagergb,(1920,900) )

    #display
    cv2.imshow('img', original_imagergb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if (save_image):
        folder_path = 'H:\Repos\western_blot_script/results'
        if (not path.exists(folder_path)):
            os.mkdir(folder_path)

        #rename file
        size = len(file_path)
        mod_string = file_path[:size - 4]
        new_path = "results/" + mod_string + "(output_1).jpeg"
        
        #if filename already exists
        i = 2
        while (True):
            if (path.exists(new_path)):
                new_path = mod_string + "(output_" + str(i) + ").jpeg"
                i = i + 1
            else:
                break
        cv2.imwrite(new_path, original_imagergb)
        
    

'''
Takes a coordinate and an image input.
Takes the grayscale value of the pixel at the coordinate. 
Loops through each pixel in the image, and subtracts the threshold value from those that are lighter or equal to the background color. 

'''
def skimageFilter(y, x, skimage_image):
   
    #gets threshold grayscale value from x,y coordinates.
    threshold1 = (skimage_image[x,y])

    # posx = (int(0.25 * len(skimage_image[0])) / height ) * 540
    # posy = (int(0.2 * len(skimage_image)) / width) * 960

    #loops through each row
    for i in range(len(skimage_image)):
        #this loops through all the values in each row (image[i]) fron left to right
        for j in range(len(skimage_image[i])):
            #if it is lighter than or equal to the threshold - a little buffer
            #grayscale values, 1 is white and 0 is black
            if skimage_image[i,j] >= threshold1 - 0.3:
                #equivalent of "subtracting" threshold value (different because in reverse)
                skimage_image[i, j] = skimage_image[i, j] + (1 - threshold1)
                
                #grayscale values only go from 0 to 1 so capping. 
                if (skimage_image[i, j] > 1):
                    skimage_image[i,j] = 1

    return skimage_image


'''
Convert image from Sci-Kit Image comptabile ndarray of float64 to OpenCV compatible ndarray of uint8. 
'''
def skimageToCV(skimage_image):
    print("skimagetocv()")
    cv_image = img_as_ubyte(skimage_image)
    # print("imagewidth" + str(width))
    # print("imageheight" + str(height))
    
    return cv_image


'''
Adaptive thresholding
'''
def threshold(cv_image):
    #for adaptive Threshold, 101 is block radius and 10 is threshold constant
    thresh = cv2.adaptiveThreshold(cv_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,101,10)
    # ret, thresh = cv2.threshold(cv_image, threshold, 255, cv2.THRESH_BINARY)

    return thresh


def cannyEdge(cv_image):
    canny = cv2.Canny(cv_image, 200, 240)
    return canny


'''
Taking thresholded image and tracing contours around all objects. 
Contours are just a set of points that create a polygon to outline the image. 
'''
def findContours(thresh):
    print("findcontours()")
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours


'''
Takes in the image and the set of contours returned by findContours().
Loops through each one and determines the grayscale value of pixels inside. 
'''
def countContours(cv_image, contours):

    print("countcontours()")

    #array containing index of contours that are large enough to be actual blots
    goodlist = np.array([])
    #containing the average grayscale values of each "good" contour
    valuelist = np.array([])

    print(len(contours))
    #for every contour
    for i in range(len(contours) - 1):

        # Create mask where white is what we want, black otherwis
        mask = np.full_like(cv_image, 255) 
        
        #Its first argument is source image, second argument is the contours which should be passed as a Python list, third argument is index of contours 
        # (useful when drawing individual contour. To draw all contours, pass -1) and remaining arguments are color, thickness etc.
        cv2.fillPoly(mask, pts =[contours[i + 1]], color=(0,0,0))

        # Extract out the object and place into output image based on where mask is black
        out = np.full_like(cv_image, 255) 
        out[mask == 0] = cv_image[mask == 0]

        #crops output image to solely rectangle around blot area (to make counting more efficient)
        (y, x) = np.where(mask == 0)
        (topy, topx) = (np.min(y), np.min(x))
        (bottomy, bottomx) = (np.max(y), np.max(x))
        out = out[topy:bottomy + 1, topx:bottomx + 1]

        #sum to find average value
        sumblack = 0
        times = 0
        for k in range(len(out)):
            for j in range(len(out[k])):
                sumblack = sumblack + out[k, j]
                times = times + 1
        
        #if the contour is big enough -- filters out smaller contours
        sizeLimit = 30
        if (times > sizeLimit):
            valuelist = np.append(valuelist, (sumblack/times))
            #store index of "good" contours
            goodlist = np.append(goodlist, [int(i + 1)])

    return goodlist, valuelist


'''
Takes goodlist, an array containing the index of all the "good" contours, and creates an array with the actual good contours
'''
def drawContourList(goodlist, contours):
    print("drawcontourlist()")
    drawcontours = []
    for k in range(len(goodlist)):
        drawcontours.append(contours[int(goodlist[k])])
    return drawcontours


#start program
pickImage("08262020_1.jpeg", "horizontal", True)
