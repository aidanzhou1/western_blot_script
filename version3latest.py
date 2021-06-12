import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.image as mpimg
from skimage import data, io
from skimage.color import rgb2gray, gray2rgb
import cv2
from skimage import img_as_ubyte
from skimage.transform import rotate


ogwidth = 2559

ogheight = 4200

thresholdglobal = 205

posx= 0
posy = 0


def nothing(x):
    # global threshold123
    # threshold123 = x
    print(x)
def main(y, x):
    global thresholdglobal

    global ogwidth,ogheight, posx, posy
    original_image, skimage_image = skimagefilter(y, x)
    cv_image, width, height = skimagetocv(skimage_image)
    original_image, asdf,asd2 = skimagetocv(original_image)
    original_image = gray2rgb(original_image)
    ogwidth = width
    ogheight = height

    #threshold doesn't really matter anymore because it's adaptive
    thresh = threshold(cv_image,205)
    # thresh = cannyedge(cv_image)
    contours = findcontours(thresh)
    goodlist, contourlist,valuelist = countcontours(cv_image, contours)
    drawcontours = drawcontourlist(goodlist, contourlist, contours)

    # cv2.createTrackbar('Threshold', "main", 225, 255, None)    
    cv_imagergb = cv2.cvtColor(cv_image,cv2.COLOR_GRAY2RGB)
    original_imagergb = original_image
    cv2.drawContours(original_imagergb, drawcontours, -1, (0, 245, 0), 3)
    
    for k in range(len(valuelist)):
        # Window name in which image is displayed 
        window_name = 'img'
        
        # font 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        
        # org 
        org = (50, 50) 
        
        # fontScale 
        fontScale = 1
        
        # Blue color in BGR 
        color = (0, 0, 255) 
        
        # Line thickness of 2 px 
        thickness = 2
        
        # Using cv2.putText() method 
        # print(drawcontours[k][0].shape)
        # print(drawcontours[k][0][0][1])
        original_imagergb = cv2.putText(original_imagergb, str(round(valuelist[k], 1)), (drawcontours[k][0][0,0],drawcontours[k][0][0,1]), font, fontScale, color, thickness, cv2.LINE_AA) 



    #circle from where the thing skimage bg source is drawn
    original_imagergb = cv2.circle(original_imagergb, (int(posx), int(posy)), radius=1, color=(0, 0, 255), thickness=-1)

    #resize the entire image
    original_imagergb = cv2.resize(original_imagergb, (1920, 900))

    # original_imagergb = cv2.resize(original_imagergb, (1920, 900))
    print("fin")

    cv2.imshow('img', original_imagergb)
    cv2.waitKey(0)
    thresholdglobal = cv2.getTrackbarPos('Threshold', 'img')

    cv2.destroyAllWindows()

        
        
        
def skimagefilter(y, x):
    global posx, posy
    # skimage_image = io.imread("example_image/testimage2.png")
    skimage_image = io.imread("example_image/05142020_1.jpeg")
    skimage_image = rgb2gray(skimage_image)
    original_image = skimage_image
    threshold1 = (skimage_image[0,0])
    height, width = skimage_image.shape

    posx = (int(0.25 * len(skimage_image[0])) / height ) * 540
    posy = (int(0.2 * len(skimage_image)) / width) * 960
    for i in range(len(skimage_image)):
        #this loops through all the values in each row (image[i]) fron left to right
        for j in range(len(skimage_image[i])):
            #if it is lighter than or equal to the threshold - a little buffer
            #not sure if this step is necessary
            if skimage_image[i,j] >= threshold1 - 0.2:
                #subtract threshold from number
                skimage_image[i, j] = skimage_image[i, j] + (1 - threshold1);
                if (skimage_image[i, j] > 1):
                    skimage_image[i,j] = 1
    return original_image, skimage_image

def skimagetocv(skimage_image):
    cv_image = img_as_ubyte(skimage_image)
    width, height = cv_image.shape[:2]
    return cv_image, width, height


def threshold(cv_image, threshold):
    thresh = cv2.adaptiveThreshold(cv_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,91,10)
    return thresh

def cannyedge(cv_image):

    canny = cv2.Canny(cv_image, 200, 240)
    cv2.imshow('asdsssfasdf', canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return canny


def findcontours(thresh):
 
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours


def countcontours(cv_image, contours):


    goodlist = np.array([])
    contourlist = np.array([])
    valuelist = np.array([])
    print("Number of Contours: " + str(len(contours)))
    #for every contour
    for i in range(len(contours) - 1):

        mask = np.full_like(cv_image, 255)  # Create mask where white is what we want, black otherwise
        #Its first argument is source image, second argument is the contours which should be passed as a Python list, third argument is index of contours 
        # (useful when drawing individual contour. To draw all contours, pass -1) and remaining arguments are color, thickness etc.
        print("Countour #:" + str(i))
        #skip the first contour
        mask = cv2.fillPoly(mask, pts =[contours[i + 1]], color=(0,0,0))
        out = np.full_like(cv_image, 255)  # Extract out the object and place into output image

        #where mask = 0 make out the real darkness
        out[mask == 0] = cv_image[mask == 0]
        print("size of out after masking: " + str(out.shape))
        print("size of mask after masking: " + str(mask.shape))

        # Now crop
        (y, x) = np.where(mask == 0)
        (topy, topx) = (np.min(y), np.min(x))
        (bottomy, bottomx) = (np.max(y), np.max(x))
        mask = mask[topy:bottomy, topx : bottomx]
        out = out[topy + 1:bottomy - 1, topx + 1:bottomx - 1]

        sumblack = 0
        times = 0
        for y in range(len(out)):
            for x in range(len(out[y])):
                if (mask.size > 0 and mask[y].size > 0):
                    if (mask[y,x] == 0 and not out[y,x] == 255):
                        sumblack = sumblack + out[y, x]
                        times = times + 1
                # else:
                    # print("This Pixel is not black so the sum was not added")
        
        #if the contour is big enough
        #THIS FILTERS OUT ALL THE SMALL ONES< ADJUSTABLE NUMBER
        if (times > 30):
            # print("checkpt1")
            # print(sumblack / times)
            valuelist = np.append(valuelist, (sumblack/times))
            goodlist = np.append(goodlist, [int(i + 1)])
            contourlist = np.append(contourlist, [sumblack / times])
    return goodlist, contourlist, valuelist


def drawcontourlist(goodlist, contourlist, contours):
    drawcontours = []
    print("Number of good contours: " + str(len(goodlist)))
    for k in range(len(goodlist)):
        print(goodlist[k])
        drawcontours.append(contours[int(goodlist[k])])
    return drawcontours
main(0,0)