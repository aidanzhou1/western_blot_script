import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.image as mpimg
from skimage import data, io
from skimage.color import rgb2gray
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

    while (1):
        cv2.namedWindow("img")
        cv2.createTrackbar('Threshold', 'img', thresholdglobal, 255, nothing)
        global ogwidth,ogheight, posx, posy
        original_image, skimage_image = skimagefilter(y, x)
        cv_image, width, height = skimagetocv(skimage_image)
        original_image, asdf,asd2 = skimagetocv(original_image)
        
        ogwidth = width
        ogheight = height

        thresh = threshold(cv_image, cv2.getTrackbarPos('Threshold', 'img'))
        contours = findcontours(cv_image, thresh)
        goodlist, contourlist,valuelist = countcontours(cv_image, contours)
        drawcontours = drawcontourlist(goodlist, contourlist, contours)

        # cv2.createTrackbar('Threshold', "main", 225, 255, None)    
        cv_imagergb = cv2.cvtColor(cv_image,cv2.COLOR_GRAY2RGB)

        cv2.drawContours(cv_imagergb, drawcontours, -1, (0, 245, 0), 3)
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
            thickness = 1
            
            # Using cv2.putText() method 
            print(drawcontours[k][0].shape)
            print(drawcontours[k][0][0][1])
            cv_imagergb = cv2.putText(cv_imagergb, str(round(valuelist[k], 7)), (drawcontours[k][0][0,0],drawcontours[k][0][0,1]), font, fontScale, color, thickness, cv2.LINE_AA) 
        print("fin")
        print(posx)
        print(posy)
        cv_imagergb = cv2.circle(cv_imagergb, (int(posx), int(posy)), radius=10, color=(0, 0, 255), thickness=-1)
        cv_imagergb = cv2.resize(cv_imagergb, (1920, 900))

        cv2.imshow('img', cv_imagergb)
        cv2.waitKey(0)
        thresholdglobal = cv2.getTrackbarPos('Threshold', 'img')

        cv2.destroyAllWindows()
        















        #TODO MAKE THE TEXTBIGGER AND ADJUST STARTING THRESHOLD






        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # # cv2.namedWindow('img')
        # # cv2.setMouseCallback('img',draw_circle, cv_image)
        
            

        
def draw_circle(event,x,y,flags, cv_image):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("XY:" + str (x) + "_ " + str(y))
        print("AAAAASSSSSSSSSSSSS" + str(cv_image[y][x]))
        main(y,x)
        
def skimagefilter(y, x):
    global posx, posy
    skimage_image = io.imread("example_image/05202020_3.jpeg")
    skimage_image = rotate(skimage_image, 90, resize=True)
    original_image = skimage_image
    skimage_image = rgb2gray(skimage_image)

    #reads the image
    # skimage_image = rgb2gray(io.imread("example_image/05142020_1.jpeg"))
    # skimage_image = rgb2gray(io.imread("example_image/9blackpixeltest.png"))


    # image[a,b] a is the #rows (yaxis) b is the #columns (xaxis)
    # this loops through all the rows from top to bottom

    global ogheight, ogwidth
    # # threshold is the background color number
    # # print('!!!!!!!!!!!!')
    # # print(y)
    # # print(x)
    # # print(skimage_image[int((y /960) * ogheight), int((x / 540) * ogwidth)])

    # threshold1 = (skimage_image[int((y /960) * ogheight), int((x / 540) * ogwidth)])
    threshold1 = (skimage_image[int(0.2 * len(skimage_image)), int(0.25 * len(skimage_image[0]))])
    height, width = skimage_image.shape
    print('!!!!!!!!asc' + str(height))
    print('!!!' + str(int(0.25 * len(skimage_image[0]))))
    print('!!!' + str(int(0.2 * len(skimage_image))))

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
    # FOR GRAYSCALE 1 IS PURE WHITE, 0 IS PURE BLACK
    return original_image, skimage_image

def skimagetocv(skimage_image):
    width, height = cv_image.shape[:2]
    print("imagewidth" + str(width))
    print("imageheioght" + str(height))

    return cv_image, width, height
def threshold(cv_image, threshold):
   
    # thresh = cv2.adaptiveThreshold(cv_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,41,10)
    ret, thresh = cv2.threshold(cv_image, 225, 255, cv2.THRESH_BINARY)
    return thresh

# def cannyedge(cv_image):
    
#     canny = cv2.canny(cv_image, 150, 250)
#     return canny

def findcontours(cv_image, thresh):
    # cv2.imshow('imasdfsg', thresh)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours


def countcontours(cv_image, contours):


    goodlist = np.array([])
    contourlist = np.array([])
    valuelist = np.array([])
    print(len(contours))
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
        #THIS FILTERS OUT ALL THE SMALL ONES< ADJUSTABLE NUMBER
        if (times > 20):
            print("checkpt1")
            print(sumblack / times)
            valuelist = np.append(valuelist, (sumblack/times))
            goodlist = np.append(goodlist, [int(i + 1)])
            contourlist = np.append(contourlist, [sumblack / times])
    return goodlist, contourlist, valuelist


def drawcontourlist(goodlist, contourlist, contours):
    drawcontours = []
    print("length" + str(len(goodlist)))
    for k in range(len(goodlist)):
        print("checkpoint2")
        print(goodlist[k])
        drawcontours.append(contours[int(goodlist[k])])
    return drawcontours
main(0,0)