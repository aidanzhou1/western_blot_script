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
        fontScale = 0.2
        
        # Blue color in BGR 
        color = (0, 0, 255) 
        
        # Line thickness of 2 px 
        thickness = 1
        
        # Using cv2.putText() method 
        # print(drawcontours[k][0].shape)
        # print(drawcontours[k][0][0][1])
        original_imagergb = cv2.putText(original_imagergb, str(round(valuelist[k], 1)), (drawcontours[k][0][0,0],drawcontours[k][0][0,1]), font, fontScale, color, thickness, cv2.LINE_AA) 



    #circle from where the thing skimage bg source is drawn
    original_imagergb = cv2.circle(original_imagergb, (int(posx), int(posy)), radius=1, color=(0, 0, 255), thickness=-1)

    #resize the entire image
    original_imagergb = cv2.resize(original_imagergb, (500, 300))

    # original_imagergb = cv2.resize(original_imagergb, (1920, 900))
    print("fin")

    cv2.imshow('img', original_imagergb)
    cv2.waitKey(0)
    thresholdglobal = cv2.getTrackbarPos('Threshold', 'img')

    cv2.destroyAllWindows()
    


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
    # skimage_image = io.imread("example_image/03102020_2.jpeg")
    skimage_image = io.imread("example_image/01242020snippet.JPG")
    skimage_image = rgb2gray(skimage_image)




    # skimage_image = rotate(skimage_image, 90, resize=True)





    original_image = skimage_image

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

    threshold1 = (skimage_image[0,0])
    # threshold1 = (skimage_image[int(0.2 * len(skimage_image)), int(0.25 * len(skimage_image[0]))])
    height, width = skimage_image.shape
    print("Image Height:" + str(height))
    # print('!!!' + str(int(0.25 * len(skimage_image[0]))))
    # print('!!!' + str(int(0.2 * len(skimage_image))))

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

    cv_image = img_as_ubyte(skimage_image)
    width, height = cv_image.shape[:2]


    return cv_image, width, height


def threshold(cv_image, threshold):
    thresh = cv2.adaptiveThreshold(cv_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,101,10)
    # ret, thresh = cv2.threshold(cv_image, threshold, 255, cv2.THRESH_BINARY)
    # cv2.imshow('imasdfsg', thresh)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
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
        print("Primary size of mask: " + str(mask.shape))
        #skip the first contour
        mask = cv2.fillPoly(mask, pts =[contours[i + 1]], color=(0,0,0))
        print("size of mask after fillpoly: " + str(mask.shape))

        out = np.full_like(cv_image, 255)  # Extract out the object and place into output image
        print("size of out: " + str(out.shape))

        
        #where mask = 0 make out the real darkness
        out[mask == 0] = cv_image[mask == 0]
        print("size of out after masking: " + str(out.shape))
        print("size of mask after masking: " + str(mask.shape))
        

        #TODO
        #MAKE SURE THIS FUNCTION IS ACCURATE
        #RIGHT NOW BLACKINMASK HAS MORE BLACK POINTS THAN BLACK IN OUT
        #OUT IS SMALLER FOR SOME REASON
        #TODO


        # Now crop
        (y, x) = np.where(mask == 0)
        (topy, topx) = (np.min(y), np.min(x))
        (bottomy, bottomx) = (np.max(y), np.max(x))


        #NOTES
        #no +1 on any: 12 10 17
        #+1 on both: 10 9 14 
        #+1 just on out: 10-9-14
        #OUT IS A SQUARE WITH THE SHAPE INSIDE.

        mask = mask[topy:bottomy, topx : bottomx]
        out = out[topy + 1:bottomy - 1, topx + 1:bottomx - 1]

        try:
            mask = cv2.resize(mask, (500, 300))
            out = cv2.resize(out, (500,300))
        except Exception as e:
            pass
            
        
        # if ( i == 2):
            
        #     blackinmask = 0
        #     blackinout = 0
        #     for v in range (len(mask)):
        #         for j in range(len(mask[i])):
        #             if (mask[v,j] == 0):
        #                 blackinmask = blackinmask + 1
        #     for v in range (len(out)):
        #         for j in range(len(out[i])):
        #             if (out[v,j] == 0):
        #                 blackinout = blackinout + 1
        
        #     print("blackinmask" + str(blackinmask))
        #     print("blackinout" + str(blackinout))
        sumblack = 0
        times = 0
        # mask = cv2.resize(mask, (1920, 900))
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # out = cv2.resize(out, (1920, 900))
        # cv2.imshow('asdfasdfasdf', out)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        
        for y in range(len(out)):
            for x in range(len(out[y])):
                # print("Mask[y,x]: " + str(mask[y,x]))
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