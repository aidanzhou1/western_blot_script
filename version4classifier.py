import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.image as mpimg
from skimage import data, io
from skimage.color import rgb2gray, gray2rgb
import cv2
from skimage import img_as_ubyte
from skimage.transform import rotate
# import autokeras as ak
from keras import models
from keras.layers import core, convolutional, pooling
import os
#Steps I can look at
#CNN with multiple inputs such as "size of mask", "total darkness"
#increase data set size
#CREATE ARTIFICIAL DATASET USING PYTHON ARRAYS




#classifier doesn't work right now because the data doesn't overlap. There aren't data with the same values. 
#has to do with way I'm slicing the data. 
#also could have to do with  rounding. maybe round to 1 decimal or something

def inputImage():
    input_image = io.imread("input_images/01242020_1.jpeg")
    input_image = rgb2gray(input_image)
    print(input_image.shape)
    temp = input_image

    input_image = rotate(input_image, 90, resize=True)


    threshold1 = (input_image[int(0.2 * len(input_image)), int(0.25 * len(input_image[0]))])

    posx = int(0.25 * len(input_image[0]))
    posy = int(0.2 * len(input_image))

    for i in range(len(input_image)):
        #this loops through all the values in each row (image[i]) fron left to right
        for j in range(len(input_image[i])):
            #if it is lighter than or equal to the threshold - a little buffer
            #not sure if this step is necessary
            if input_image[i,j] >= threshold1 - 0.2:
                #subtract threshold from number
                input_image[i, j] = input_image[i, j] + (1 - threshold1)
                if (input_image[i, j] > 1):
                    input_image[i,j] = 1

    cv_image = img_as_ubyte(input_image)

    thresh = cv2.adaptiveThreshold(cv_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,101,10)
    return thresh

def loadImages():
    images = []
    values = []
    print("stuck on filename")

    # past = 38.01
    # current = 0
    # timesthrough = 0
    for filename in os.listdir('training_data_small_set'):
        img = cv2.imread(os.path.join('Training_data_small_set',filename))
        img = rgb2gray(img)
        img = cv2.resize(img, (30,30))
        value = float(filename[0:len(filename) - 4])

        # current = round(value,2)
        # print('running')
        # if (current == past):
        print(filename)
        # timesthrough+=1
        if img is not None:
            images.append(img)

            #round to the tenths place
            values.append(round(value,2))
            # else:
            #     print("failure")
        # else:
        #     print ("notpast")
        # if (timesthrough > 30):


        #     past = round(past + 0.01, 2)
        #     print(past)
        #     timesthrough = 0
            
        





    return images, values


def formatData (images, testimages):



    print("running formatData()")
    for i in range(18):
        input_image = io.imread("mltest/test" + str(i + 1) + ".JPG")
        input_image = rgb2gray(input_image)
        cv_image = img_as_ubyte(input_image)
        cv_image = cv2.resize(cv_image, (30,30))
        print("cv_image.shape" + str(cv_image.shape))
        if images == []:
            images = [cv_image]
            testimages = [cv_image]
        else:
            images = np.concatenate((images, [cv_image]), axis=0)

            print(images.shape)
            testimages = np.concatenate((testimages, [cv_image]), axis=0)
    
    return images, testimages


def runModel(images, testimages, labels, testlabels):
    print("running runModel()")
    print(images.shape)
    x_train = tf.keras.utils.normalize(images, axis=1)
    x_test = tf.keras.utils.normalize(testimages, axis=1)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(30, 30)))
    model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.softmax))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, labels, epochs=150)


    loss, accuracy = model.evaluate(x_test, testlabels)
    print(loss)
    print(accuracy)

    test(model)
    
    
def mainTODO():
    #get the values and images from generated_blots. 
    images,values = loadImages()
    images_train = []
    images_test = []
    values_train = []
    values_test = []
    for i in range(len(images)):
        if (not(i % 7 == 0)):
            images_train.append(images[i])
        elif (i % 7 == 0):
            images_test.append(images[i])
    for i in range(len(values)):
        if (not(i % 7 == 0)):
            values_train.append(values[i])
        elif (i % 7 == 0):
            values_test.append(values[i])
    images_train = np.array(images_train)
    images_test = np.array(images_test)
    values_train = np.array(values_train)
    values_test = np.array(values_test)
    # images_train = np.array(images[0:int(2 * len(images)/3)])
    # images_test = np.array(images[int(2 * len(images)/3): len(images)])
    # values_train = np.array(values[0:int(2*len(values)/3)])
    # values_test = np.array(values[int(2*len(values)/3): len(values)])


    cv2.imshow(str(values_train[24]), images_train[24])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    runModel(images_train, images_test, values_train, values_test)
    # cv2.imshow(str(values[2]), images[2])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows

def mainTest():
    labels = np.array([0, 205, 0, 0, 0, 127, 0, 195, 127, 139.5, 127, 0, 129, 0, 139, 127, 0, 139.5])
    images = []
    testlabels = np.array([0, 205, 0, 0, 0, 127, 0, 195, 127, 139, 127, 0, 129, 0, 139, 127, 0, 139])
    testimages = []
    images, testimages = formatData(images, testimages)

    for i in range (len(images)):
        cv2.imshow('img' + str (i), images[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    runModel(images, testimages, labels, testlabels)


def test(model):
    input_image = io.imread('mltest/tester.jpg')
    input_image = rgb2gray(input_image)
    cv_image = img_as_ubyte(input_image)
    cv_image = cv2.resize(cv_image, (30,30))
    
    
    thresh = cv2.adaptiveThreshold(cv_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,51,6)


    prediction = model.predict(np.array([thresh]))
    print(str(np.argmax(prediction)))
    


mainTODO()
    
