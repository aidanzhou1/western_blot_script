from numpy.core.numeric import outer
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.image as mpimg
from skimage import data, io
from skimage.color import rgb2gray, gray2rgb
from cv2 import cv2
from skimage import img_as_ubyte
from skimage.transform import rotate
# import autokeras as ak
from keras import models
from keras.layers import core, convolutional, pooling
import os
import keras

def loadImages():
    images = []
    labels = []

    past = 38.01
    current = 0
    timesthrough = 0
    i = 0
    print("going through band files now")
    for filename in os.listdir('training_data_smalltotiny_set'):
        img = cv2.imread(os.path.join('Training_data_smalltotiny_set',filename))
        img = cv2.resize(img, (28,28))
        # cv2.imshow('img', img)
        # cv2.waitKey(0) 
        label = 0 #band
        # if i >100:
        #     label = 3
        # print(filename)
        if img is not None:
            images.append(img)

            #round to the tenths place
            labels.append(label)
        i+=1
       
        

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)

    x_traingray = x_train.reshape((60000, 28, 28, 1))
    x_testgray = x_test.reshape((10000, 28, 28, 1))

    # Convert images and store them in X3
    x_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_traingray),name=None)
    x_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_testgray),name=None)

    x_train = x_train[0:10000]
    x_test = x_test[0:5000]

    label2 = []


    for i in range(15000):
        label2.append(1) #teXt
        # print('appending labels')
    images = np.array(images)
    print(x_train[12])

    #invert mnist blacks and whites
    for image in x_train:
        image = gray2rgb(image)
        image = ~image
#         cv2.imshow('image', np.uint8(image))
#         cv2.waitKey(0)
    for image in x_test:
        image = gray2rgb(image)
        image = ~image



    print(x_train.shape)
    print(type(x_train))
    print(type(x_test))
    print(type(images))
    print('images before appending shape' + str(images.shape))
    print('x_train before appending shape' + str(x_train.shape))
    print('x_test before appending shape' + str(x_test.shape))
    print(images[1].shape)
    images = np.concatenate((images, x_train), axis = 0)
    images = np.concatenate((images, x_test), axis = 0)


    print('images after appending' + str(images.shape))
    print(str(len(label2)))
    labels = np.array(labels)
    label2 = np.array(label2)
    labels = np.append(labels, label2)
    print('labels shape after appending should be 71001' + str(labels.shape))



    # cv2.imshow('images[24]' + str (labels[24]), images[24])
    # cv2.waitKey(0)
    # cv2.imshow('images[10000]' + str (labels[10000]), images[10000])
    # cv2.waitKey(0)
    # cv2.imshow('images[80000]' + str (labels[80000]), images[80000])
    # cv2.waitKey(0)
    # cv2.imshow('images[2]' + str (labels[2]), images[2])
    # cv2.waitKey(0)
    # cv2.imshow('images[77221]' + str (labels[77221]), images[77221])
    # cv2.waitKey(0)
    # cv2.imshow('images[77242]' + str (labels[77242]), images[77242])
    # cv2.waitKey(0)
    # cv2.imshow('images[6242]' + str (labels[6242]), images[6242])
    # cv2.waitKey(0)
    # cv2.imshow('images[46]' + str (labels[46]), images[46])
    # cv2.waitKey(0)
    # cv2.imshow('images[9000]' + str (labels[9000]), images[9000])
    # cv2.waitKey(0)
    # cv2.imshow('images[33]' + str (labels[33]), images[33])
    # cv2.waitKey(0)
    return images, labels




def runModel(images, testimages, labels, testlabels):
    print("running runModel()")

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(56, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(56, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=36, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=6, activation=tf.nn.softmax))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # print("x_train")
    # print(x_train.shape)
    # print("labels")
    # print(labels)

    # print('x_test')
    # print(x_test.shape)
    # print('testlabels')
    # print(testlabels)
    # print(testlabels[80])
    # print(testlabels[8])

    # print(testlabels[18])


    history = model.fit(images, labels, epochs=4, validation_data=(testimages, testlabels))


    # loss, accuracy = model.evaluate(testimages, testlabels)

    
    # print(loss)
    # print(accuracy)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['loss'], label = 'loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Evaluation')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()
    test(model)
    model.save("bandsvstext")
    
    
def mainTODO():
    # #get the values and images from generated_blots. 
    # images,values = loadImages()
    # images_train = []
    # images_test = []
    # values_train = []
    # values_test = []

    # print("Length Images" + str(len(images)))
    # print("Length Values" + str(len(values)))
    # print(values)
    # for i in range(len(images)):
    #     if (not(i % 7 == 0)):
    #         images_train.append(images[i])
    #     elif (i % 7 == 0):
    #         images_test.append(images[i])
    # for i in range(len(values)):
    #     if (not(i % 7 == 0)):
    #         values_train.append(values[i])
    #     elif (i % 7 == 0):
    #         values_test.append(values[i])
    # images_train = np.array(images_train)
    # images_test = np.array(images_test)
    # values_train = np.array(values_train)
    # values_test = np.array(values_test)

    # print("_")
    # print("_")
    # print("_")
    # print("images and then values")

    # print(images_test)
    # print(values_test)
    # # images_train = np.array(images[0:int(2 * len(images)/3)])
    # # images_test = np.array(images[int(2 * len(images)/3): len(images)])
    # # values_train = np.array(values[0:int(2*len(values)/3)])
    # # values_test = np.array(values[int(2*len(values)/3): len(values)])


    # # cv2.imshow(str(values_train[24]), images_train[24])
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # runModel(images_train, images_test, values_train, values_test)
    # # cv2.imshow(str(values[2]), images[2])
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows



    images,values = loadImages()
    assert len(images) == len (values)
    p = np.random.permutation(len(images))
    images = images[p]
    values = values[p]
    print(values)
    testimages = images[5 * int(len(images)/6):len(images)]
    testvalues = values[5 * int(len(values)/6):len(values)]
    images = images[0: 5 * int(len(images)/6)]
    values = values[0 : 5 * int(len(values)/6)]

    (print('images shape 11111' + str(images.shape)))
    (print('values shape 11111' + str(values.shape)))
    (print('testimages shape 11111' + str(testimages.shape)))

    (print('testvalues shape 11111' + str(testvalues.shape)))


    runModel(images, testimages, values, testvalues)



def test(model):


    input_image = io.imread('mltest/test622.jpg')
    input_image = rgb2gray(input_image)
    cv_image = img_as_ubyte(input_image)
    cv_image = cv2.resize(cv_image, (28,28))
    
    
    thresh = cv2.adaptiveThreshold(cv_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,51,6)

    input_image =  cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)

    prediction = model.predict(np.array([input_image]))
    print('prediction' + str(prediction))
    print('prediction shape' + str(prediction.shape))
    cv2.imshow('thresh', np.array(input_image))
    cv2.waitKey(0)
    output = np.argmax(prediction)

    if output == 0:
        print('band')
    if output == 1:
        print('text')

    input_image1 = io.imread('mltest/3.png')

    input_image1 = rgb2gray(input_image1)
    cv_image1 = img_as_ubyte(input_image1)
    cv_image1 = cv2.resize(cv_image1, (28,28))
    
    cv2.imshow('3', cv_image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    thresh1 = cv2.adaptiveThreshold(cv_image1,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,51,6)

    input_image2 = cv2.cvtColor(thresh1,cv2.COLOR_GRAY2RGB)

    prediction = model.predict(np.array([input_image2]))
    print(prediction)
    # prediction = np.bincount(prediction[: ,0].astype(int))
    output = np.argmax(prediction)

    print(output)
    if output == 0:
        print('band')
    if output == 1:
        print('text')



    input_image3 = io.imread('mltest/bandtest.jpeg')
    input_image3 = rgb2gray(input_image3)
    cv_image3 = img_as_ubyte(input_image3)
    cv_image3 = cv2.resize(cv_image3, (28,28))
    
    
    thresh2 = cv2.adaptiveThreshold(cv_image3,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,51,6)

    input_image3 =  cv2.cvtColor(thresh2,cv2.COLOR_GRAY2RGB)

    prediction = model.predict(np.array([input_image3]))
    print('prediction' + str(prediction))
    print('prediction shape' + str(prediction.shape))
    cv2.imshow('thresh', np.array(input_image3))
    cv2.waitKey(0)
    output = np.argmax(prediction)

    if output == 0:
        print('band')
    if output == 1:
        print('text')


mainTODO()
    
