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
import autokeras as ak
from keras import models
from keras.layers import core, convolutional, pooling

#Steps I can look at
#CNN with multiple inputs such as "size of mask", "total darkness"
#increase data set size

labels = np.array([0, 205, 0, 0, 0, 127, 0, 195, 127, 139.5, 127, 0, 129, 0, 139, 127, 0, 139.5])
images = []
testlabels = np.array([0, 205, 0, 0, 0, 127, 0, 195, 127, 139, 127, 0, 129, 0, 139, 127, 0, 139])
testimages = []
print (labels.size)
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



model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=32, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=16, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, kernel_initializer='normal'))
model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
model.fit(images, labels, epochs=2000)
model.summary()


loss, accuracy = model.evaluate(testimages, testlabels)
print(loss)
print(accuracy)

input_image = io.imread('mltest/test15.JPG')
input_image = rgb2gray(input_image)
cv_image = img_as_ubyte(input_image)
cv_image = cv2.resize(cv_image, (30,30))
prediction = model.predict(np.array([cv_image]))
print("for mltest15: " + str(np.argmax(prediction)))