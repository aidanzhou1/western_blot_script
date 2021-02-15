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




input_image = io.imread("example_image/01242020_1.jpeg")
input_image = rgb2gray(input_image)
print(input_image.shape)
temp = input_image


cv_image = img_as_ubyte(input_image)

# threshold1 = (input_image[int(0.2 * len(input_image)), int(0.25 * len(input_image[0]))])

# posx = int(0.25 * len(input_image[0]))
# posy = int(0.2 * len(input_image))

# for i in range(len(input_image)):
#     #this loops through all the values in each row (image[i]) fron left to right
#     for j in range(len(input_image[i])):
#         #if it is lighter than or equal to the threshold - a little buffer
#         #not sure if this step is necessary
#         if input_image[i,j] >= threshold1 - 0.2:
#             #subtract threshold from number
#             input_image[i, j] = input_image[i, j] + (1 - threshold1)
#             if (input_image[i, j] > 1):
#                 input_image[i,j] = 1


thresh = cv2.adaptiveThreshold(cv_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,101,10)


cv2.imshow('asdf',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

io.imshow(input_image)
plt.show()

# mnist = tf.keras.datasets.mnist

# #x is the images, y is the label 
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.softmax))

# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5)


# loss, accuracy = model.evaluate(x_test, y_test)
# print(loss)
# print(accuracy)