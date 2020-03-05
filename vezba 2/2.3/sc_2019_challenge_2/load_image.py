from __future__ import print_function
#import potrebnih biblioteka
import cv2
import collections

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.models import model_from_json

#Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans


import numpy as np
import matplotlib.pylab as plt



def load_image(path):

    img_loaded=cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img_grey= cv2.cvtColor(img_loaded, cv2.COLOR_RGB2GRAY)
    ret, image_bin = cv2.threshold(img_grey, 127, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    img_invert = 255 - image_bin
    plt.imshow(img_invert, 'gray')
    plt.show()

    kernel = np.ones((3, 3))
    img_dil = cv2.dilate(image, kernel, iterations=1)
    img_ero = cv2.erode(image, kernel, iterations=1)

    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)