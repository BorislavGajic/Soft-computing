# import libraries here
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import matplotlib.pylab as plt
import os
import numpy as np
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # KNN
from joblib import dump, load

# keras

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.models import model_from_json

from sklearn import datasets
from sklearn.cluster import KMeans

emotions = ["contempt", "anger", "disgust", "happiness", "neutral", "surprise", "sadness"]
# inicijalizaclija dlib detektora (HOG)
detector = dlib.get_frontal_face_detector()
# ucitavanje pretreniranog modela za prepoznavanje karakteristicnih tacaka
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
clf_svm = SVC(kernel='linear', probability=True)

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


def get_shape(image):

    rects = detector(image, 1)

    # iteriramo kroz sve detekcije korak 1.
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        # odredjivanje kljucnih tacaka - korak 2
        shape = predictor(image, rect)
        # shape predstavlja 68 koordinata
        #shape = face_utils.shape_to_np(shape)  # konverzija u NumPy niz

        x_point=[]
        y_point=[]
        for j in range(17, 68):
            x_point.append(float(shape.part(j).x))
            y_point.append(float(shape.part(j).y))

        x_center=np.mean(x_point)
        y_center=np.mean(y_point)

        dist=[]
        for k in range(len(x_point)):
            dist.append(x_point[k] - x_center)
            dist.append((y_point[k] - y_center))


        # konvertovanje pravougaonika u bounding box koorinate
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # crtanje pravougaonika oko detektovanog lica
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ispis rednog broja detektovanog lica
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # crtanje kljucnih tacaka
#        for (x, y) in shape:
#            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        # plt.imshow(image)
        # plt.show()
    return dist

def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))



def train_or_load_facial_expression_recognition_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju
    model=None
    try:
        model=load('svm2.joblib')
    except:
        model=None
    if model == None:
        data=[]
        labels=[]
        for i in range(len(train_image_paths)):
            image=load_image(train_image_paths[i])
            data.append(get_shape(image))
            labels.append(train_image_labels[i])

        x = np.array(data)
        y_train = np.array(labels)

        #x_train=reshape_data(x)
        clf_svm.fit(x, y_train)
        dump(clf_svm, 'svm2.joblib')
        model=clf_svm

    return model


def extract_facial_expression_from_image(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje ekspresije lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati ekspresiju.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <String>  Naziv prediktovane klase (moguce vrednosti su: 'anger', 'contempt', 'disgust', 'happiness', 'neutral', 'sadness', 'surprise'
    """
    facial_expression = ""
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)

    image=load_image(image_path)
    #plt.imshow(image)
    #plt.show()
    shape=get_shape(image)
    #print(shape.shape)
    data=[]
    data.append(shape)
    x = np.array(data)
    #print(x.shape)

    #shape=reshape_data(x)
    temp = trained_model.predict(x)
    facial_expression=temp[0]
    return facial_expression
