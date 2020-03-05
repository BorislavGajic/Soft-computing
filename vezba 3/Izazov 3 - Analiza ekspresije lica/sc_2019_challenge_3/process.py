# import libraries here



from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

import os
import numpy as np
import cv2 # OpenCV
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # KNN
from joblib import dump, load
import matplotlib
import matplotlib.pyplot as plt
# prikaz vecih slika
'''matplotlib.rcParams['figure.figsize'] = 16,12'''

emotions = ["contempt", "anger", "disgust", "happiness", "neutral", "surprise", "sadness"]
clf_svm = SVC(kernel='linear', probability=True)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

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
    model = None
    try:
        model = load('svm.joblib')
    except:
        model = None

    if model == None:
        print("Obucavanje pocelo.")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        xtrain=[]
        for img in train_image_paths:
            image = cv2.imread(img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            for (i, rect) in enumerate(rects):
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                xtrain.append(shape)
                print("Dimenzije prediktor matrice: {0}".format(shape.shape))  # 68 tacaka (x,y)
                print("Prva 3 elementa matrice")
                print(shape[:3])
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                for (x, y) in shape:
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                #display_image(image)


        x_train = np.array(xtrain)
        x_train = reshape_data(x_train)

        labels = []
        for label in train_image_labels:
            labels.append(label)
        # print(len(labels))


        y_train = np.array(labels)
        print('Train shape: ', x_train.shape, y_train.shape)

        clf_svm.fit(x_train, y_train)
        dump(clf_svm, 'svm.joblib')
        model = clf_svm
        print("Model sacuvan")

    print("Model prosledjen")
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

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    facial_expression = ""
    xtrain=[]


    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)  # konverzija u NumPy niz
        xtrain.append(shape)
        #print("Dimenzije prediktor matrice: {0}".format(shape.shape))  # 68 tacaka (x,y)
        #print("Prva 3 elementa matrice")
        #print(shape[:3])


    try:
        x_train = np.array(xtrain)
        x_train = reshape_data(x_train)
        temp = trained_model.predict(x_train)
        facial_expression = temp[0]
    except:
        facial_expression = emotions[0]


    #x_train = reshape_data(shape)
    '''data = []
    data.append(shape)
    x = np.array(data)
    temp = trained_model.predict(x)
    facial_expression = temp[0]'''






    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)

    return facial_expression


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def display_image(image):
    plt.imshow(image, 'gray')
    plt.show()



def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))