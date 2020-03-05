# import libraries here

import os
import numpy as np
import cv2 # OpenCV
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # KNN
#from skimage.feature import hog
from joblib import dump, load
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
# prikaz vecih slika
matplotlib.rcParams['figure.figsize'] = 16,12

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def display_image(image):
    plt.imshow(image, 'gray')


def train_or_load_traffic_sign_model(train_positive_images_paths, train_negative_images_path, train_image_labels):
    """
    Procedura prima listu putanja do pozitivnih i negativnih fotografija za obucavanje, liste
    labela za svaku fotografiju iz pozitivne liste, kao i putanju do foldera u koji treba sacuvati model(e) nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model(e) i da ih sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model(e) ako on nisu istranirani, ili da ih samo ucita ako su prethodno
    istrenirani i ako se nalaze u folderu za serijalizaciju

    :param train_positive_images_paths: putanje do pozitivnih fotografija za obucavanje
    :param train_negative_images_path: putanje do negativnih fotografija za obucavanje
    :param train_image_labels: labele za pozitivne fotografije iz liste putanja za obucavanje - tip znaka i tacne koordinate znaka
    :return: lista modela
    """
    # TODO - Istrenirati modele ako vec nisu istrenirani, ili ih samo ucitati iz foldera za serijalizaciju

    models = []
    try:
        models[0] = load('svm.joblib')
        models[1] = load('svm1.joblib')
    except:
        models = []

    if models == []:
        print("Obucavanje pocelo.")

        pos_features = []
        neg_features = []
        labels = []

        nbins = 9  # broj binova
        cell_size = (8, 8)  # broj piksela po celiji
        block_size = (3, 3)  # broj celija po bloku

        pos_imgs = []
        neg_imgs = []
        for pos in train_positive_images_paths:
            pimg = cv2.cvtColor(cv2.imread(pos), cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(pimg, cv2.COLOR_RGB2GRAY)
            reg = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_NEAREST)
            pos_imgs.append(reg)
            hogg = cv2.HOGDescriptor(_winSize=(reg.shape[1] // cell_size[1] * cell_size[1],
                                               reg.shape[0] // cell_size[0] * cell_size[0]),
                                     _blockSize=(block_size[1] * cell_size[1],
                                                block_size[0] * cell_size[0]),
                                     _blockStride=(cell_size[1], cell_size[0]),
                                     _cellSize=(cell_size[1], cell_size[0]),
                                     _nbins=nbins)
            #hogg.detectMultiScale(reg, found, 0, Size(6, 6), Size(32, 32), 1.05, 2);
            pos_features.append(hogg.compute(reg))
            labels.append(1)
        for neg in train_negative_images_path:
            nimg = cv2.cvtColor(cv2.imread(neg), cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(nimg, cv2.COLOR_RGB2GRAY)
            reg = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_NEAREST)
            neg_imgs.append(reg)

            hogg = cv2.HOGDescriptor(_winSize=(reg.shape[1] // cell_size[1] * cell_size[1],
                                               reg.shape[0] // cell_size[0] * cell_size[0]),
                                     _blockSize=(block_size[1] * cell_size[1],
                                                block_size[0] * cell_size[0]),
                                     _blockStride=(cell_size[1], cell_size[0]),
                                     _cellSize=(cell_size[1], cell_size[0]),
                                     _nbins=nbins)
            neg_features.append(hogg.compute(reg))
            labels.append(0)

        pos_features = np.array(pos_features)
        #nsamples, nx, ny = pos_features.shape
        #d2_train_dataset1 = pos_features.reshape((nsamples, nx * ny))

        neg_features = np.array(neg_features)
        #nsamples, nx, ny = neg_features.shape
        #d2_train_dataset2 = neg_features.reshape((nsamples, nx * ny))

        x = np.vstack((pos_features, neg_features))
        x_train = reshape_data(x)


        y_train = np.array(labels)

        print('Train shape: ', x_train.shape, y_train.shape)

        clf_svm = SVC(kernel='linear', probability=True)
        clf_svm.fit(x_train, y_train)
        dump(clf_svm, 'svm.joblib')
        models.append(clf_svm)
        print("Model 1 sacuvan")

        pos_imgs1 = []
        pos_features1 = []
        for pos in train_positive_images_paths:
            pimg = cv2.cvtColor(cv2.imread(pos), cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(pimg, cv2.COLOR_RGB2GRAY)
            reg = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_NEAREST)
            pos_imgs1.append(reg)
            hogg = cv2.HOGDescriptor(_winSize=(reg.shape[1] // cell_size[1] * cell_size[1],
                                               reg.shape[0] // cell_size[0] * cell_size[0]),
                                     _blockSize=(block_size[1] * cell_size[1],
                                                block_size[0] * cell_size[0]),
                                     _blockStride=(cell_size[1], cell_size[0]),
                                     _cellSize=(cell_size[1], cell_size[0]),
                                     _nbins=nbins)
            #hogg.detectMultiScale(reg, found, 0, Size(6, 6), Size(32, 32), 1.05, 2);
            pos_features1.append(hogg.compute(reg))

        pos_features1 = np.array(pos_features1)
        x_train = reshape_data(pos_features1)

        labels = []
        for label in train_image_labels:
            labels.append(label)
            #print(label)
        y_train = np.array(labels)
        print('Train shape: ', x_train.shape, y_train.shape)

        clf_svm1 = SVC(kernel='linear', probability=True)
        clf_svm1.fit(x_train, y_train)
        dump(clf_svm1, 'svm1.joblib')
        models.append(clf_svm1)
        print("Model 2 sacuvan")



        #rects = hog.detectMultiScale(img, found, 0, Size(6,6), Size(32,32), 1.05, 2);


    return models


def detect_traffic_signs_from_image(trained_models, image_path):
    """
    Procedura prima listu istreniranih modela za detekciju i klasifikaciju saobracajnih znakova i putanju do fotografije na kojoj
    se nalazi novi znakovi koje treda detektovati i klasifikovati

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_models: Istreniranih modela za detekciju i klasifikaciju saobracajnih znakova
    :param image_path: Putanja do fotografije sa koje treba detektovati 
    :return: Naziv prediktovanog tipa znaka, koordinate detektovanog znaka
    """
    #print(image_path)
    # TODO - Detektovati saobracajne znakove i vratiti listu detektovanih znakova:
    # za 2 znaka primer povratne vrednosti[[10, 15, 20, 20, "ZABRANA"], [30, 40, 60, 70, "DRUGI"]]
    detections = [[0, 0, 0, 0, "DRUGI"]]  # x_min, y_min, x_max, y_max, tip znaka
    return detections


def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))
