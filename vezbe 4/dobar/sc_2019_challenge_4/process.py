# import libraries here

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
#from imblearn.over_sampling import RandomOverSampler
#%matplotlib inline
# prikaz vecih slika
matplotlib.rcParams['figure.figsize'] = 16,12

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def display_image(image):
    plt.imshow(image, 'gray')

def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def train_or_load_traffic_sign_model(train_positive_images_paths, train_negative_images_path, train_image_labels):
    """
    Procedura prima listu putanja do pozitivnih i negativnih fotografija za obucavanje, liste
    labela za svaku fotografiju iz pozitivne liste

    Procedura treba da istrenira model(e) i da ih sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model(e) ako on nisu istranirani, ili da ih samo ucita ako su prethodno
    istrenirani i ako se nalaze u folderu za serijalizaciju

    :param train_positive_images_paths: putanje do pozitivnih fotografija za obucavanje
    :param train_negative_images_path: putanje do negativnih fotografija za obucavanje
    :param train_image_labels: labele za pozitivne fotografije iz liste putanja za obucavanje - tip znaka
    :return: lista modela
    """
    # TODO - Istrenirati modele ako vec nisu istrenirani, ili ih samo ucitati iz foldera za serijalizaciju

    models = []
    try:
        models1 = load('svm.joblib')
        models2 = load('svm1.joblib')
        models.append(models1)
        models.append(models2)
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
            img = load_image(pos)
            img = cv2.resize(img, (75, 75), interpolation=cv2.INTER_NEAREST)
            hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                              img.shape[0] // cell_size[0] * cell_size[0]),
                                    _blockSize=(block_size[1] * cell_size[1],
                                                block_size[0] * cell_size[0]),
                                    _blockStride=(cell_size[1], cell_size[0]),
                                    _cellSize=(cell_size[1], cell_size[0]),
                                    _nbins=nbins)
            pos_features.append(hog.compute(img))
            labels.append(1)
        for neg in train_negative_images_path:
            img = load_image(neg)
            img = cv2.resize(img, (75, 75), interpolation=cv2.INTER_NEAREST)
            hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                              img.shape[0] // cell_size[0] * cell_size[0]),
                                    _blockSize=(block_size[1] * cell_size[1],
                                                block_size[0] * cell_size[0]),
                                    _blockStride=(cell_size[1], cell_size[0]),
                                    _cellSize=(cell_size[1], cell_size[0]),
                                    _nbins=nbins)
            neg_features.append(hog.compute(img))
            labels.append(0)

        pos_features = np.array(pos_features)
        x2 = pos_features
        # nsamples, nx, ny = pos_features.shape
        # d2_train_dataset1 = pos_features.reshape((nsamples, nx * ny))

        neg_features = np.array(neg_features)
        # nsamples, nx, ny = neg_features.shape
        # d2_train_dataset2 = neg_features.reshape((nsamples, nx * ny))

        x = np.vstack((pos_features, neg_features))
        x_train = reshape_data(x)

        y_train = np.array(labels)

        print('Train shape: ', x_train.shape, y_train.shape)

        #ros = RandomOverSampler(random_state=42)
        #x_train, y_train = ros.fit_resample(x_train,y_train)

        clf_svm = SVC(kernel='linear', probability=True)
        clf_svm.fit(x_train, y_train)
        dump(clf_svm, 'svm.joblib')
        models.append(clf_svm)
        print("Model 1 sacuvan")


        #2
        x_train = reshape_data(x2)
        y = np.array(train_image_labels)


        print('Train shape: ', x_train.shape, y.shape)

        clf_svm1 = SVC(kernel='linear', probability=True)
        clf_svm1.fit(x_train,y)
        dump(clf_svm1, 'svm1.joblib')
        models.append(clf_svm1)
        print("Model 2 sacuvan")

        # rects = hog.detectMultiScale(img, found, 0, Size(6,6), Size(32,32), 1.05, 2);

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

    detections = window_slide(image_path, trained_models)

    return detections


def window_slide(image_path, trained_models):
    # read the image and define the stepSize and window size
    # (width,height)
    image = load_image(image_path)
    image = cv2.resize(image, (1500, 1500), interpolation=cv2.INTER_NEAREST)
    stepSize = 75
    (w_width, w_height) = (75, 75)  # window size
    nbins = 9  # broj binova
    cell_size = (8, 8)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku

    model1 = trained_models[0]
    model2 = trained_models[1]

    detections = []

    for x in range(0, image.shape[1] - w_width, stepSize):
        for y in range(0, image.shape[0] - w_height, stepSize):
            window = image[x:x + w_width, y:y + w_height]
            # classify content of the window with your classifier and
            # determine if the window includes an object (cell) or not
            features = []

            hog = cv2.HOGDescriptor(_winSize=(window.shape[1] // cell_size[1] * cell_size[1],
                                              window.shape[0] // cell_size[0] * cell_size[0]),
                                    _blockSize=(block_size[1] * cell_size[1],
                                                block_size[0] * cell_size[0]),
                                    _blockStride=(cell_size[1], cell_size[0]),
                                    _cellSize=(cell_size[1], cell_size[0]),
                                    _nbins=nbins)

            features.append(hog.compute(window))
            x1 = np.array(features)
            x_train = reshape_data(x1)
            try:
                y_test_predict1 = model1.predict(x_train)
            except:
                continue

            #print(y_test_predict1)

            if (y_test_predict1 == 1):
                y_test_predict2 = model2.predict(x_train)
                # za 2 znaka primer povratne vrednosti[[10, 15, 20, 20, "ZABRANA"], [30, 40, 60, 70, "DRUGI"]]
                y_str = str(y_test_predict2)
                duzina = len(y_str)
                #print(y_str)
                result = []
                result.append(x)
                result.append(y)
                result.append(x + w_width)
                result.append(y + w_height)
                result.append(y_str[2:duzina - 2])
                detections.append(result)

    return detections
