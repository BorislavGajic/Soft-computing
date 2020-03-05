# import libraries here

from __future__ import print_function
#import potrebnih biblioteka

import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

import matplotlib.pylab as pylab


def train_or_load_character_recognition_model(train_image_paths, serialization_folder):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta), kao i
    putanju do foldera u koji treba sacuvati model nakon sto se istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija alfabeta
    :param serialization_folder: folder u koji treba sacuvati serijalizovani model
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju
    print(len(train_image_paths))
    img_ucitan = cv2.cvtColor(cv2.imread(train_image_paths), cv2.COLOR_BGR2RGB)
    plt.imshow(img_ucitan)
    plt.show()

    img_gray = np.ndarray((img_ucitan.shape[0], img_ucitan.shape[1]))
    img_gray = 0.21 * img_ucitan[:, :, 0] + 0.72 * img_ucitan[:, :, 1] + 0.07 * img_ucitan[:, :, 2]
    img_gray = img_gray.astype('uint8')

    img_tresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 5)

    mali = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    veliki = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))

    img_tresh_dilacija = cv2.dilate(img_tresh, mali, iterations=1)
    img_erozija = cv2.erode(img_tresh_dilacija, veliki, iterations=1)

    img_invert = 255 - img_erozija

    img_resize = cv2.resize(img_invert,(28,28), interpolation = cv2.INTER_NEAREST)
    plt.imshow(img_resize)
    plt.show()
    img, contours, hierarchy = cv2.findContours(img_resize.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    plt.imshow(img)
    plt.show()
    '''sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
        regions_array = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area > 50 and h < 120 and h > 15 and w > 25:
                # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
                # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
                region = img_resize[y:y + h + 1, x:x + w + 1]
                regions_array.append([resize_region(region), (x, y, w, h)])
                cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
        regions_array = sorted(regions_array, key=lambda item: item[1][0])
        sorted_regions = [region[0] for region in regions_array]'''


    model = None
    return model


def extract_text_from_image(trained_model, image_path, vocabulary):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje znakova (karaktera), putanju do fotografije na kojoj
    se nalazi tekst za ekstrakciju i recnik svih poznatih reci koje se mogu naci na fotografiji.
    Procedura treba da ucita fotografiju sa prosledjene putanje, i da sa nje izvuce sav tekst koriscenjem
    openCV (detekcija karaktera) i prethodno istreniranog modela (prepoznavanje karaktera), i da vrati procitani tekst
    kao string.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba procitati tekst.
    :param vocabulary: <Dict> Recnik SVIH poznatih reci i ucestalost njihovog pojavljivanja u tekstu
    :return: <String>  Tekst procitan sa ulazne slike
    """
    extracted_text = ""
    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string

    return extracted_text
