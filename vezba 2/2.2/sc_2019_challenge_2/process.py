# import libraries here

from __future__ import print_function
# import potrebnih biblioteka

import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections


# keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.models import model_from_json

from sklearn import datasets
from sklearn.cluster import KMeans

from fuzzywuzzy import process



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
    # print(train_image_paths);


    ann = load_trained_ann(serialization_folder)
    if ann == None:
        print("Traniranje modela zapoceto.")
        img_ucitan1 = cv2.cvtColor(cv2.imread(train_image_paths[0]), cv2.COLOR_BGR2RGB)
        img_ucitan2 = cv2.cvtColor(cv2.imread(train_image_paths[1]), cv2.COLOR_BGR2RGB)

        img_gray1 = cv2.cvtColor(img_ucitan1, cv2.COLOR_RGB2GRAY)

        img_gray2 = cv2.cvtColor(img_ucitan2, cv2.COLOR_RGB2GRAY)

        #    plt.imshow(img_gray1)
         #   plt.show()

        ret, image_bin1 = cv2.threshold(img_gray1, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


        img_invert1 = 255 - image_bin1


        ret2, image_bin2 = cv2.threshold(img_gray2, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_invert2 = 255 -image_bin2

        kernel = np.ones((6, 2))
        dil1 = cv2.dilate(img_invert1, kernel, iterations=4)
        ero1 = cv2.erode(dil1, kernel, iterations=3)


        dil2 = cv2.dilate(img_invert2, kernel, iterations=4)
        ero2 = cv2.erode(dil2, kernel, iterations=3)



        img, contours, hierarchy = cv2.findContours(ero1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        regions_array = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area > 100 and h < 500 and h > 100 and w >80 and w<200:
                region = ero1[y:y + h + 1, x:x + w + 1]
                #plt.imshow(region)
                #plt.show()
                reg = cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)
                regions_array.append([reg, (x, y, w, h)])
                cv2.rectangle(img_ucitan1, (x, y), (x + w, y + h), (0, 255, 0), 2)


        regions_array = sorted(regions_array, key=lambda item: item[1][0])
        sorted_regions = [region[0] for region in regions_array]
        #print(len(sorted_regions))

        img, contours2, hierarchy = cv2.findContours(ero2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions_array2 = []
        for contour2 in contours2:
            x, y, w, h = cv2.boundingRect(contour2)
            area2 = cv2.contourArea(contour2)
            if area2 > 100 and h < 500 and h > 100 and w >80 and w<150:
                region2 = ero2[y:y + h + 1, x:x + w + 1]
                #plt.imshow(region2)
                #plt.show()
                reg = cv2.resize(region2, (28, 28), interpolation=cv2.INTER_NEAREST)
                regions_array2.append([reg, (x, y, w, h)])
                cv2.rectangle(img_ucitan2, (x, y), (x + w, y + h), (0, 255, 0), 2)


        regions_array2 = sorted(regions_array2, key=lambda item: item[1][0])
        sorted_regions2 = [region2[0] for region2 in regions_array2]


        sorted_regions3 = sorted_regions+sorted_regions2;
        #print(len(sorted_regions3))



        ready_for_ann = []
        for region in sorted_regions3:
            scale = region / 255
            #plt.imshow(scale)
            #plt.show()
            ready_for_ann.append(scale.flatten())
            #print(len(ready_for_ann))

        alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L','M','N','O','P','Q','R','S','Š','T','U','V','W','X','Y','Z','Ž',
                    'a', 'b', 'c', 'č', 'ć','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','š','t','u','v','w','x','y','z','ž']

        nn_outputs = []
        for index in range(len(alphabet)):
            output = np.zeros(len(alphabet))
            output[index] = 1
            nn_outputs.append(output)
            izlazna_slova = np.array(nn_outputs)


        ann = Sequential()
        ann.add(Dense(128, input_dim = 784, activation='sigmoid'))
        ann.add(Dense(60, activation='sigmoid'))

        x_train =  np.array(ready_for_ann, np.float32)
        y_train = np.array(izlazna_slova, np.float32)




        sgd = SGD(lr=0.01, momentum=0.9)
        ann.compile(loss='mean_squared_error', optimizer=sgd)

        ann.fit(x_train, y_train, epochs=4000, batch_size=1, verbose=0, shuffle=False)
        print("Treniranje modela zavrseno.")

        #winner = max(enumerate(alphabet), key=lambda x: x[1])[0]

        '''result = []
            for output in y_train:
                result.append(alphabet[max(enumerate(output), key=lambda x: x[1])[0]])
                print(result)'''

        model_json = ann.to_json()
        with open(serialization_folder + "/neuronska.json", "w") as json_file:
            json_file.write(model_json)
        ann.save_weights(serialization_folder + "/neuronska.h5")
        print("Model je upisan.")

    model = ann
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
    ann = trained_model;

    img_ucitan=cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img_gray= cv2.cvtColor(img_ucitan, cv2.COLOR_RGB2GRAY)
    ret,image_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_invert = 255 - image_bin

    kernel = np.ones((7, 2))
    dil = cv2.dilate(img_invert, kernel, iterations=4)
    ero = cv2.erode(dil, kernel, iterations=3)
    #plt.imshow(ero,'gray')
    #plt.show()

    selected_regions, letters, distances = select_roi(img_ucitan.copy(), ero)
    #plt.imshow(selected_regions)
    #plt.show()


    distances = np.array(distances).reshape(len(distances), 1)
    k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
    try:
        k_means.fit(distances)
    except:
        denosi = cv2.fastNlMeansDenoising(img_gray, None, 10, 7,21)
        ret, image_bin11 = cv2.threshold(denosi, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_invert11 = 255-image_bin11

        kernel11 = np.ones((7, 2))
        dill11= cv2.dilate(img_invert11, kernel11, iterations=5)
        ero11 = cv2.erode(dill11, kernel11, iterations=4)
        selected_regions, letters, distances = select_roi(img_ucitan.copy(), ero11)
        #plt.imshow(selected_regions)
        #plt.show()

        distances = np.array(distances).reshape(len(distances), 1)
        k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
        try:
            k_means.fit(distances)
        except:
            return

    alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž',
                'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']
    inputs = prepare_for_ann(letters)
    results = ann.predict(np.array(inputs, np.float32))

    rezultat = display_result(results, alphabet, k_means)
    #print(rezultat)

    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string



    novi_vokabular = []
    for rec in vocabulary:
        novi_vokabular.append(rec)
    #print(novi_vokabular)
    novi_rezultat = rezultat.split(' ')
    #print(novi_rezultat)
    krajnji_string=''
    for ciljana_rec in novi_rezultat:
        rec = process.extractOne(ciljana_rec,novi_vokabular)
        prvo_slovo= rec[0]
        if not ciljana_rec[0].isupper():
            prvo_slovo = prvo_slovo.lower()
        if krajnji_string == '':
            krajnji_string+= prvo_slovo;
            # print(krajnji_string)
        else:
            krajnji_string +=' '+prvo_slovo
            #print(krajnji_string)

    #print(krajnji_string)
    extracted_text = krajnji_string

    return extracted_text


def load_trained_ann(serialization_folder):
    try:
        # Ucitaj JSON i kreiraj arhitekturu neuronske mreze na osnovu njega
        json_file = open(serialization_folder+'/neuronska.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        # ucitaj tezine u prethodno kreirani model
        ann.load_weights(serialization_folder+"/neuronska.h5")
        print("Istrenirani model uspesno ucitan.")
        return ann
    except Exception as e:
        # ako ucitavanje nije uspelo, verovatno model prethodno nije serijalizovan pa nema odakle da bude ucitan
        return None

def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')

def convert_output(outputs):
    return np.eye(len(outputs))
def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def select_roi(image_orig, image_bin):
    '''
    Funkcija kao u vežbi 2, iscrtava pravougaonike na originalnoj slici, pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sačuva rastojanja između susednih regiona.
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
    regions_array = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        region = image_bin[y:y+h+1,x:x+w+1];
        regions_array.append([resize_region(region), (x,y,w,h)])
        cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)

    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles)-1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index+1]
        distance = next_rect[0] - (current[0]+current[2]) #X_next - (X_current + W_current)
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances


def resize_region(region):
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    return resized


def matrix_to_vector(image):
    return image.flatten()
def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann


def display_result(outputs, alphabet, k_means):
    '''
    Funkcija određuje koja od grupa predstavlja razmak između reči, a koja između slova, i na osnovu
    toga formira string od elemenata pronađenih sa slike.
    Args:
        outputs: niz izlaza iz neuronske mreže.
        alphabet: niz karaktera koje je potrebno prepoznati
        kmeans: obučen kmeans objekat
    Return:
        Vraća formatiran string
    '''
    # Odrediti indeks grupe koja odgovara rastojanju između reči, pomoću vrednosti iz k_means.cluster_centers_
    w_space_group = max(enumerate(k_means.cluster_centers_), key = lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:,:]):
        # Iterativno dodavati prepoznate elemente kao u vežbi 2, alphabet[winner(output)]
        # Dodati space karakter u slučaju da odgovarajuće rastojanje između dva slova odgovara razmaku između reči.
        # U ovu svrhu, koristiti atribut niz k_means.labels_ koji sadrži sortirana rastojanja između susednih slova.
        if (k_means.labels_[idx] == w_space_group):
            result += ' '
        result += alphabet[winner(output)]
    return result

def scale_to_range(image):
    return image / 255