# import libraries here
import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections
# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import model_from_json

#Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans

#matplotlib inline
import numpy as np
import matplotlib.pylab as plt

from fuzzywuzzy import fuzz

plt.rcParams['figure.figsize'] = 16, 12 # za prikaz većih slika i plotova, zakomentarisati ako nije potrebno


def scale_to_range(image):
    return image / 255
def matrix_to_vector(image):
    return image.flatten()
def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann
def convert_output(outputs):
    return np.eye(len(outputs))
def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    #ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 125, 3)
    return image_bin
def image_bin_za_kose(image_gs):
    image_denoise = cv2.fastNlMeansDenoising(image_gs, None, 10, 7,21)
    #image_bin = cv2.adaptiveThreshold(image_gs, 215, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 215, 2)
    ret, image_bin = cv2.threshold(image_denoise, 128, 255, cv2.THRESH_OTSU)

    #ret, image_bin = cv2.threshold(image_gs,89 , 234, cv2.THRESH_BINARY)
    return image_bin
def invert(image):
    return 255-image
def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
def dilate(image):
    kernel = np.ones((3,3))
    return cv2.dilate(image, kernel, iterations=5)
def erode(image):
    kernel = np.ones((7,4))
    return cv2.erode(image, kernel, iterations=3)
def dilate_za_kose(image):
    kernel = np.ones((2,2))
    return cv2.dilate(image, kernel, iterations=4)
def erode_za_kose(image):
    kernel = np.ones((4,3))
    return cv2.erode(image, kernel, iterations=3)

def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

def serialize_ann(ann, path):
    # serijalizuj arhitekturu neuronske mreze u JSON fajl
    model_json = ann.to_json()
    with open(path+"/neuronska.json", "w") as json_file:
        json_file.write(model_json)
    # serijalizuj tezine u HDF5 fajl
    ann.save_weights(path+"/neuronska.h5")

def load_trained_ann(path):
    try:
        # Ucitaj JSON i kreiraj arhitekturu neuronske mreze na osnovu njega
        json_file = open(path+'/neuronska.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        # ucitaj tezine u prethodno kreirani model
        ann.load_weights(path+"/neuronska.h5")
        print("Istrenirani model uspesno ucitan.")
        return ann
    except Exception as e:
        # ako ucitavanje nije uspelo, verovatno model prethodno nije serijalizovan pa nema odakle da bude ucitan
        return None

def select_roi(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)

        if area > 1000 and h < 1000 and h > 100 and w < 500:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])  # X_next - (X_current + W_current)
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances

def select_roi_kosa_slova(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        rows, cols = img.shape
        rect = cv2.minAreaRect(contours[0])
        center = rect[0]
        angle = rect[2]

        rot = cv2.getRotationMatrix2D(center, angle - 90, 1)
        # image_orig = cv2.warpAffine(image_orig, rot, (rows, cols))

        if area > 1230 and h < 1000 and w < 500:
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)


    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])  # X_next - (X_current + W_current)
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances

def display_result(outputs, alphabet, k_means):
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

def create_ann():
    '''
    Implementirati veštačku neuronsku mrežu sa 28x28 ulaznih neurona i jednim skrivenim slojem od 128 neurona.
    Odrediti broj izlaznih neurona. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    # Postaviti slojeve neurona mreže 'ann'
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(60, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.025, momentum=0.9)
    #adam = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, amsgrad=False)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=1500, batch_size=1, verbose=0, shuffle=False)

    return ann

def obrada_slike(img):
    gray = image_gray(img)
    bin = image_bin(gray)
    ero = erode(bin)
    dil = dilate(ero)
    #BILO da vraca ero
    return dil

def obrada_kose_slike(img):
    gray = image_gray(img)
    bin = image_bin_za_kose(gray)
    ero = erode_za_kose(bin)
    dil = dilate_za_kose(ero)

    return dil

def levenshtein_ratio_and_distance(s, t, ratio_calc = False):

    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])

def train_or_load_character_recognition_model(train_image_paths, serialization_folder):
    # OBRADA PRVE - VELIKA SLOVA
    image1 = load_image(train_image_paths[0])
    obradjena1 = obrada_slike(image1)
    image_orig1, letters, region_distances = select_roi(image1, obradjena1)
    # DOBIJAM NAZAD LETTERS I RAZMAKE IZMEDJU SLOVA = REGION_DISTANCES

    # OBRADA DRUGE - MALA SLOVA, MALO DRUGACIJA OBRADA
    image2 = load_image(train_image_paths[1])
    obradjena2 = obrada_slike(image2)
    image_orig2, letters2, region_distances2 = select_roi(image2, obradjena2)

    letters = letters + letters2

    alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']

    inputs = prepare_for_ann(letters)
    outputs = convert_output(alphabet)
    # probaj da ucitas prethodno istreniran model
    ann = load_trained_ann(serialization_folder)

    # ako je ann=None, znaci da model nije ucitan u prethodnoj metodi i da je potrebno istrenirati novu mrezu
    if ann == None:
        print("Traniranje modela zapoceto.")
        ann = create_ann()
        ann = train_ann(ann, inputs, outputs)
        print("Treniranje modela zavrseno.")
        # serijalizuj novu mrezu nakon treniranja, da se ne trenira ponovo svaki put
        serialize_ann(ann, serialization_folder)

    # print(display_result(outputs, alphabet, k_means))

    model = ann
    return model

def fuzzy(str, vocabulary):
    rijeci = str.split(" ")
    retVal = ""
    for rijec in rijeci:
        if (len(rijec) > 2):
            maxRatio = 0
            maxKey = ""
            for key in vocabulary:
                if(len(rijec) == len(key)):
                    # ratio = levenshtein_ratio_and_distance(rijec.lower(),key.lower(),ratio_calc = True)
                    ratio = fuzz.token_sort_ratio(rijec.lower(), key.lower())
                    if (ratio > maxRatio):
                        maxRatio = ratio
                        maxKey = key
                if (maxRatio > 0.65):
                    retVal += " "
                    retVal += maxKey
                else:
                    retVal += " "
                    retVal += rijec
                    break;
        else:
            retVal += " "
            retVal += rijec
    return retVal

def extract_text_from_image(trained_model, image_path, vocabulary):
    img = load_image(image_path)
    obradjena = obrada_slike(img)
    image_orig, letters, region_distances = select_roi(img, obradjena)
    #display_image(obradjena)
    #plt.show()

    # Podešavanje centara grupa K-means algoritmom
    region_distances = np.array(region_distances).reshape(len(region_distances), 1)
    # Neophodno je da u K-means algoritam bude prosleđena matrica u kojoj vrste određuju elemente
    k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)

    #Ovo hvata loše obrađene slike (na kojima se ne mogu prepoznati razmaci izmedju kontura niti konture) - zbog tresholda
    try:
        k_means.fit(region_distances)
    except:
        obradjena = obrada_kose_slike(image_orig)
        #display_image(obradjena)
        image_orig, letters, region_distances = select_roi_kosa_slova(img, obradjena)

        #display_image(image_orig)
        #plt.show()

        region_distances = np.array(region_distances).reshape(len(region_distances), 1)
        k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
        try:
            k_means.fit(region_distances)
        except:
            return ""

    inputs = prepare_for_ann(letters)
    results = trained_model.predict(np.array(inputs, np.float32))

    alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                'Š', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's',
                'š', 't', 'u',
                'v', 'w', 'x', 'y', 'z', 'ž']

    str=display_result(results, alphabet, k_means)
    #print(str)

    retVal = fuzzy(str, vocabulary)

    #print(retVal)
    #plt.show()
    #retVal = str
    return retVal