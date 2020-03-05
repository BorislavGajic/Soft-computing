# import libraries here
from builtins import print

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

def count_blood_cells(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj krvnih zrnaca.

    Ova procedura se poziva automatski iz main procedure i taj deo kod nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih krvnih zrnaca
    """
    blood_cell_count = 0
    # TODO - Prebrojati krvna zrnca i vratiti njihov broj kao povratnu vrednost ove procedure

    img_ucitan = cv2.imread(image_path)
    img_ucitan = cv2.cvtColor(img_ucitan, cv2.COLOR_BGR2RGB)

    img_gray = np.ndarray((img_ucitan.shape[0], img_ucitan.shape[1]))
    img_gray = 0.21 * img_ucitan[:, :, 0] + 0.72 * img_ucitan[:, :, 1] + 0.07 * img_ucitan[:, :, 2]
    img_gray = img_gray.astype('uint8')

    img_treshh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)

    #height, width = img_gray.shape[0:2]
    #x = range(0, 256)
    #y = np.zeros(256)

    #for i in range(0, height):
    #    for j in range(0, width):
    #        pixel = img_gray[i, j]
    #        y[pixel] += 1

    mali = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    veliki = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    img_tresh_dilacija = cv2.dilate(img_treshh, mali, iterations=1)
    img_erozija = cv2.erode(img_tresh_dilacija, veliki, iterations=1)


    img, contours, hiearchy = cv2.findContours(img_erozija, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img = img_ucitan.copy()
    contours_circle = []
    centers = []
    xs = []
    ys = []
    i = 0

    for contour in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        if 35 < radius:
            xs.append(x)
            ys.append(y)
            xs.sort()
            ys.sort()

            i = i + 1
            if i >= 2:
                j = 0
                while j < i - 1:
                    if abs(xs[j + 1] - xs[j]) < 10:
                        break
                    elif abs(ys[j + 1] - ys[j]) < 10:
                        break
                    else:
                        if centers.count(center) == 0:
                            # cv2.circle(img, center, int(radius), (0, 0, 255), 2)
                            contours_circle.append(contour)
                            centers.append(center)

                    j += 1
            else:
                # cv2.circle(img, center, int(radius), (0, 0, 255), 2)
                contours_circle.append(contour)
                centers.append(center)

    cv2.drawContours(img, contours_circle, -1, (255, 0, 0), 1)
    #plt.imshow(img)
    #plt.show()
    blood_cell_count = len(contours_circle)
    print('Ukupan broj regiona: %d' % len(contours_circle))

    return blood_cell_count
