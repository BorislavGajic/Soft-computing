from sklearn.metrics import accuracy_score
from statistics import mean
import sys
import os
import pandas as pd
import glob


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# ------------------------------------------------------------------
# Ovaj fajl ne menjati, da bi automatsko ocenjivanje bilo moguce
if len(sys.argv) > 1:
    VALIDATION_DATASET_PATH = sys.argv[1]
else:
    VALIDATION_DATASET_PATH = '.' + os.path.sep + 'dataset' + os.path.sep + 'validation' + os.path.sep
# ------------------------------------------------------------------
RESULTS_PATH = './result.csv'

validation_results = pd.read_csv(VALIDATION_DATASET_PATH + 'annotations.csv', sep=";")

# read results file generated by the main.py
results_df = pd.read_csv(RESULTS_PATH, sep=";").sort_values(by=['image_name'])
classification_true_labels = []
classification_predicted_labels = []
iou_all = []

for image_path in glob.glob(VALIDATION_DATASET_PATH + "*.jpg"):
    image_directory, image_name = os.path.split(image_path)
    expected_results = validation_results[validation_results['image_name'] == image_name].sort_values(
        by=['x_min', 'y_min'])
    predicted_results = results_df[results_df['image_name'] == image_name].sort_values(by=['x_min', 'y_min'])
    predicted_count = predicted_results.shape[0]


    for i, row in enumerate(expected_results.iterrows()):
        row = row[1]
        bbox_true = row['x_min'], row['y_min'], row['x_max'], row['y_max']
        sign_true = row['sign_type']
        bbox_predicted = (0, 0, 0, 0)
        sign_predicted = 'DRUGI'

        if i < predicted_count:
            predicted_row = predicted_results.iloc[i]
            bbox_predicted = predicted_row['x_min'], predicted_row['y_min'], predicted_row['x_max'], predicted_row[
                'y_max']
            sign_predicted = predicted_row['sign_type']
        classification_true_labels.append(sign_true)
        classification_predicted_labels.append(sign_predicted)
        iou_all.append(bb_intersection_over_union(bbox_true, bbox_predicted))


percentage = accuracy_score(classification_true_labels, classification_predicted_labels) * 100
mean_iou = mean(iou_all)
print(percentage)
print(mean_iou)
