from process import detect_traffic_signs_from_image, train_or_load_traffic_sign_model
import glob
import sys
import os

# ------------------------------------------------------------------
# Ovaj fajl ne menjati, da bi automatsko ocenjivanje bilo moguce
if len(sys.argv) > 1:
    TRAIN_DATASET_PATH = sys.argv[1]
else:
    TRAIN_DATASET_PATH = '.' + os.path.sep + 'dataset' + os.path.sep + 'train' + os.path.sep

TRAIN_DATASET_POSITIVE = TRAIN_DATASET_PATH + 'positive' + os.path.sep
TRAIN_DATASET_NEGATIVE = TRAIN_DATASET_PATH + 'negative' + os.path.sep

if len(sys.argv) > 1:
    VALIDATION_DATASET_PATH = sys.argv[2]
else:
    VALIDATION_DATASET_PATH = '.' + os.path.sep + 'dataset' + os.path.sep + 'validation' + os.path.sep
# -------------------------------------------------------------------

# # indeksiranje labela za brzu pretragu
label_dict_positive = dict()
with open(TRAIN_DATASET_POSITIVE + 'annotations.csv', 'r') as file:
    lines = file.readlines()
    for index, line in enumerate(lines):
        if index > 0:
            cols = line.replace('\n', '').split(';')
            image_path = cols[0]
            sign_type = cols[1]
            if image_path in label_dict_positive:
                label_dict_positive[image_path].append(sign_type)
            else:
                label_dict_positive[image_path] = [sign_type]

# priprema skupa podataka za metodu za treniranje
train_positive_image_paths = []
train_positive_image_labels = []
for image_name in os.listdir(TRAIN_DATASET_POSITIVE):
    if '.jpg' in image_name:
        train_positive_image_paths.append(os.path.join(TRAIN_DATASET_POSITIVE, image_name))
        train_positive_image_labels.append(label_dict_positive[image_name])

train_negative_image_paths = []
for image_name in os.listdir(TRAIN_DATASET_NEGATIVE):
    if '.jpg' in image_name:
        train_negative_image_paths.append(os.path.join(TRAIN_DATASET_NEGATIVE, image_name))
# istrenirati model(e) za detekciju i klasifikaciju saobracajnih znakova
model = train_or_load_traffic_sign_model(train_positive_image_paths, train_negative_image_paths,
                                         train_positive_image_labels)

# izvrsiti detekciju i klasifikaciju znakova iz validacionog skupa
processed_image_names = []
detected_traffic_signs = []

for image_path in glob.glob(VALIDATION_DATASET_PATH + "*.jpg"):
    image_directory, image_name = os.path.split(image_path)
    processed_image_names.append(image_name)
    detected_traffic_signs.append(detect_traffic_signs_from_image(model, image_path))

# -----------------------------------------------------------------
# Kreiranje fajla sa rezultatima detekcije za svaku sliku
result_file_contents = "image_name;x_min;y_min;x_max;y_max;sign_type\n"
for image_index, image_name in enumerate(processed_image_names):
    detection_results = detected_traffic_signs[image_index]
    for result in detection_results:
        x_min = int(result[0])
        y_min = int(result[1])
        x_max = int(result[2])
        y_max = int(result[3])
        sign_type = result[4]
        result_file_contents += "%s;%s;%s;%s;%s;%s\n" % (image_name, x_min, y_min, x_max, y_max, sign_type)

# sacuvaj formirane rezultate u csv fajl
with open('result.csv', 'w') as output_file:
    output_file.write(result_file_contents)

# ------------------------------------------------------------------
