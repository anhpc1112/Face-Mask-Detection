from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
import cv2
import os
import numpy as np
from tqdm import tqdm
import random

WITH_MASK_TEST = r"C:\Users\admin\Documents\Python\FaceDetection\Face Mask Dataset\Test\WithMask"
WITHOUT_MASK_TEST = r"C:\Users\admin\Documents\Python\FaceDetection\Face Mask Dataset\Test\WithoutMask"


def load_data(LIST_PATH):
    data = []
    labels = []
    for path in LIST_PATH:
        for filename in tqdm(os.listdir(path)):
            image = cv2.imread(os.path.join(path, filename))
            image = cv2.resize(image, (150,150),3)
            data.append(image)
            if 'WithMask' in path:
                labels.append(np.array([0,1]))
            if 'WithoutMask' in path:
                labels.append(np.array([1,0]))
    return data, labels


data_test, labels_test = load_data([WITH_MASK_TEST, WITHOUT_MASK_TEST])
random.shuffle(list(zip(data_test, labels_test)))
data_test = np.array(data_test)
labels_test = np.array(labels_test)


model = load_model('model.h5')
targets_names = ['without_mask', 'with_mask']

y_pred = model.predict(data_test)

labels_true = []
for i in range(len(labels_test)):
    labels_true.append(np.argmax(labels_test[i]))
labels_pred = []
for i in range(len(y_pred)):
    labels_pred.append(np.argmax(y_pred[i]))

print(classification_report(labels_true, labels_pred, target_names=targets_names, digits=4))