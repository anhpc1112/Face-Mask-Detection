import cv2
import os
import numpy as np
from tqdm import tqdm
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

WITH_MASK_TRAIN = r"C:\Users\admin\Documents\Python\FaceDetection\Face Mask Dataset\Train\WithMask"
WITHOUT_MASK_TRAIN = r"C:\Users\admin\Documents\Python\FaceDetection\Face Mask Dataset\Train\WithoutMask"

WITH_MASK_VALID = r"C:\Users\admin\Documents\Python\FaceDetection\Face Mask Dataset\Validation\WithMask"
WITHOUT_MASK_VALID = r"C:\Users\admin\Documents\Python\FaceDetection\Face Mask Dataset\Validation\WithoutMask"

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

data_train, labels_train = load_data([WITH_MASK_TRAIN, WITHOUT_MASK_TRAIN])
random.shuffle(list(zip(data_train, labels_train)))
data_train = np.array(data_train)
labels_train = np.array(labels_train)

data_valid, labels_valid = load_data([WITH_MASK_VALID, WITHOUT_MASK_VALID])
random.shuffle(list(zip(data_valid, labels_valid)))
data_valid = np.array(data_valid)
labels_valid = np.array(labels_valid)

data_test, labels_test = load_data([WITH_MASK_TEST, WITHOUT_MASK_TEST])
random.shuffle(list(zip(data_test, labels_test)))
data_test = np.array(data_test)
labels_test = np.array(labels_test)


filepath = "weights--{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose = 1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

train_datagen = ImageDataGenerator(rescale=1.0/255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode = 'nearest')

def build_model():
    model = Sequential()
    model.add(Conv2D(100, (3,3), activation='relu', input_shape = (150,150,3)))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(100, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model


model = build_model()
model.compile(optimizer='adam', steps_per_execution=50, loss= 'binary_crossentropy', metrics=['acc'])
model.summary()

H = model.fit(train_datagen.flow(data_train, labels_train, batch_size=64), epochs=30, 
            validation_data=train_datagen.flow(data_valid, labels_valid, batch_size=64), callbacks=callbacks_list, verbose=1)

model.save('model.h5')









        


