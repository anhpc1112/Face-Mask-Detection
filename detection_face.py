from imutils.video import VideoStream
import numpy as numpy
import argparse
import imutils
import time
import cv2
from tensorflow.keras.models import load_model
import numpy as np
model = load_model("model.h5")
labels_dict = {0: 'without_mask', 1: 'with_mask'}
color_dict = {0: (0,0,255), 1:(0,255,0)}

PROTOTEXT_PATH = r"C:\Users\admin\Documents\Python\FaceDetection\deploy.prototxt.txt"
MODEL_PATH = r"C:\Users\admin\Documents\Python\FaceDetection\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(PROTOTEXT_PATH, MODEL_PATH)
vs = VideoStream(src=0).start()
time.sleep(0.5)
prevTime = 0
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=700)
    curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1/(sec)
    str = "FPS: %0.1f" % fps
    cv2.putText(frame, str, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    print(detections.shape)

    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence < 0.6:
            continue
        
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (startX, startY, endX, endY) = box.astype("int")

        face = frame[startY:endY, startX:endX]
        face = cv2.resize(face, (150,150))
        normalized = face/255.0
        reshaped = np.reshape(normalized, (1,150,150,3))
        reshaped = np.vstack([reshaped])
        result = model.predict(reshaped)
        label = np.argmax(result)

        text = "{}".format(labels_dict[label])

        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), color_dict[label], 2)

        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_dict[label], 2)
    
    cv2.imshow("Predict", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()