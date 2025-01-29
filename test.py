import cv2
import pickle
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier


video = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
    
with open('data/faces.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(FACES,LABELS)


while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3,5)
    for (x,y,w,h) in faces:
        cropped_img = frame[y:y+h,x:x+w, :]
        resized_img= cv2.resize(cropped_img,(50,50)).flatten().reshape(1,-1)
        output = knn.predict(resized_img)
        #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame,str(output[0]),(x,y-15),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,255,255),1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1)

    cv2.imshow('Student tracker',frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
