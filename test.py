import cv2
import pickle
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import csv
import time
from datetime import datetime

import pyttsx3





def speaker(text, voice_id=None, rate=150):
    try:
        engine = pyttsx3.init()
        
        # Set properties
        engine.setProperty('rate', rate)
        
        if voice_id:
            engine.setProperty('voice', voice_id)
            
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

video = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
    
with open('data/faces.pkl', 'rb') as f:
    FACES = pickle.load(f)

if len(FACES.shape) == 1:
    # Calculate expected features: 50x50x3 = 7500
    FACES = FACES.reshape(-1, 7500)


print("FACES shape:", FACES.shape)  # Should be (200, 7500) for 2 students
print("LABELS length:", len(LABELS)) 

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(FACES,LABELS)
COL_NAMES = ['NAME', 'TIME']


backgroundImg = cv2.imread("static/backgroundImg.png")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3,5)
    for (x,y,w,h) in faces:
        cropped_img = frame[y:y+h,x:x+w, :]
        resized_img= cv2.resize(cropped_img,(50,50)).reshape(1,-1)
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        exist = os.path.isfile("Attendance/Attendance_"+date+".csv")

        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame,str(output[0]),(x,y-15),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,255,255),1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1)
        attendance = [str(output[0]), str(timestamp)]
    backgroundImg[162:162+480, 55:55+640] =frame


    cv2.imshow('Student tracker',backgroundImg)
    k = cv2.waitKey(1)
    if k == ord('o'):
        speaker("Attendace taken for student ")
        time.sleep(5)
      
        if exist:
            with open("Attendance/Attendance_"+date+".csv","+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/Attendance_"+date+".csv","+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)

            csvfile.close()

    if k == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
