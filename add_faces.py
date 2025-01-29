import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')


def add_face():

    faces_data = []
    i =0
    name = input("Please enter your full name: ")
    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3,5)
        for (x,y,w,h) in faces:
            cropped_img = frame[y:y+h,x:x+w, :]
            resized_img= cv2.resize(cropped_img,(50,50))
            if len(faces_data) <=100 and i%3 ==0:
                faces_data.append(resized_img)
            i +=1
            cv2.putText(frame,str(len(faces_data)),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1)

        cv2.imshow('Student tracker',frame)
        k = cv2.waitKey(1)
        if k == ord('q') or len(faces_data) ==100:
            break


    video.release()
    cv2.destroyAllWindows()
    if len(faces_data) != 100:
        return
    else:
            
        faces_data = np.asarray(faces_data).reshape(100,-1)

        ind_name = [name] *100

        #for names

        if 'names.pkl' not in os.listdir('data/'):
            with open('data/names.pkl','wb') as f:
                pickle.dump(ind_name,f)


        else:
            with open('data/names.pkl', 'rb') as f:
                names = pickle.load(f)
            names = names + ind_name
            with open('data/names.pkl','wb') as f:
                pickle.dump(names,f)

        #for faces

        if 'faces.pkl' not in os.listdir('data/'):
            with open('data/faces.pkl','wb') as f:
                pickle.dump(faces_data,f)


        else:
            with open('data/faces.pkl', 'rb') as f:
                faces = pickle.load(f)
            faces = np.vstack((faces, faces_data))
            with open('data/faces.pkl','wb') as f:
                pickle.dump(faces,f)
                



if __name__ =="__main__":
    add_face()