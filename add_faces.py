import cv2
import pickle
import numpy as np
import os
import sqlite3
import argparse

# Database initialization
def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    # Students table
    c.execute('''CREATE TABLE IF NOT EXISTS students
                (student_id TEXT PRIMARY KEY,
                 name TEXT NOT NULL)''')
    
    # Classes table
    c.execute('''CREATE TABLE IF NOT EXISTS classes
                (class_id TEXT PRIMARY KEY,
                 class_name TEXT NOT NULL)''')
    
    # Enrollments table
    c.execute('''CREATE TABLE IF NOT EXISTS enrollments
                (enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 student_id TEXT,
                 class_id TEXT,
                 FOREIGN KEY(student_id) REFERENCES students(student_id),
                 FOREIGN KEY(class_id) REFERENCES classes(class_id))''')
    
    conn.commit()
    conn.close()

init_db()

video = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

def add_student(student_id, name):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO students (student_id, name) VALUES (?, ?)",
                 (student_id, name))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        print(f"Error: Student ID {student_id} already exists!")
        return False
    finally:
        conn.close()

def capture_faces(student_id, name):
    faces_data = []
    i = 0
    
    # Face capture loop
    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cropped_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(cropped_img, (50, 50))
            
            if len(faces_data) <= 100 and i % 3 == 0:
                faces_data.append(resized_img)
            
            i += 1
            cv2.putText(frame, f"Captured: {len(faces_data)}/100", 
                       (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

        cv2.imshow('Student Registration', frame)
        k = cv2.waitKey(1)
        if k == ord('q') or len(faces_data) == 100:
            break

    video.release()
    cv2.destroyAllWindows()

    if len(faces_data) != 100:
        print("Failed to capture enough face data!")
        return False

    # Save face data
    faces_data = np.asarray(faces_data).reshape(100, -1)
    student_ids = [student_id] * 100

    # Update face data files
    if 'ids.pkl' not in os.listdir('data/'):
        with open('data/ids.pkl', 'wb') as f:
            pickle.dump(student_ids, f)
        with open('data/faces.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open('data/ids.pkl', 'rb') as f:
            existing_ids = pickle.load(f)
        updated_ids = existing_ids + student_ids
        with open('data/ids.pkl', 'wb') as f:
            pickle.dump(updated_ids, f)
        
        with open('data/faces.pkl', 'rb') as f:
            existing_faces = pickle.load(f)
        updated_faces = np.vstack((existing_faces, faces_data))
        with open('data/faces.pkl', 'wb') as f:
            pickle.dump(updated_faces, f)

    return True

def create_class(class_id, class_name):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO classes VALUES (?, ?)", (class_id, class_name))
        conn.commit()
        print(f"Class {class_id} created successfully!")
        return True
    except sqlite3.IntegrityError:
        print(f"Error: Class ID {class_id} already exists!")
        return False
    finally:
        conn.close()

def enroll_student(student_id, class_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO enrollments (student_id, class_id) VALUES (?, ?)",
                 (student_id, class_id))
        conn.commit()
        print(f"Student {student_id} enrolled in {class_id}")
        return True
    except sqlite3.IntegrityError:
        print("Invalid student or class ID!")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Student Management System')
    parser.add_argument('--add-student', action='store_true', help='Register new student')
    parser.add_argument('--create-class', action='store_true', help='Create new class')
    parser.add_argument('--enroll', action='store_true', help='Enroll student in class')
    
    args = parser.parse_args()

    if args.add_student:
        student_id = input("Enter Student ID: ").strip()
        name = input("Enter Full Name: ").strip()
        if add_student(student_id, name):
            if capture_faces(student_id, name):
                print(f"Successfully registered {name} ({student_id})")

    elif args.create_class:
        class_id = input("Enter Class ID: ").strip()
        class_name = input("Enter Class Name: ").strip()
        create_class(class_id, class_name)

    elif args.enroll:
        student_id = input("Enter Student ID: ").strip()
        class_id = input("Enter Class ID: ").strip()
        enroll_student(student_id, class_id)

    else:
        print("Please specify an operation: --add-student, --create-class, or --enroll")