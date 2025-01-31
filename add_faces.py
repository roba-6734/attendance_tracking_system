import cv2
import pickle
import numpy as np
import os
import sqlite3
import argparse
from reset_db import init_db

# Initialize database safely
init_db(safe=True)

video = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

def validate_student(student_id):
    """Check if student exists in database"""
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT student_id FROM students WHERE student_id=?", (student_id,))
    exists = c.fetchone()
    conn.close()
    return exists is not None

def atomic_registration(student_id, name):
    """Handle complete student registration atomically"""
    conn = sqlite3.connect('attendance.db')
    conn.execute("BEGIN TRANSACTION")
    
    try:
        if validate_student(student_id):
            print(f"Error: Student ID {student_id} already exists!")
            return False
            
        c = conn.cursor()
        c.execute("INSERT INTO students VALUES (?, ?)", (student_id, name))
        
        faces_data = []
        i = 0
        try:
            while len(faces_data) < 200:  # Collect 200 samples with augmentation
                ret, frame = video.read()
                if not ret:
                    raise IOError("Camera error")

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=6,
                    minSize=(150, 150),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                for (x, y, w, h) in faces:
                    if len(faces_data) >= 200:
                        break

                    # Extract and preprocess face
                    face_roi = gray[y:y+h, x:x+w]
                    resized = cv2.resize(face_roi, (50, 50))
                    equalized = cv2.equalizeHist(resized)
                    
                    # Data augmentation
                    flipped = cv2.flip(equalized, 1)
                    
                    faces_data.extend([equalized, flipped])
                    
                    # Visual feedback
                    cv2.putText(frame, f"Captured: {len(faces_data)}/200", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                cv2.imshow('Registration', frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            if len(faces_data) < 200:
                raise ValueError("Insufficient face samples captured")

            # Save normalized features
            faces_array = np.array(faces_data).reshape(200, -1)
            student_ids = [student_id] * 200

            # Update face data files
            if os.path.exists('data/faces.pkl'):
                with open('data/faces.pkl', 'rb') as f:
                    existing_faces = pickle.load(f)
                faces_array = np.vstack((existing_faces, faces_array))
            
            with open('data/faces.pkl', 'wb') as f:
                pickle.dump(faces_array, f)
                
            if os.path.exists('data/ids.pkl'):
                with open('data/ids.pkl', 'rb') as f:
                    existing_ids = pickle.load(f)
                student_ids = existing_ids + student_ids
            
            with open('data/ids.pkl', 'wb') as f:
                pickle.dump(student_ids, f)

            conn.commit()
            

            model_files = ['data/svm_model.pkl', 'data/scaler.pkl']
            for fpath in model_files:
                if os.path.exists(fpath):
                    os.remove(fpath)
                    print(f"Removed old model file: {fpath}")

            return True

        except Exception as e:
            conn.rollback()
            print(f"Registration failed: {e}")
            return False
            
        finally:
            video.release()
            cv2.destroyAllWindows()
            
    except Exception as e:
        conn.rollback()
        print(f"Database transaction failed: {e}")
        return False
    finally:
        conn.close()

def create_class(class_id, class_name):
    """Create new class entry"""
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO classes VALUES (?, ?)", (class_id, class_name))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        print(f"Class {class_id} already exists!")
        return False
    finally:
        conn.close()

def enroll_student(student_id, class_id):
    """Enroll student in class with validation"""
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    try:
        c.execute("SELECT student_id FROM students WHERE student_id=?", (student_id,))
        if not c.fetchone():
            print(f"Student {student_id} not registered!")
            return False

        c.execute("SELECT class_id FROM classes WHERE class_id=?", (class_id,))
        if not c.fetchone():
            print(f"Class {class_id} doesn't exist!")
            return False
            
        c.execute("SELECT enrollment_id FROM enrollments WHERE student_id=? AND class_id=?", 
                 (student_id, class_id))
        if c.fetchone():
            print(f"Student {student_id} already enrolled in {class_id}")
            return False
            
        c.execute("INSERT INTO enrollments VALUES (NULL, ?, ?)", (student_id, class_id))
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Enrollment error: {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student Management System")
    parser.add_argument('--add-student', action='store_true', help='Register new student')
    parser.add_argument('--create-class', action='store_true', help='Create new class')
    parser.add_argument('--enroll', action='store_true', help='Enroll student in class')
    
    args = parser.parse_args()
    
    if args.add_student:
        student_id = input("Enter Student ID: ").strip()
        name = input("Enter Full Name: ").strip()
        if atomic_registration(student_id, name):
            print(f"Successfully registered {name} ({student_id})")
        else:
            print("Registration failed. No data saved.")
            
    elif args.create_class:
        class_id = input("Enter Class ID: ").strip()
        class_name = input("Enter Class Name: ").strip()
        if create_class(class_id, class_name):
            print(f"Class {class_id} created successfully!")
            
    elif args.enroll:
        student_id = input("Enter Student ID: ").strip()
        class_id = input("Enter Class ID: ").strip()
        if enroll_student(student_id, class_id):
            print(f"Student {student_id} enrolled in {class_id}")
            
    else:
        print("Specify an operation: --add-student, --create-class, or --enroll")