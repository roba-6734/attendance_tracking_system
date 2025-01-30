import cv2
import pickle
import numpy as np
import sqlite3
from sklearn.neighbors import KNeighborsClassifier
import time
from datetime import datetime, timedelta
import pyttsx3
import os
import csv
from reset_db import init_db  # Add this import

# Database setup


init_db()

# Load face data
with open('data/ids.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/faces.pkl', 'rb') as f:
    FACES = pickle.load(f)

if len(FACES.shape) == 1:
    FACES = FACES.reshape(-1, 7500)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Database functions
def start_class_session(class_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO sessions (class_id, start_time) VALUES (?, ?)",
             (class_id, start_time))
    conn.commit()
    session_id = c.lastrowid
    conn.close()
    return session_id

def end_session(session_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("UPDATE sessions SET end_time = ? WHERE session_id = ?",
             (end_time, session_id))
    conn.commit()
    conn.close()

def save_attendance(student_id, session_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Include session_id in the INSERT statement
        c.execute('''INSERT INTO attendance 
                     (student_id, timestamp, session_id) 
                     VALUES (?, ?, ?)''',
                 (student_id, timestamp, session_id))
        conn.commit()
        return True
    except Exception as e:
        print(f"Database error: {e}")
        return False
    finally:
        conn.close()

def get_absent_students(session_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    c.execute('''SELECT s.student_id, s.name 
                FROM enrollments e
                JOIN students s ON e.student_id = s.student_id
                WHERE e.class_id = (
                    SELECT class_id FROM sessions WHERE session_id = ?
                )
                AND s.student_id NOT IN (
                    SELECT student_id FROM attendance 
                    WHERE session_id = ?
                )''', (session_id, session_id))
    
    absentees = c.fetchall()
    conn.close()
    return absentees

def generate_absence_report(session_id):
    absentees = get_absent_students(session_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/absence_report_{session_id}_{timestamp}.csv"
    
    os.makedirs("reports", exist_ok=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Student ID", "Name", "Session ID"])
        writer.writerows([(*row, session_id) for row in absentees])
    
    return filename

def speaker(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

#main program
video = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Start class session
class_id = input("Enter Class ID to monitor: ").strip()
session_id = start_class_session(class_id)
session_start = datetime.now()

# Track announced students
saved_students = set()

try:
    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cropped_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(cropped_img, (50, 50)).reshape(1, -1)
            student_id = knn.predict(resized_img)[0]

            # Display info
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, f"ID: {student_id}", (x, y-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Save and announce only once
            if student_id not in saved_students:
                if save_attendance(student_id, session_id):
                    speaker(f"Attendance saved for {student_id}")
                    saved_students.add(student_id)
                    cv2.putText(frame, "SAVED", (x+10, y+h+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

        cv2.imshow('Attendance System', frame)

        # Auto-end after 60 minutes
        if (datetime.now() - session_start) > timedelta(minutes=60):
            break

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    video.release()
    cv2.destroyAllWindows()
    end_session(session_id)
    
    # Generate report
    report_file = generate_absence_report(session_id)
    print(f"\nAbsence report generated: {report_file}")
    
    # Display absentees
    absentees = get_absent_students(session_id)
    print("\nAbsent Students:")
    for sid, name in absentees:
        print(f"{sid} - {name}")