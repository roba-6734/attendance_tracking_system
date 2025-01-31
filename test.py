import cv2
import pickle
import numpy as np
import sqlite3
import os
import time
from datetime import datetime, timedelta
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score
import pyttsx3
import csv
from reset_db import init_db  # Import from external file

# ------------------ Configuration ------------------
CASCADE_PATHS = {
    'face': [
        'data/haarcascade_frontalface_alt2.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
    ],
    'eye': [
        'data/haarcascade_eye.xml',
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    ],
    'eye_glasses': [
        'data/haarcascade_eye_tree_eyeglasses.xml',
        cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
    ]
}

# Initialize database safely
init_db(safe=True)

# ------------------ Cascade Loading ------------------
# ------------------ Cascade Loading ------------------
class CascadeLoader:
    def __init__(self):
        self.classifiers = {}
        self.load_cascades()
    
    def load_cascades(self):
        """Load all required Haar cascades with validation"""
        for cascade_type, paths in CASCADE_PATHS.items():
            self.classifiers[cascade_type] = None
            for path in paths:
                try:
                    if os.path.exists(path):
                        print(f"Attempting to load {cascade_type} from: {path}")
                        classifier = cv2.CascadeClassifier(path)
                        if not classifier.empty():
                            self.classifiers[cascade_type] = classifier
                            print(f"Successfully loaded {cascade_type}")
                            break
                except Exception as e:
                    print(f"Error loading {path}: {str(e)}")
            
            if self.classifiers[cascade_type] is None:
                available_files = [p for p in paths if os.path.exists(p)]
                raise RuntimeError(f"Failed to load {cascade_type} cascade.\n"
                                 f"Tried paths: {paths}\n"
                                 f"Existing files: {available_files}")

def verify_xml(file_path):
    try:
        with open(file_path, 'r') as f:
            header = f.readline().strip()
            if header != '<?xml version="1.0"?>':
                print(f"Invalid XML header in {file_path}")
                return False
            return True
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return False

# Add before CascadeLoader initialization
for cascade_type in CASCADE_PATHS:
    for path in CASCADE_PATHS[cascade_type]:
        if os.path.exists(path):
            if not verify_xml(path):
                print(f"Invalid XML file: {path}")
                print("Redownload from OpenCV GitHub repository")
                exit(1)


try:
    cascade_loader = CascadeLoader()
except RuntimeError as e:
    print("\n" + "="*60)
    print("CRITICAL ERROR: Failed to load cascade classifiers")
    print("="*60)
    print(e)
    print("\nRequired files:")
    print("- haarcascade_frontalface_alt2.xml")
    print("- haarcascade_eye.xml")
    print("- haarcascade_eye_tree_eyeglasses.xml")
    print("\nDownload them from:")
    print("https://github.com/opencv/opencv/tree/4.x/data/haarcascades")
    print("and place in the 'data' folder")
    exit(1)
# ------------------ Temporal Filter ------------------
class TemporalFilter:
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.history = {}
        
    def update(self, face_id, prediction):
        if face_id not in self.history:
            self.history[face_id] = []
        self.history[face_id].append((prediction, time.time()))
        if len(self.history[face_id]) > self.buffer_size:
            self.history[face_id].pop(0)
            
    def get_consensus(self, face_id):
        records = self.history.get(face_id, [])
        if not records:
            return None, 0.0
            
        weights = np.linspace(0.5, 1.0, len(records))
        scores = {}
        for (label, conf), weight in zip([r[0] for r in records], weights):
            scores[label] = scores.get(label, 0.0) + conf * weight
            
        best_label = max(scores, key=scores.get) if scores else None
        total_weight = sum(weights)
        confidence = scores[best_label] / total_weight if best_label else 0.0
        return best_label, confidence

# ------------------ Face Processing ------------------
def non_max_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = x1 + boxes[:,2]
    y2 = y1 + boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    
    return boxes[pick]

def align_face(face_roi):
    try:
        eyes = cascade_loader.classifiers['eye'].detectMultiScale(face_roi, 1.1, 5)
        if len(eyes) < 2:
            eyes = cascade_loader.classifiers['eye_glasses'].detectMultiScale(face_roi, 1.1, 5)
        
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda x: x[0])[:2]
            (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes
            dx, dy = ex2 - ex1, ey2 - ey1
            angle = np.degrees(np.arctan2(dy, dx))
            
            center = (face_roi.shape[1]//2, face_roi.shape[0]//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(face_roi, M, face_roi.shape[::-1], 
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        print(f"Alignment warning: {str(e)}")
    
    return face_roi

# ------------------ Model Handling ------------------
def load_model():
    try:
        with open('data/faces.pkl', 'rb') as f:
            faces = pickle.load(f)
        with open('data/ids.pkl', 'rb') as f:
            labels = pickle.load(f)
            
        scaler = StandardScaler()
        faces_scaled = scaler.fit_transform(faces)
        
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        model = SVC(kernel='rbf', C=5, gamma=0.005,
                  class_weight=dict(zip(np.unique(labels), class_weights)),
                  probability=True)
        
        scores = cross_val_score(model, faces_scaled, labels, cv=3)
        print(f"Model CV accuracy: {np.mean(scores):.2f} (±{np.std(scores):.2f})")
        
        model.fit(faces_scaled, labels)
        return model, scaler
        
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        exit(1)

model, scaler = load_model()
temporal_filter = TemporalFilter()

# ------------------ Core Functions ------------------
def start_class_session(class_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO sessions (class_id, start_time) VALUES (?, ?)",
                (class_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        return c.lastrowid
    except sqlite3.Error as e:
        print(f"Session error: {e}")
        return None
    finally:
        conn.close()

def save_attendance(student_id, session_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    try:
        c.execute('''INSERT INTO attendance 
                    (student_id, timestamp, session_id)
                    VALUES (?, ?, ?)''',
                (student_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session_id))
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Attendance error: {e}")
        return False
    finally:
        conn.close()

def recognize_face(frame, confidence_threshold=0.8):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    try:
        faces = cascade_loader.classifiers['face'].detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=8,
            minSize=(200, 200),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    except Exception as e:
        print(f"Face detection error: {str(e)}")
        return []
    
    faces = non_max_suppression(np.array(faces))
    predictions = []
    
    for (x, y, w, h) in faces:
        try:
            face_roi = gray[y:y+h, x:x+w]
            aligned_face = align_face(face_roi)
            processed = cv2.resize(aligned_face, (50, 50))
            processed = cv2.equalizeHist(processed)
            processed = cv2.GaussianBlur(processed, (3, 3), 0)
            
            features = scaler.transform([processed.flatten()])
            probas = model.predict_proba(features)[0]
            pred_label = model.classes_[np.argmax(probas)]
            confidence = np.max(probas)
            
            face_id = f"{x}_{y}_{w}_{h}"
            temporal_filter.update(face_id, (pred_label, confidence))
            
            final_label, final_conf = temporal_filter.get_consensus(face_id)
            if final_conf >= confidence_threshold:
                predictions.append((x, y, w, h, final_label, final_conf))
                
        except Exception as e:
            print(f"Face processing error: {str(e)}")
            
    return predictions

def generate_report(session_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    try:
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
        
        os.makedirs('reports', exist_ok=True)
        filename = f'reports/absence_{session_id}_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Student ID', 'Name', 'Session ID'])
            writer.writerows([(*row, session_id) for row in absentees])
            
        return filename
    finally:
        conn.close()

def speaker(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS error: {e}")

def main():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not access camera")
        return

    class_id = input("Enter Class ID to monitor: ").strip()
    session_id = start_class_session(class_id)
    if not session_id:
        return

    saved_students = set()
    session_start = datetime.now()

    try:
        while (datetime.now() - session_start) < timedelta(minutes=60):
            ret, frame = video.read()
            if not ret:
                break

            predictions = recognize_face(frame)
            
            for (x, y, w, h, label, confidence) in predictions:
                color = (0, 255, 0) if confidence >= 0.8 else (0, 0, 255)
                text = f"{label} ({confidence:.2f})" if confidence >= 0.8 else "Unknown"
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if confidence >= 0.8 and label not in saved_students:
                    if save_attendance(label, session_id):
                        speaker(f"Attendance saved for {label}")
                        saved_students.add(label)

            cv2.imshow('Attendance System', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        video.release()
        cv2.destroyAllWindows()
        
        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()
        c.execute("UPDATE sessions SET end_time = ? WHERE session_id = ?",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session_id))
        conn.commit()
        conn.close()
        
        report_file = generate_report(session_id)
        print(f"\nAbsence report generated: {report_file}")

if __name__ == "__main__":
    main()