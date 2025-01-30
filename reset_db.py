# reset_db.py
import sqlite3

def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    # Drop existing tables
    c.execute("DROP TABLE IF EXISTS attendance")
    c.execute("DROP TABLE IF EXISTS sessions")
    c.execute("DROP TABLE IF EXISTS enrollments")
    c.execute("DROP TABLE IF EXISTS classes")
    c.execute("DROP TABLE IF EXISTS students")
    
    # Recreate tables with updated schema
    c.execute('''CREATE TABLE students
                (student_id TEXT PRIMARY KEY,
                 name TEXT NOT NULL)''')
    
    c.execute('''CREATE TABLE classes
                (class_id TEXT PRIMARY KEY,
                 class_name TEXT NOT NULL)''')
    
    c.execute('''CREATE TABLE enrollments
                (enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 student_id TEXT,
                 class_id TEXT,
                 FOREIGN KEY(student_id) REFERENCES students(student_id),
                 FOREIGN KEY(class_id) REFERENCES classes(class_id))''')
    
    c.execute('''CREATE TABLE sessions
                (session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 class_id TEXT,
                 start_time DATETIME,
                 end_time DATETIME,
                 FOREIGN KEY(class_id) REFERENCES classes(class_id))''')
    
    c.execute('''CREATE TABLE attendance
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 student_id TEXT NOT NULL,
                 timestamp DATETIME NOT NULL,
                 session_id INTEGER NOT NULL,
                 FOREIGN KEY(student_id) REFERENCES students(student_id),
                 FOREIGN KEY(session_id) REFERENCES sessions(session_id))''')
    
    conn.commit()
    conn.close()
    print("Database reset successfully!")

if __name__ == "__main__":
    init_db()