# reset_db.py
import sqlite3

def init_db(safe=True):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    if not safe:
        # Destructive reset (for development)
        c.execute("DROP TABLE IF EXISTS attendance")
        c.execute("DROP TABLE IF EXISTS sessions")
        c.execute("DROP TABLE IF EXISTS enrollments")
        c.execute("DROP TABLE IF EXISTS classes")
        c.execute("DROP TABLE IF EXISTS students")
        print("⚠️ Database reset - all data deleted!")

    # Create tables if not exists (safe)
    c.execute('''CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS classes (
                class_id TEXT PRIMARY KEY,
                class_name TEXT NOT NULL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS enrollments (
                enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                class_id TEXT,
                FOREIGN KEY(student_id) REFERENCES students(student_id) ON DELETE CASCADE,
                FOREIGN KEY(class_id) REFERENCES classes(class_id) ON DELETE CASCADE)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_id TEXT,
                start_time DATETIME,
                end_time DATETIME,
                FOREIGN KEY(class_id) REFERENCES classes(class_id) ON DELETE CASCADE)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                session_id INTEGER NOT NULL,
                FOREIGN KEY(student_id) REFERENCES students(student_id) ON DELETE CASCADE,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE)''')
    
    conn.commit()
    conn.close()
    print("Database initialized safely." if safe else "Database reset complete.")

if __name__ == "__main__":
    init_db(safe=False)  # Full reset when run directly