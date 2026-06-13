import sqlite3
import os

# ဒေတာဘေ့စ် ဖိုင်အမည် သတ်မှတ်ခြင်း
DB_PATH = "students.db"

def init_db():
    print("⏳ Starting database initialization...")
    
    # ဒေတာဘေ့စ်ဖိုင်သို့ ချိတ်ဆက်ခြင်း (မရှိသေးပါက အလိုအလျောက် ဖန်တီးပေးမည်)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 👥 Students Table ကို စနစ်တကျ တည်ဆောက်ခြင်း
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admission_id TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            roll_no TEXT NOT NULL,
            department TEXT NOT NULL,
            semester TEXT NOT NULL,
            image_path TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database and 'students' table initialized successfully!")

if __name__ == "__main__":
    init_db()