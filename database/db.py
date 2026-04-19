import sqlite3

conn = sqlite3.connect("inspection.db")
cursor = conn.cursor()

def init_db():
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TEXT,
        batch_id TEXT,
        total INTEGER,
        normal INTEGER,
        chipped INTEGER,
        capped INTEGER,
        status TEXT
    )
    """)
    conn.commit()


def save_to_db(data):
    cursor.execute("""
    INSERT INTO results (time, batch_id, total, normal, chipped, capped, status)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        data["time"],
        data["batch_id"],
        data["total"],
        data["normal"],
        data["chipped"],
        data["capped"],
        data["status"]
    ))

    conn.commit()