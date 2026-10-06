import sqlite3


connection = sqlite3.connect(
    "data/operational/robot_data.db"
)

cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS robots (
    robot_id TEXT PRIMARY KEY,
    battery_level INTEGER,
    status TEXT,
    error_code TEXT
)
""")

robots = [
    ("A300-01", 85, "operational", None),
    ("A300-02", 42, "charging", None),
    ("A300-03", 67, "error", "E102"),
]

cursor.executemany("""
INSERT OR REPLACE INTO robots (
    robot_id,
    battery_level,
    status,
    error_code
)
VALUES (?, ?, ?, ?)
""", robots)

connection.commit()
connection.close()