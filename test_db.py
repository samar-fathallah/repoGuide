import sqlite3
conn = sqlite3.connect(r"C:\Users\samos\Documents\CVs\repoGuide\src\data\indices\sparseDrive\structural.db")
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(cur.fetchall())

cur.execute("SELECT * FROM definitions LIMIT 10")
print(cur.fetchall())