import sqlite3

conn = sqlite3.connect('factbook.db')
cur = conn.cursor()

query = "select name from facts order by population desc limit 10;"

print(cur.execute(query).fetchall())