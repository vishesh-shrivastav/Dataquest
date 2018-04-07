import pandas as pd
import sqlite3 

conn = sqlite3.connect('factbook.db')
cur = conn.cursor()

query_area_land = 'select sum(area_land) from facts where area_land != "";'
area_land = cur.execute(query_area_land).fetchall()[0][0]

query_area_water = 'select sum(area_water) from facts where area_water != "";'
area_water = cur.execute(query_area_water).fetchall()[0][0]

print(float(area_land)/float(area_water))
#27.7516
