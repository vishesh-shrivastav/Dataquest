import pandas as pd
import sqlite3
import math

conn = sqlite3.connect('factbook.db')
cur = conn.cursor()

facts = pd.read_sql_query("select * from facts",conn)
facts = facts.dropna(axis=0)
facts = facts[facts['area_land'] > 0]

def project_pop(initial_pop,growth_rate):
    final_pop = initial_pop * ((math.e)**((growth_rate/100) * 35))
    return final_pop
                               
facts['pop2050'] = project_pop(facts['population'],
                         facts['population_growth'])

facts = facts.sort(['pop2050'],ascending=False)
print(facts[['name','pop2050']][:10])

                                                
                               
                               
                              

                              
                               
                               
                               