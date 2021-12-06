# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 15:13:00 2020

"""

# SQLite3 can be integrated with Python using sqlite3 module, which was written
# by Gerhard Haring. It provides an SQL interface compliant with the DB-API 2.0
# specification 


import sqlite3


# To use sqlite3 module, you must first create a connection object that represents 
# the database and then optionally you can create a cursor object, which will 
# help you in executing all the SQL statements.
con = sqlite3.connect('./data/sql-test.db')

# create table
with con:
    con.execute("""
        CREATE TABLE USER (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER
        );
    """)
        
# insetr statement    
sql = 'INSERT INTO USER (id, name, age) values(?, ?, ?)'
data = [
    (1, 'Alice', 21),
    (2, 'Bob', 22),
    (3, 'Chris', 23),
]        
with con:
    con.executemany(sql, data)

# select statement    
with con:
    data = con.execute("SELECT * FROM USER WHERE age <= 22")
    for row in data:
        print(row)    
        
# close connectin
con.close()        