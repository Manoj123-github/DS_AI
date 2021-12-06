# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 15:13:00 2020


"""

# import os.path
from os import path
import sqlite3
import pandas as pd

# Using a connection object we can create a cursor object which allows us to 
# execute SQLite command/queries through Python.
# We can create as many cursors as we want from a single connection object. 
# Like connection object, this cursor object is also not thread-safe. the sqlite3 
# module doesn’t allow sharing cursors between threads. If you still try to do so, 
# you will get an exception at runtime.

# the advantage of cursor is that it retuns extra information at insert level

DBName = './data/sql-employee.db'

# craete new database
def create_conn(pDBName):
    if path.exists(pDBName):
        print("Database File Exists ...")
        pConn = None
        return (pConn)    
    try:
        pConn = sqlite3.connect(pDBName)
        print("Database Creation Successful ...")
    except:
        print("Database Creation Failed ...")
        pConn = None
    return (pConn)    


# open existing database
def open_conn(pDBName):
    if not path.exists(pDBName):
        print("Database File Not Found ...")
        pConn = None
        return (pConn)    
    try:
        pConn = sqlite3.connect(pDBName)
        print("Connection Successful ...")
    except:
        print("Connection Failed ...")
        pConn = None
    return (pConn)    


# close database
def close_conn(pConn):
    pConn.close()
    print("Connection Closed ...")
    return


# show tables
def show_tables(pConn):
	conn = open_conn(DBName)
	# query
	try:
		curs = conn.cursor()
		rows = curs.execute(""" select name from sqlite_master where type = 'table'; """)
		print("Query Successful ...")
		for row in rows:
			print(row)    
	except:  
		print("Query Failed ...")
	#commit the changes to db			
	conn.commit()
	#close the connnection
	close_conn(conn)
	#
	return


#=======================================================
# create db
#=======================================================
conn = create_conn(DBName)
if conn != None:
    close_conn(conn)


#=======================================================
# create table
#=======================================================
conn = open_conn(DBName)
# query
try:
    curs = conn.cursor()
    curs.execute("""
        CREATE TABLE employee (
            id INTEGER,
            name TEXT,
            donation REAL,
            location TEXT,
            status INTEGER 
        );
    """)
    print("Table Creation Successful ...")
except:  
    print("Table Creation Failed ...")
# commit the changes to db			
if conn != None:
    conn.commit()
# close the connnection
if conn != None:
    close_conn(conn)


#=======================================================
# show tables
#=======================================================
show_tables(DBName)


#=======================================================
# insert single record
#=======================================================
conn = open_conn(DBName)
try:
    sql = """INSERT INTO employee (id, name, donation, location, status) 
                values (101,'Vipul Pats', 6800.00, 'India', 1)"""
    curs = conn.cursor()
    curs.execute(sql)
    conn.commit()        
    print("Insert Successful ...")
    print("Rows Inserted: "+str(curs.rowcount))
except:
    print("Insert Failed ...")
# close the connnection
if conn != None:
    close_conn(conn)


#=======================================================
# insert multiple records
#=======================================================
# prepare
conn = open_conn(DBName)
try:
    sql = """INSERT INTO employee (id, name, donation, location, status) values (?, ?, ?, ?, ?);"""
    data = [(102, 'Adil Null', 5500.00, 'India', 1),
            (103, 'Abbas connti', 6500.00, 'India', 1),
            (104, 'Abzi Book', 7000.00, 'India', 0)]        
    # insert group records
    curs = conn.cursor()
    curs.executemany(sql, data)
    conn.commit()
    #print('We have inserted', curs.rowcount, 'records to the table.')
    print("Group Insert Successful ...")
    print("Rows Inserted: "+str(curs.rowcount))
except:
    print("Group Insert Failed ...")
# close the connnection
if conn != None:
    close_conn(conn)


#=======================================================
# select statement    
#=======================================================
conn = open_conn(DBName)
try:
    curs = conn.cursor()
    rows = curs.execute("""SELECT * FROM employee""")
    print("Select Successful ...")
    for row in rows:
        print(row)    
except:
    print("Select Failed ...")
# close the connnection
if conn != None:
    close_conn(conn)


#=======================================================
# check if table exists
# not possible with con objet only
# requries cursor object
#=======================================================
conn = open_conn(DBName)
try:
    curs = conn.cursor()
    curs.execute(""" SELECT count(name) FROM sqlite_master WHERE type='table' AND name='employee' """)
    # if the count is 1, then table exists
    if (curs.fetchone()[0] == 1):
        print('Table Exists ...')
except:
    print('Table Not Exists ...')
# close the connnection
if conn != None:
    close_conn(conn)


#=======================================================
# update records
#=======================================================
conn = open_conn(DBName)
try:
    sql = """ UPDATE employee SET donation = 9000 where id = 101 """
    curs = conn.cursor()
    curs.execute(sql)
    conn.commit()        
    print("Update Successful ...")
    print("Rows Updated: "+str(curs.rowcount))
except:
    print("Update Failed ...")
# close the connnection
if conn != None:
    close_conn(conn)


#=======================================================
# delete records
#=======================================================
conn = open_conn(DBName)
try:
    sql = """ DELETE from employee where id = 101 """
    curs = conn.cursor()
    curs.execute(sql)
    conn.commit()        
    print("Delete Successful ...")
    print("Rows Deleted: "+str(curs.rowcount))
except:
    print("Delete Failed ...")
# close the connnection
if conn != None:
    close_conn(conn)


#=======================================================
# drop tables
#=======================================================
# create table
conn = open_conn(DBName)
try:
    curs = conn.cursor()
    curs.execute("""
        CREATE TABLE salary (
            id INTEGER,
            salary REAL,
            joindate TEXT
        );
    """)
    print("Table Creation Successful ...")
except:  
    print("Table Creation Failed ...")
# commit the changes to db			
if conn != None:
    conn.commit()
# close the connnection
if conn != None:
    close_conn(conn)
#
# show tables
show_tables(DBName)
#
# drop table
conn = open_conn(DBName)
try:
    sql = """ DROP table IF EXISTS salary; """
    curs = conn.cursor()
    curs.execute(sql)
    conn.commit()        
    print("Drop Table Successful ...")
    print("Rows Deleted: "+str(curs.rowcount))
except:
    print("Drop Table Failed ...")
# commit the changes to db			
if conn != None:
    conn.commit()
# close the connnection
if conn != None:
    close_conn(conn)
#
# show tables
show_tables(DBName)


#=======================================================
# select condition
#=======================================================
conn = open_conn(DBName)
try:
    curs = conn.cursor()
    rows = curs.execute("""SELECT * FROM employee WHERE donation > 6000 """)
    print("Select Successful ...")
    for row in rows:
        print(row)    
except:
    print("Select Failed ...")
# close the connnection
if conn != None:
    close_conn(conn)


#=======================================================
# select order by
#=======================================================
conn = open_conn(DBName)
try:
    curs = conn.cursor()
    rows = curs.execute("""SELECT * FROM employee ORDER BY id """)
    print("Select Successful ...")
    for row in rows:
        print(row)    
except:
    print("Select Failed ...")
# close the connnection
if conn != None:
    close_conn(conn)


#=======================================================
# select limit
#=======================================================
conn = open_conn(DBName)
try:
    curs = conn.cursor()
    rows = curs.execute("""SELECT * FROM employee ORDER BY id LIMIT 2 """)
    print("Select Successful ...")
    for row in rows:
        print(row)    
except:
    print("Select Failed ...")
# close the connnection
if conn != None:
    close_conn(conn)


#=======================================================
# select condition and
#=======================================================
conn = open_conn(DBName)
try:
    curs = conn.cursor()
    rows = curs.execute("""SELECT * FROM employee WHERE donation >= 6000 AND donation < 7000 """)
    print("Select Successful ...")
    for row in rows:
        print(row)    
except:
    print("Select Failed ...")
# close the connnection
if conn != None:
    close_conn(conn)


#=======================================================
# select condition or
#=======================================================
conn = open_conn(DBName)
try:
    curs = conn.cursor()
    rows = curs.execute("""SELECT * FROM employee WHERE donation < 6000 or donation >= 7000 """)
    print("Select Successful ...")
    for row in rows:
        print(row)    
except:
    print("Select Failed ...")
# close the connnection
if conn != None:
    close_conn(conn)


#=======================================================
# select condition like
#=======================================================
conn = open_conn(DBName)
try:
    curs = conn.cursor()
    rows = curs.execute("""SELECT * FROM employee WHERE name like "Ab%" """)
    print("Select Successful ...")
    for row in rows:
        print(row)    
except:
    print("Select Failed ...")
# close the connnection
if conn != None:
    close_conn(conn)



#=======================================================
# select condition like
#=======================================================
conn = open_conn(DBName)
try:
    curs = conn.cursor()
    rows = curs.execute("""SELECT * FROM employee WHERE name like "%pul%" """)
    print("Select Successful ...")
    for row in rows:
        print(row)    
except:
    print("Select Failed ...")
# close the connnection
if conn != None:
    close_conn(conn)



#=======================================================
# sql to df
#=======================================================
conn = open_conn(DBName)
try:
    df = pd.read_sql_query("SELECT * from employee", conn)
    print("SQL To DF Successful ...")
except:
    print("SQL To DF Failed ...")
# close the connnection
if conn != None:
    close_conn(conn)
print(df.head())
print(df.info())


#=======================================================
# df to sql
#=======================================================
df = pd.read_csv('./data/employee.csv')
print(df)
conn = open_conn(DBName)
try:
    # Write the new DataFrame to a new SQLite table
    df.to_sql("employee", conn, if_exists="append", index=False)
    #df.to_sql("employee", conn, if_exists="replace", index=False)
    #df.to_sql("employee", conn, if_exists="fail", index=False)
    conn.commit()        
    print("DF To SQL Successful ...")
except:
    print("DF To SQL Failed ...")
# commit the changes to db			
if conn != None:
    conn.commit()
# close the connnection
if conn != None:
    close_conn(conn)
    