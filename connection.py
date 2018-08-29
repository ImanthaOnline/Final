import pymysql;

def connection():
    conn=pymysql.connect(host="localhost",user="root", password="123",db="criminalinformation")
    c=conn.cursor()
    return c,conn



