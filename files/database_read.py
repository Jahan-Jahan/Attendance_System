import os
import mysql.connector
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

def get_data(name):
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT embedding FROM users WHERE name = %s;", (name, ))
            res = cursor.fetchone()
            embedding = np.frombuffer(res[0], dtype=np.float64)
            return embedding

        except Exception as execution_error:
            print(f"There is a problem in query execution:\n{execution_error}")

    except mysql.connector.Error as connection_error:
        print(f"There is a problem in connection:\n{connection_error}")

    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()
            cursor.close()
            print("Database connection closed.")
