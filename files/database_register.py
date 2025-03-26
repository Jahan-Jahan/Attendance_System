import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

def get_path(name):
    return os.path.join("people", f"{name}.jpg")

def save_into_database(name, embedding):
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = conn.cursor()

        try:
            embedding_bytes = embedding.tobytes()
            img_path = get_path(name)

            cursor.execute("INSERT INTO users (name, image_path, embedding) VALUES (%s, %s, %s);", 
                        (name, img_path, embedding_bytes))
            conn.commit()

            print(f"Successfully inserted {name} into database.")

        except FileNotFoundError:
            print(f"Error: Image not found for {name}")
        except Exception as e:
            print(f"Error processing {name}: {e}")

    except mysql.connector.Error as e:
        print(f"Database Error: {e}")

    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
            print("Database connection closed.")