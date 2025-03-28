import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

class DatabaseWriter:
    def __init__(self, name=None):
        self.name = name

    def set_name(self, name):
        self.name = name

    def get_path(self):
        return os.path.join("people", f"{self.name}.jpg")

    def save_into_database(self, name, embedding):
        try:
            self.set_name(name)
            conn = mysql.connector.connect(
                host=self.DB_HOST,
                user=self.DB_USER,
                password=self.DB_PASSWORD,
                database=self.DB_NAME
            )
            cursor = conn.cursor()

            try:
                embedding_bytes = embedding.tobytes()
                img_path = self.get_path()

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