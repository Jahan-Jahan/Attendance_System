import sys
import mediapipe as mp
import cv2 as cv
import numpy as np
import glob
import os
import time
from deepface import DeepFace
from scipy.spatial.distance import cosine
from database_read import DatabaseReader
from database_register import DatabaseWriter
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QPushButton, QLabel, QStackedWidget, QLineEdit, 
                            QMessageBox, QSpacerItem, QSizePolicy, QGraphicsDropShadowEffect)
from PyQt6.QtGui import QImage, QPixmap, QFontDatabase, QFont, QColor
from PyQt6.QtCore import QTimer, Qt, QUrl
from PyQt6.QtMultimedia import QSoundEffect
import qdarktheme
import logging

log_dir = f"{os.path.abspath(os.getcwd())}\\base\\logging"
log_file = os.path.join(log_dir, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

class AuthWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        layout = QVBoxLayout()
        
        self.username = QLineEdit()
        self.username.setPlaceholderText("Username")
        self.input_username = None
        
        self.login_btn = QPushButton("Login")
        self.register_btn = QPushButton("Register")
        self.exit_btn = QPushButton("Exit")

        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(10)
        self.shadow.setXOffset(5)
        self.shadow.setYOffset(5)
        self.shadow.setColor(QColor(0, 0, 0, 150))
        self.setGraphicsEffect(self.shadow)

        self.login_btn.setStyleSheet('''
            background-color: #c99402;
        ''')
        self.login_btn.setGraphicsEffect(self.shadow)

        self.register_btn.setStyleSheet('''
            background-color: #5b87f2;
        ''')
        self.register_btn.setGraphicsEffect(self.shadow)

        self.exit_btn.setStyleSheet('''
            background-color: #3a52e0;
        ''')
        self.exit_btn.setGraphicsEffect(self.shadow)

        self.setStyleSheet('''
            color: rgb(230, 230, 230);
        ''')
        self.setGraphicsEffect(self.shadow)

        background = QLabel()
        background_pixmap = QPixmap("./source/attendance_system.jpg")
        background.setPixmap(background_pixmap)
        background.setScaledContents(True)
        
        layout.addWidget(background)
        layout.addWidget(self.username)
        layout.addWidget(self.login_btn)
        layout.addWidget(self.register_btn)
        layout.addWidget(self.exit_btn)
        
        self.login_btn.clicked.connect(self.attempt_login)
        self.register_btn.clicked.connect(self.attempt_register)
        self.exit_btn.clicked.connect(self.attempt_exit)

        self.setLayout(layout)

    def get_input_username(self):
        self.input_username = self.username.text().strip().lower()
        return self.input_username

    def username_check(self, username):
        filepath = os.path.join("./people", f"{username}.jpg")

        if os.path.exists(filepath):
            return True
        else:
            return False
    
    def attempt_login(self):
        self.parent.play_click()
        username = self.get_input_username()

        if username:
            response = self.username_check(self.get_input_username())

            if response:
                logging.info(f"User '{self.get_input_username().capitalize()}' logged in.")
                self.parent.login = True
                self.parent.change_screen(1)
            else:
                logging.warning(f"Failed login attempt for username: '{username}'")
                QMessageBox.warning(self, "Error", "This username is not valid.")
        else:
            logging.warning("Login attempt with empty username.")
            QMessageBox.warning(self, "Error", "Please enter your username.")

    def attempt_register(self):
        self.parent.play_click()
        username = self.get_input_username()

        if username:
            response = self.username_check(username)

            if not response:
                logging.info(f"New user '{self.get_input_username().capitalize()}' registered.")
                self.parent.register = True
                self.parent.change_screen(1)
            else:
                logging.warning(f"You've been already registered using username '{username}'.")
                QMessageBox.warning(self, "Error", f"You've been already registered using username '{username}'.")
        else:
            logging.warning(f"Please enter your username.")
            QMessageBox.warning(self, "Error", "Please enter your username.")

    def attempt_exit(self):
        logging.info("Application exited by user.")
        self.parent.play_click()
        QApplication.instance().quit()

class WebcamWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        self.threshold = 0.75

        self.sources = {}
        self.all_people = glob.glob(os.path.join("./people", "*.jpg"))

        self.database_writer = DatabaseWriter()
        self.database_reader = DatabaseReader()

        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

        self.success_sound = QSoundEffect(self)
        sound_path = os.path.abspath("./source/success.wav")
        self.success_sound.setSource(QUrl.fromLocalFile(sound_path))
        self.success_sound.setVolume(0.9)
        self.success_sound.setLoopCount(1)

        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(10)
        self.shadow.setXOffset(5)
        self.shadow.setYOffset(5)
        self.shadow.setColor(QColor(0, 0, 0, 150))
        self.setGraphicsEffect(self.shadow)

        layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setGraphicsEffect(self.shadow)
        
        self.capture_btn = QPushButton("Detect")
        self.capture_btn.clicked.connect(self.capture_and_process)
        self.capture_btn.setGraphicsEffect(self.shadow)
        self.capture_btn.setStyleSheet('''
            background-color: #c99402;
        ''')
        
        self.back_btn = QPushButton("Cancel")
        self.back_btn.clicked.connect(self.cancel)
        self.back_btn.setGraphicsEffect(self.shadow)
        self.back_btn.setStyleSheet('''
            background-color: #5b87f2;
        ''')

        self.setStyleSheet('''
            color: rgb(230, 230, 230);
        ''')
        self.setGraphicsEffect(self.shadow)
        
        layout.addWidget(self.video_label)
        layout.addWidget(self.capture_btn)
        layout.addWidget(self.back_btn)
        
        self.setLayout(layout)
        
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
    
    def start_webcam(self):
        self.cap = cv.VideoCapture(0)
        self.timer.start(30)
    
    def stop_webcam(self):
        if self.cap:
            self.timer.stop()
            self.cap.release()
            self.cap = None

    def ok_successfully(self):
        self.success_sound.play()

    def preprocess_frame(self, frame):
        H, W, _ = frame.shape

        x = None
        y = None
        w = None
        h = None

        out = self.face_detection.process(frame)

        detections = out.detections
        if detections is not None:
            ret = True
            for detection in detections:
                location = detection.location_data
                bbox = location.relative_bounding_box

                x, y, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

                x = int(x * W)
                y = int(y * H)
                w = int(w * W)
                h = int(h * H)

                break
        else:
            ret = False
        
        return ret, (x, y, w, h)
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            is_face, (x, y, W, H) = self.preprocess_frame(frame)

            if is_face:
                cv.rectangle(frame, (x, y), (x + W, y + H), (0, 255, 0), 2, cv.LINE_AA)

            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def get_embedding(self, frame):
        embedding = None
        try:
            embedding = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)
            embedding = np.array(embedding[0]["embedding"])
        except Exception as e:
            logging.error(f"There is a problem in face detection: {e}")
            QMessageBox(self, "Error", "There is a problem in face detection.")
        finally:
            return embedding
        
    def get_similarity(self, emb1, emb2):
        sim = 0
        try:
            sim = 1 - cosine(emb1, emb2)
        except Exception as e:
            logging.error(f"There is a problem in getting similarity operation: {e}")
        finally:
            return sim
        
    def check_from_database(self, username, embedding):
        if len(self.sources) == 0:
            for p in self.all_people:
                username_ = os.path.basename(p).replace(".jpg", "")
                self.sources[username_] = self.database_reader.get_data(username_)
        
        status = False
        source_img = None
        max_sim = 0
        for another_username, source_embedding in self.sources.items():
            if embedding is None:
                break

            similarity = self.get_similarity(embedding, source_embedding)
            max_sim = max(max_sim, similarity)

            if similarity > self.threshold and another_username == username:
                source_img_path = os.path.join("./people", f"{another_username}.jpg")
                source_img = cv.imread(source_img_path)
                status = True
                break
        
        return status, source_img, max_sim
    
    def click_ok_btn(self):
        self.parent.play_click()
        time.sleep(0.1)
        QApplication.instance().quit()
    
    def create_final_screen(self, source_img, message="Welcome!"):
        h, w, ch = source_img.shape
        bytes_per_line = ch * w
        q_img = QImage(source_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)

        if self.layout() is not None:
            for i in reversed(range(self.layout().count())):
                widget = self.layout().itemAt(i).widget()
                if widget:
                    widget.deleteLater()

        welcome_label = QLabel(message)
        welcome_label.setStyleSheet('''
            margin: 0;
            padding: 0;
            margin-top: 25px;
            color: rgb(230, 230, 230);
            font-size: 38px;
        ''')
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        image_label = QLabel()
        image_label.setPixmap(QPixmap.fromImage(q_img))
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.click_ok_btn)
        ok_btn.setStyleSheet('''
            background-color: #c99402;
        ''')

        if self.layout() is None:
            layout = QVBoxLayout(self)
            self.setLayout(layout)
        else:
            layout = self.layout()

        layout.addWidget(image_label)
        layout.addWidget(welcome_label)

        layout.addSpacerItem(QSpacerItem(10, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        layout.addWidget(ok_btn)
        self.update()
    
    def capture_and_process(self):
        self.parent.play_click()
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.timer.stop()
                username = self.parent.get_username()
                current_embedding = self.get_embedding(frame)

                if self.parent.login:
                    status, source_img, max_sim = self.check_from_database(username, current_embedding)
                    if status:
                        message = f"Welcome {username.capitalize()}\nConfidence: %{100*max_sim:.2f}"
                        self.create_final_screen(source_img, message)
                        self.ok_successfully()
                    else:
                        if max_sim > self.threshold:
                            logging.warning(f"This face is under another username(Confidence: {100*max_sim:.2f})")
                            QMessageBox.warning(self, "Error", f"This face is under another username(Confidence: {100*max_sim:.2f})")
                        else:
                            logging.warning(f"Sorry! it sounds you are not that person(Confidence: {100*max_sim:.2f})")
                            QMessageBox.warning(self, "Error", f"Sorry! it sounds you are not that person(Confidence: {100*max_sim:.2f})")
                        self.timer.start()
                else:
                    message = "Since now, you are registered😊❤️"
                    logging.info(message)
                    self.database_writer.save_into_database(username, current_embedding)
                    self.create_final_screen(frame, message)
                    cv.imwrite(os.path.join("./people", f"{username}.jpg"), frame)
    
    def cancel(self):
        self.parent.play_click()
        self.parent.login = False
        self.parent.register = False
        self.stop_webcam()
        self.parent.change_screen(0)
    
    def showEvent(self, event):
        self.start_webcam()
        super().showEvent(event)
    
    def hideEvent(self, event):
        self.stop_webcam()
        super().hideEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Detection App")
        self.setGeometry(300, 75, 800, 600)
        
        self.click_sound = QSoundEffect(self)
        sound_path = os.path.abspath("./source/click.wav")
        self.click_sound.setSource(QUrl.fromLocalFile(sound_path))
        self.click_sound.setVolume(0.9)
        self.click_sound.setLoopCount(1)
        
        font_id = QFontDatabase.addApplicationFont("./source/Comic_Neue/ComicNeue-Regular.ttf")
        if font_id != -1:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            app_font = QFont(font_family, 22)
            QApplication.setFont(app_font)
        else:
            logging.warning("Failed to load custom font.")
        
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.login = False
        self.register = False
        
        self.auth_screen = AuthWindow(self)
        self.webcam_screen = WebcamWindow(self)

        self.stacked_widget.addWidget(self.auth_screen)
        self.stacked_widget.addWidget(self.webcam_screen)

    def get_username(self):
        return self.auth_screen.get_input_username()
    
    def change_screen(self, index):
        self.stacked_widget.setCurrentIndex(index)

    def play_click(self):
        self.click_sound.play()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    stylesheet = qdarktheme.load_stylesheet(theme="dark", corner_shape="sharp")
    app.setStyleSheet(stylesheet)

    window = MainWindow()
    window.show()

    logging.info("App is running.")
    sys.exit(app.exec())
    logging.infor("App closed.")