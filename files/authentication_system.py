import mediapipe as mp
import cv2 as cv
import numpy as np
import glob
import os
from deepface import DeepFace
from scipy.spatial.distance import cosine
from database_read import DatabaseReader
from database_register import DatabaseWriter

class AuthenticationSystem:
    def __init__(self):
        self.threshold = 0.75

        self.sources = {}
        self.people_path = os.path.join("people", "*.jpg")
        self.all_people = glob.glob(self.people_path)

        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

        self.database_writer = DatabaseWriter()
        self.database_reader = DatabaseReader()

    def preprocess_frame(self, frame):
        H, W, _ = frame.shape

        x = None
        y = None
        w = None
        h = None

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        out = self.face_detection.process(frame_rgb)

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

    def capture(self, screen_name="Register"):
        cap = cv.VideoCapture(0)

        img = None
        x = None
        y = None
        w = None
        h = None

        while True:
            ret, frame = cap.read()

            if not ret:
                break
            
            is_face, (x, y, w, h) = self.preprocess_frame(frame)
            frame_copy = frame.copy()
            
            cv.putText(frame_copy, "Click \"ENTER\" to take a picture.", (25, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 25, 255), 2)

            if is_face:
                cv.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2, cv.LINE_AA)

            cv.imshow(screen_name, frame_copy)

            if cv.waitKey(1) & 0xFF == 13:
                if is_face:
                    img = frame
                    break
                else:
                    print("No face detected!")
        
        cap.release()
        cv.destroyAllWindows()

        return img

    def get_embedding(self, img):
        try:
            embedding = DeepFace.represent(img, model_name="Facenet", enforce_detection=False)
            embedding = np.array(embedding[0]["embedding"])
        except:
            print("There is a problem in face detection.")
            return None
        
        return embedding

    def get_similarity(self, emb1, emb2):
        try:
            return 1 - cosine(emb1, emb2)
        except Exception as e:
            print(f"There is a problem about the dimension of vector: {e}")
            return 0

    def login_check(self, name):
        filepath = os.path.join("people", f"{name}.jpg")

        if os.path.exists(filepath):
            return True
        else:
            return False

    def login(self, name):
        response = self.login_check(name)

        if response:
            img = self.capture("Login")
            current_embedding = self.get_embedding(img)
            if current_embedding is None:
                return False, 0

            source_embedding = self.database_reader.get_data(name)
            similarity = self.get_similarity(current_embedding, source_embedding)
            
            if similarity > self.threshold:
                return True, similarity
            
            return False, similarity
        else:
            print("You can not login.")
            return False, 0

    def register_check(self, name, img=None):
        filepath = os.path.join("people", f"{name}.jpg")
        output = None
        embedding = None

        if os.path.exists(filepath):
            embedding = self.database_reader.get_data(name)
        else:
            if img is not None:
                embedding = self.get_embedding(img)
            output = (True, 1)  # Not registered

        if embedding is None:
            output = (True, 3)  # Failure

        sources = {}
        people_path = os.path.join("people", "*.jpg")
        all_people = glob.glob(people_path)

        for p in all_people:
            name_ = os.path.basename(p).replace(".jpg", "")
            sources[name_] = self.database_reader.get_data(name_)

        for another_name, source_embedding in sources.items():
            if embedding is None:
                break

            similarity = self.get_similarity(embedding, source_embedding)
            print(another_name, similarity)

            if similarity > self.threshold and name != another_name:
                output = (True, 2)  # Face registered under another name
                break

        return output if output is not None else (False, 0)

    def register(self):
        img = self.capture()
        name = input("Enter your name: ").strip().lower()
        response, code = self.register_check(name, img=img)

        if response:
            if code == 1:
                img_filepath = os.path.join("people", f"{name}.jpg")

                embedding = self.get_embedding(img)
                self.database_writer.save_into_database(name, embedding)

                cv.imwrite(img_filepath, img)

                # embedding_filepath = os.path.join("source", f"{name}.npy")
                # np.save(embedding_filepath, embedding)

                print("You registered now!")
            elif code == 2:
                print("This face registered with another name.")
            else:
                print("Failure.")
        else:
            print("Error")

    def start(self):
        task = input("Login(L/l) or Register(R/r)?").strip().lower()

        if task in ["login", "l"]:
            name = input("Enter your name: ").strip().lower()

            response, similarity = self.login(name)
            print(f"Confidence: %{100 * similarity:.2f}")

            if response:
                print("Login successful!")
            else:
                print("You are'n that person or you've been never register before, please do it.")
                ans = input("Register(R/r) or Cancel(C/c)? ").strip().lower()

                if ans in ["register", "r"]:
                    self.register()
                else:
                    print("Bye Bye!")
        else:
            self.register()

    def run(self):
        print("Authentication system starts.")
        self.start()
