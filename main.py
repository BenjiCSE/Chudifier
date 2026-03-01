import cv2
import base64
from flask import Flask, request, jsonify, render_template, Response
import os
import numpy as np
from PIL import Image
import pillow_heif

# 1. SETUP PATHS IMMEDIATELY
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINER_FILE = os.path.join(BASE_DIR, "trainer.yml")
CASCADE_PATH = os.path.join(BASE_DIR, "features", "haarcascade_frontalface_default.xml")

app = Flask(__name__, static_folder="templates/assets", static_url_path="/assets")
IS_RENDER = os.environ.get("RENDER", "False").lower() == "true"

# 2. INITIALIZE GLOBALS AT THE TOP LEVEL
# This ensures every Flask worker has access to them
face_detector = cv2.CascadeClassifier(CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 3. LOAD DATA IMMEDIATELY
if os.path.exists(TRAINER_FILE):
    print("SUCCESS: Loading trainer.yml")
    recognizer.read(TRAINER_FILE)
else:
    print("WARNING: trainer.yml not found. Prediction will fail.")


# camera
def get_camera():
    if IS_RENDER:
        return None
    cap = cv2.VideoCapture(0)
    return cap if cap.isOpened() else None


# Join the path to the features folder
cascade_path = os.path.join(BASE_DIR, "features", "haarcascade_frontalface_default.xml")

# Load the classifier
face_detector = cv2.CascadeClassifier(cascade_path)

# Verification check (very helpful for debugging logs!)
if face_detector.empty():
    print(f"ERROR: Could not load cascade from {cascade_path}")
else:
    print("SUCCESS: Cascade loaded correctly.")


def train_from_heic(folder):
    faces = []
    ids = []
    if not os.path.exists(folder):
        print(f"Error: Folder {folder} not found.")
        return False

    for filename in os.listdir(folder):
        img_numpy = None  # Reset this for every new file
        path = os.path.join(folder, filename)
        if filename.lower().endswith(".heic"):
            try:
                heif_file = pillow_heif.read_heif(path)
                image = Image.frombytes(
                    heif_file.mode, heif_file.size, heif_file.data
                ).convert("L")
                img_numpy = np.array(image, "uint8")

                detected = face_detector.detectMultiScale(img_numpy)
                for x, y, w, h in detected:
                    faces.append(img_numpy[y : y + h, x : x + w])
                    ids.append(1)  # ID 1 represents 'Your Data'
            except Exception as e:
                print(f"Skipping {filename}: {e}")
        elif filename.lower().endswith((".jpg", ".jpeg", ".png")):
            # Standard OpenCV read for regular images
            img_color = cv2.imread(path)
            if img_color is not None:
                img_numpy = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        # --- Process the detected image data ---
        if img_numpy is not None:
            detected = face_detector.detectMultiScale(img_numpy, 1.1, 5)
            for x, y, w, h in detected:
                faces.append(img_numpy[y : y + h, x : x + w])
                ids.append(1)
                print(f"Learned face from: {filename}")

    if len(faces) > 0:
        recognizer.train(faces, np.array(ids))
        print(f"Successfully learned from {len(faces)} face samples.")
        return True
    return False


def get_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Check if we already have a saved brain
    if os.path.exists(TRAINER_FILE):
        print("Loading existing face data...")
        recognizer.read(TRAINER_FILE)
        return recognizer, True

    # If no file, perform the training from your HEIC/JPG folder
    print("No saved data found. Training from scratch...")
    success = train_from_heic("./positives")

    if success:
        recognizer.write(TRAINER_FILE)  # This creates the .yml file
        print(f"Training complete. Data saved to {TRAINER_FILE}")

    return recognizer, success


WHITE = (215, 226, 221)
RED = (247, 116, 149)
SIZE = 5


# detect
def detect_and_score(frame, gray):
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    overall_match_percent = 0
    num_faces = len(faces)

    if num_faces == 0:
        return frame, 0

    for x, y, w, h in faces:
        roi_gray = gray[y : y + h, x : x + w]

        # SAFETY CHECK: Ensure the face isn't empty or too small
        if roi_gray.size == 0:
            continue

        try:
            # If the recognizer isn't trained, predict() might throw an error
            label_id, confidence = recognizer.predict(roi_gray)
            current_match = max(0, min(100, 100 - confidence))
            overall_match_percent += current_match
            # Color-code based on match
            color = WHITE if current_match > 40 else RED
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, SIZE)
        except Exception as e:
            print(f"Prediction error: {e}")
            continue

    final_score = overall_match_percent / num_faces

    return frame, final_score


def gen_frames():
    camera = get_camera()
    # Only try to loop if camera was actually initialized
    if camera is not None and camera.isOpened():
        while True:
            success, frame = camera.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame, score = detect_and_score(frame, gray)
            ret, bw_frame = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Draw the score on the frame
            cv2.putText(
                bw_frame,
                f"Match: {score:.1f}%",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                WHITE,
                2,
            )

            ret, buffer = cv2.imencode(".jpg", bw_frame)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    else:
        print("Skipping video_feed: No camera hardware detected.")
        # Optional: yield a single "Camera Not Found" static image here


@app.route("/")
def index():
    # if IS_RENDER:
    #     return "<h1>Chudifier Online</h1><p>Upload an image to process it!</p>"
    return render_template("index.html")


@app.route("/process_frame", methods=["POST"])
def process_frame():
    file = request.files.get("frame")
    if not file:
        return jsonify({"error": "No frame"}), 400

    # Convert bytes to OpenCV image
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply threshold: 127 is the cutoff, 255 is the value assigned to white
        t_lower = 50
        t_upper = 150
        edge = cv2.Canny(gray, t_lower, t_upper)
        processed_frame, score = detect_and_score(edge, gray)

        # Encode the PROCESSED frame back to base64 to send to JS
        _, buffer = cv2.imencode(".jpg", processed_frame)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        print(round(score, 2))
        return jsonify(
            {
                "score": round(score, 2),
                "image": f"data:image/jpeg;base64,{encoded_image}",
            }
        )

    return jsonify({"score": 0})


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ... (all your functions remain the same) ...

# 1. INITIALIZE & LOAD IMMEDIATELY (Outside any if blocks)
print("Initializing Recognizer and Loading Data...")
# We use the global recognizer created at the top of the script
if os.path.exists(TRAINER_FILE):
    print(f"SUCCESS: Loading {TRAINER_FILE}")
    recognizer.read(TRAINER_FILE)
else:
    # If no trainer exists, we try to train it once on startup
    print("No saved data found. Training from scratch...")
    train_from_heic("./positives")
    if os.path.exists(TRAINER_FILE):  # If train_from_heic writes the file
        recognizer.read(TRAINER_FILE)

# 2. START THE SERVER
if __name__ == "__main__":
    # This block now ONLY handles the port and host settings
    if IS_RENDER:
        port = int(os.environ.get("PORT", 10000))
        app.run(host="0.0.0.0", port=port)
    else:
        app.run(host="127.0.0.1", port=5050, debug=True)
