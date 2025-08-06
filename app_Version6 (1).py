from flask import Flask, render_template, Response, request, jsonify
import cv2
import face_recognition
import numpy as np
import threading
import time

app = Flask(__name__)

# Haar Cascade for Viola-Jones face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Globals for storing faceprint, anomaly log, tab switches, and audio
known_face_encoding = None
anomalies = []
tab_switch_count = 0
audio_detected = False
last_tab_switch_log = 0

def capture_faceprint():
    """Capture and store the initial faceprint (encoding) for verification."""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    if face_locations:
        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
        return face_encoding
    return False

def detect_mobile_object(frame):
    # Placeholder: looks for large rectangular regions with "phone-like" aspect ratio
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edged = cv2.Canny(blurred, 50, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:  # large enough
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.4 < aspect_ratio < 0.7:  # phone aspect ratio
                return True
    return False

def face_angle_deviation(face_landmarks):
    # Estimate face orientation using landmarks (simple version)
    # Returns yaw (left-right) angle in degrees
    # We'll use the horizontal displacement between eyes and nose
    left_eye = np.mean(face_landmarks['left_eye'], axis=0)
    right_eye = np.mean(face_landmarks['right_eye'], axis=0)
    nose = np.mean(face_landmarks['nose_bridge'], axis=0)
    eyes_center = (left_eye + right_eye) / 2
    dx = nose[0] - eyes_center[0]
    dy = nose[1] - eyes_center[1]
    angle = np.degrees(np.arctan2(dx, dy))  # yaw
    return abs(angle)

def gen_frames():
    global known_face_encoding, anomalies, tab_switch_count, audio_detected
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_face_encodings = face_recognition.face_encodings(rgb_frame)
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame) if faces is not None and len(faces) == 1 else []

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        num_faces = len(faces)

        # Anomaly: No face
        if num_faces == 0:
            anomalies.append("No face was detected during proctoring.")
        # Anomaly: Multiple faces
        elif num_faces > 1:
            anomalies.append("Multiple faces were detected during proctoring.")
        # Anomaly: Faceprint missing
        elif known_face_encoding is not None:
            match_found = False
            for encoding in current_face_encodings:
                match = face_recognition.compare_faces([known_face_encoding], encoding, tolerance=0.5)
                if match[0]:
                    match_found = True
                    break
            if not match_found and num_faces == 1:
                anomalies.append("Faceprint detected during verification is missing")

        # Mobile detection
        if detect_mobile_object(frame):
            anomalies.append("Mobile object detected.")

        # Face angle deviation
        if face_landmarks_list:
            deviation = face_angle_deviation(face_landmarks_list[0])
            if deviation > 30:
                anomalies.append(f"Candidate face angle deviation more than 25-30 degrees ({int(deviation)}°).")

        # Voice/audio detection (threaded, see below)
        if audio_detected:
            anomalies.append("Voice/audio detected.")

        # Tab switching
        if tab_switch_count > 2 and (time.time() - last_tab_switch_log > 3):
            anomalies.append("Tab switching during the exam more than 2 times.")
            globals()['last_tab_switch_log'] = time.time()
        elif 0 < tab_switch_count <= 2 and (time.time() - last_tab_switch_log > 3):
            anomalies.append("Tab switching during exam less than 2 times.")
            globals()['last_tab_switch_log'] = time.time()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Audio detection (simple energy threshold)
def audio_thread():
    import sounddevice as sd
    import numpy as np
    global audio_detected
    duration = 0.5  # seconds
    threshold = 0.03  # adjust based on noise
    while True:
        audio = sd.rec(int(44100 * duration), samplerate=44100, channels=1, dtype='float64')
        sd.wait()
        volume_norm = np.linalg.norm(audio) / len(audio)
        if volume_norm > threshold:
            audio_detected = True
        else:
            audio_detected = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    """Endpoint to capture and store the initial faceprint."""
    global known_face_encoding
    face_encoding = capture_faceprint()
    if face_encoding is not False:
        known_face_encoding = face_encoding
        return jsonify({'status': 'success', 'message': 'Faceprint captured for verification.'})
    else:
        return jsonify({'status': 'fail', 'message': 'No face detected during verification.'})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/anomalies')
def get_anomalies():
    global anomalies
    response = {'anomalies': anomalies[-10:]}  # last 10
    return jsonify(response)

@app.route('/tab_switch', methods=['POST'])
def tab_switch():
    global tab_switch_count
    tab_switch_count += 1
    return jsonify({'status': 'ok', 'count': tab_switch_count})

# Start audio detection
def start_audio_detection():
    t = threading.Thread(target=audio_thread, daemon=True)
    t.start()

if __name__ == '__main__':
    start_audio_detection()
    app.run(debug=True)