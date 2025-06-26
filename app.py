import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__) 

# Load the updated MobileNetV2 model
model = load_model('drowsiness_mobilenetv2.keras')  

# Global alarm status
alarm_triggered = False

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (96, 96))  # Updated to 96x96 for MobileNetV2
    normalized = resized_frame / 255.0
    input_img = np.expand_dims(normalized, axis=0)  # (1, 96, 96, 3)
    return input_img

def gen_frames():
    global alarm_triggered
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        input_img = preprocess_frame(frame)
        prediction = model.predict(input_img, verbose=0)
        
        if prediction[0][0] > 0.5:
            label = "AWAKE"
            color = (0, 255, 0)
            alarm_triggered = False
        else:
            label = "DROWSY!"
            color = (0, 0, 255)
            alarm_triggered = True

        cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Encode frame to send over HTTP
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')  # Your HTML page to show webcam feed

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alarm_status')
def alarm_status():
    return "play" if alarm_triggered else "stop"

if __name__ == '__main__':  # FIXED
    app.run(debug=True)
