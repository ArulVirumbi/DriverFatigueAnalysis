from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from keras.utils import load_img, img_to_array
from keras.models import load_model
from collections import deque
import datetime
from playsound import playsound


app = Flask(__name__, template_folder='template')

video_stream = None
video_started = False


@app.route('/')
def index():
    return render_template('index.html')


def video_output():
    status = "hello"

    def sound_alarm():
        # mixer.music.play()
        playsound(
            r"C:/Users/ArulVirumbi/Downloads/CAT/Driver Fatigue Detection/Model/Alarm3.wav")

    def compute_eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)  # eye aspect ratio

    def compute_average_eye_aspect_ratios(shape):
        left_eye = shape[42:48]
        right_eye = shape[36:42]
        left_eye_aspect_ratio = compute_eye_aspect_ratio(
            left_eye)
        right_eye_aspect_ratio = compute_eye_aspect_ratio(
            right_eye)

        return (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0

    FRAMES_PER_SECOND = 3
    YAWN_DURATION = 2

    def blink_count_threshold(current_time, travel_duration):

        # basic threshold (seconds per frame * seconds of blink)
        threshold = FRAMES_PER_SECOND * (10 / 2)
        if current_time.hour >= 22 or current_time.hour <= 5:
            threshold -= 1
        if travel_duration.seconds >= 60 * 60 * 2.5:
            threshold -= 1
        return threshold

    def yawn_count_threshold(current_time, travel_duration):
        threshold = FRAMES_PER_SECOND * YAWN_DURATION * 3
        if current_time.hour >= 22 or current_time.hour <= 5:
            threshold -= FRAMES_PER_SECOND * YAWN_DURATION
        if travel_duration.seconds >= 60 * 60 * 2.5:
            threshold -= FRAMES_PER_SECOND * YAWN_DURATION

        return threshold

    def compute_lips_distance(shape):
        start_top_lip = shape[50:53]
        end_top_lip = shape[61:64]
        start_low_lip = shape[56:59]
        end_low_lip = shape[65:68]
        top_lip = np.concatenate(
            (start_top_lip, end_top_lip))
        low_lip = np.concatenate(
            (start_low_lip, end_low_lip))
        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)
        return abs(top_mean[1] - low_mean[1])

    def process_frame(gray_frame):
        frame = cv2.resize(gray_frame, (256, 256))
        frame = img_to_array(frame)
        frame = np.expand_dims(frame, axis=0)
        frame /= 255.
        return frame

    def predict_yawn(gray_frame, model):
        return model.predict(process_frame(gray_frame))[0]

    FRAMES_PER_SECOND = 3
    MINUTES_PER_WINDOW = 5
    WINDOW_SIZE = 60 * MINUTES_PER_WINDOW * FRAMES_PER_SECOND
    EYE_ASPECT_RATIO_THRESHOLD = 0.2

    yawn_queue = deque()
    blink_counter = yawn_counter = 0
    alarm_on = False

    cap = cv2.VideoCapture(0)
    start_drive_time = last_frame_time = datetime.datetime.now()
    travel_duration = datetime.timedelta(0)

    yawn_queue = deque()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "Model\shape_predictor_68_face_landmarks.dat")

    model = load_model("Model\yawn_detection.h5")

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        if faces:
            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()

                face_frame = frame.copy()
                cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                landmarks = predictor(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)

                if datetime.datetime.now() - last_frame_time >= datetime.timedelta(seconds=1/FRAMES_PER_SECOND):
                    if compute_average_eye_aspect_ratios(landmarks) < EYE_ASPECT_RATIO_THRESHOLD:
                        blink_counter += 1
                    else:
                        blink_counter = 0

                prediction = predict_yawn(gray, model)
                yawn = prediction[0] <= prediction[1] and True or False

                if len(yawn_queue) > WINDOW_SIZE:
                    yawn_counter -= 1 if yawn_queue.popleft() else 0

                yawn_queue.append(yawn)
                yawn_counter += 1 if yawn else 0

                current_time = datetime.datetime.now()
                travel_duration = current_time - start_drive_time

                print(blink_counter, yawn_counter)
                if blink_counter >= blink_count_threshold(current_time, travel_duration) or \
                        yawn_counter >= yawn_count_threshold(current_time, travel_duration):

                    yawn_queue = deque()
                    blink_counter = 0
                    yawn_counter = 0

                    if not alarm_on:
                        sound_alarm()
                        alarm_on = True

                else:
                    alarm_on = False
                last_frame_time = current_time
                for n in range(0, 68):
                    (x, y) = landmarks[n]
                    cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
        else:
            sound_alarm()

        cv2.putText(frame, str(travel_duration)[
                    :-7], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    # Release video capture
    cap.release()
    video_stream = None


@app.route('/video_feed')
def video_feed():
    return Response(video_output(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream_text')
def stream_text():
    global status
    return Response(status, mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
