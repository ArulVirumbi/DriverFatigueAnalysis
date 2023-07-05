from flask import Flask, render_template, Response
import cv2
import numpy as np
import dlib
from imutils import face_utils


app = Flask(__name__, template_folder='template')

video_stream = None
video_started = False


@app.route('/')
def index():
    return render_template('index.html')


def video_output():
    cap = cv2.VideoCapture(0)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "Model/shape_predictor_68_face_landmarks.dat")

    sleep = 0
    drowsy = 0
    active = 0
    status = ""
    color = (0, 0, 0)

    def compute(ptA, ptB):
        dist = np.linalg.norm(ptA-ptB)
        return dist

    def blinked(a, b, c, d, e, f):
        up = compute(b, d) + compute(c, e)
        down = compute(a, f)
        ratio = up/(2.0*down)

        # Checking if it is blinked
        if (ratio > 0.25):
            return 2
        elif (ratio > 0.21 and ratio <= 0.25):
            return 1
        else:
            return 0

    def yawning(a, b, c, d):
        ulip, llip = [], []
        for i in range(3):
            ulip.append(a[i])
            llip.append(c[i])
        for i in range(3):
            ulip.append(b[i])
            llip.append(d[i])

        ulip_mean = np.mean(ulip, axis=0)

        llip_mean = np.mean(llip, axis=0)
        # Calculate the distance between the centroids
        lips_dist = compute(ulip_mean, llip_mean)
        return lips_dist

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

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                landmarks = predictor(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)

                # The numbers are actually the landmarks which will show eye
                left_blink = blinked(
                    landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
                right_blink = blinked(
                    landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

                yawn = yawning(
                    landmarks[50:53], landmarks[61:64], landmarks[65:68], landmarks[56:59])

                if yawn > 20:
                    # drowsy+=5
                    status = "Drowsy !, yawning"
                if (left_blink == 0 or right_blink == 0):
                    sleep += 1
                    drowsy = 0
                    active = 0
                    if (sleep > 5):
                        status = "SLEEPING !!!"
                        color = (255, 0, 0)

                elif (left_blink == 1 or right_blink == 1):
                    sleep = 0
                    drowsy += 1
                    active = 0
                    if (drowsy > 5):
                        status = "Drowsy !, droopy eyes"
                        color = (0, 0, 255)
                else:
                    drowsy = 0
                    sleep = 0
                    active += 1
                    if (active > 5):
                        status = "Active :)"
                        color = (0, 255, 0)

                for n in range(0, 68):
                    (x, y) = landmarks[n]
                    cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
        else:
            status = "sleeping !!!, slanting head"
            color = (255, 0, 0)
        cv2.putText(frame, status, (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        # cv2.imshow("Frame", frame)
        # cv2.imshow("Result of detector", face_frame)

        # Encode the frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        # Yield the frame for the video stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    # Release video capture
    cap.release()
    video_stream = None


@app.route('/video_feed')
def video_feed():
    return Response(video_output(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_video')
def start_video():
    global video_stream, video_started
    if not video_started:
        video_started = True
        video_stream = video_output()
    return 'Video stream started'


@app.route('/stop_video')
def stop_video():
    global video_stream, video_started
    if video_started:
        video_started = False
        video_stream = None
    return 'Video stream stopped'


if __name__ == '__main__':
    app.run(debug=True, port=5001)
