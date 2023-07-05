import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from keras.utils import load_img, img_to_array
from keras.models import load_model
from collections import deque
import datetime
# from pygame import mixer
from playsound import playsound
# import smtplib
# import ssl

# mixer.init()
# mixer.music.load("Model/alarm2.wav")


def sound_alarm():
    # mixer.music.play()
    playsound(
        r"C:/Users/ArulVirumbi/Downloads/CAT/Driver Fatigue Detection/Model/Alarm3.wav")


def compute_eye_aspect_ratio(eye):
    """The function computes the eye aspect ratio"""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)  # eye aspect ratio


def compute_average_eye_aspect_ratios(shape):
    """The function computes the average of the two eye aspect ratios"""
    left_eye = shape[42:48]  # indexes of left eye
    right_eye = shape[36:42]  # indexes of right eye
    left_eye_aspect_ratio = compute_eye_aspect_ratio(
        left_eye)  # left eye aspect ratio
    right_eye_aspect_ratio = compute_eye_aspect_ratio(
        right_eye)  # right eye aspect ratio
    # average the eye aspect ratios
    return (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0


FRAMES_PER_SECOND = 3
YAWN_DURATION = 2  # humans yawn for an average length of 7 seconds


def blink_count_threshold(current_time, travel_duration):

    # basic threshold (seconds per frame * seconds of blink)
    threshold = FRAMES_PER_SECOND * (10 / 2)
    # the threshold is lower if the time is 22:00 - 05:00
    if current_time.hour <= 22 or current_time.hour >= 5:
        threshold -= 1

    # the threshold is lower if the travel duration is more than 2.5 hours
    if travel_duration.seconds >= 60 * 60 * 2.5:
        threshold -= 1
    return threshold


def yawn_count_threshold(current_time, travel_duration):

    # basic threshold (seconds per frame * seconds of single yawn * number of yawns)
    threshold = FRAMES_PER_SECOND * YAWN_DURATION * 3
    # the threshold is lower if the time is 22:00 - 05:00
    if current_time.hour >= 22 or current_time.hour <= 5:
        threshold -= FRAMES_PER_SECOND * YAWN_DURATION

    # the threshold is lower if the travel duration is more than 2.5 hours
    if travel_duration.seconds >= 60 * 60 * 2.5:
        threshold -= FRAMES_PER_SECOND * YAWN_DURATION

    return threshold


def compute_lips_distance(shape):
    start_top_lip = shape[50:53]  # start indexes of top lip
    end_top_lip = shape[61:64]  # end indexes of top lip
    start_low_lip = shape[56:59]  # start indexes of low lip
    end_low_lip = shape[65:68]  # end indexes of low lip
    top_lip = np.concatenate(
        (start_top_lip, end_top_lip))  # top lip coordinates
    low_lip = np.concatenate(
        (start_low_lip, end_low_lip))  # low lip coordinates
    top_mean = np.mean(top_lip, axis=0)  # top lip mean
    low_mean = np.mean(low_lip, axis=0)  # low lip mean
    # distance between the top and the low lips
    return abs(top_mean[1] - low_mean[1])


def process_frame(gray_frame):
    frame = cv2.resize(gray_frame, (256, 256))  # resize the frame
    # convert frame to array (height, width, channels)
    frame = img_to_array(frame)
    # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    frame = np.expand_dims(frame, axis=0)
    frame /= 255.  # set values [0, 1]
    return frame


def predict_yawn(gray_frame, model):
    """The function checks for yawn in the grayscale frame, and returns the predictions of [not yawn, yawn]"""
    return model.predict(process_frame(gray_frame))[0]


# constants and thresholds
# number of frame per a second the drowsiness classification is based on
FRAMES_PER_SECOND = 3
MINUTES_PER_WINDOW = 5  # approximate number of minutes the frame window contains
# frame window size (60 seconds * minutes * frames)
WINDOW_SIZE = 60 * MINUTES_PER_WINDOW * FRAMES_PER_SECOND
EYE_ASPECT_RATIO_THRESHOLD = 0.2  # eye aspect ratio threshold

yawn_queue = deque()  # yawn window queue
blink_counter = yawn_counter = 0
alarm_on = False

# Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)
# number of blinks / yawns
# beginning time; last time a frame was analyzed
start_drive_time = last_frame_time = datetime.datetime.now()
travel_duration = datetime.timedelta(0)

yawn_queue = deque()

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Model\shape_predictor_68_face_landmarks.dat")
# load the yawning classification model
model = load_model("Model\yawn_detection.h5")


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    # detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # see if a sufficient time passed since the previous frame was analysed
        if datetime.datetime.now() - last_frame_time >= datetime.timedelta(seconds=1/FRAMES_PER_SECOND):

            # to detect drowsiness, check for a blink and a yawn in the frame:

            # blink - by computing the eye aspect ratios
            if compute_average_eye_aspect_ratios(landmarks) < EYE_ASPECT_RATIO_THRESHOLD:
                # keep counting if the driver is still blinking to see how many frames the blink lasts
                blink_counter += 1
            else:
                blink_counter = 0  # stop counting and reset counter if the driver is not blinking anymore

        prediction = predict_yawn(gray, model)  # [not yawn, yawn]
        yawn = prediction[0] <= prediction[1] and True or False

        if len(yawn_queue) > WINDOW_SIZE:  # if the queue is full
            # pop the first frame (oldest) out, and update the counter
            yawn_counter -= 1 if yawn_queue.popleft() else 0

        yawn_queue.append(yawn)  # insert the new frame to the end of the queue
        yawn_counter += 1 if yawn else 0  # update the counter

        # compute the current time and the drive duration to determine the thresholds (late and long time = lower thresholds)
        current_time = datetime.datetime.now()
        travel_duration = current_time - start_drive_time

        # compare the counters to thresholds to see if the driver is classified as drowsy - based on blinks OR yawns
        if blink_counter >= blink_count_threshold(current_time, travel_duration) or \
                yawn_counter >= yawn_count_threshold(current_time, travel_duration):

            # reset queue and counters to
            yawn_queue = deque()
            blink_counter = 0
            yawn_counter = 0

            if not alarm_on:  # check if the alarm is not on
                sound_alarm()  # start a thread to have the alarm sound played in the background
                alarm_on = True

        else:  # not classified as fatigue
            alarm_on = False

        last_frame_time = current_time
        cv2.putText(frame, str(travel_duration)[
                    :-7], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
