# Import necessary libraries
import cv2  # Library for computer vision
import dlib  # Library for machine learning, used for face detection and facial landmark prediction
import imutils  # Utility functions for image processing
import numpy as np  # Library for numerical operations
from scipy.spatial import distance as dist  # Functions for calculating distances
from imutils import face_utils  # Utilities for facial landmark detection
import pygame  # Library for sound playback
import time  # Time-related functions
import tkinter as tk  # Library for creating GUI applications
from tkinter import Frame  # Frame widget from tkinter for organizing GUI layout

class DrowsinessDetector:
    def __init__(self):
        pygame.mixer.init()  # Initialize the mixer module for sound playback

        # Threshold values for detecting blinks, mouth opening, and yawns
        self.blink_thresh = 0.4
        self.mouth_thresh = 0.6
        self.yawn_thresh = 10

        # Frame counters
        self.succ_frame = 2
        self.count_frame = 0
        self.yawn_frame = 0

        # Flags and counters for detecting blinks and yawns
        self.blink_detected = False
        self.yawn_detected = False
        self.blink_count = 0
        self.yawn_count = 0

        # Time and alert management variables
        self.face_not_detected_start_time = None
        self.face_not_detected_duration_thresh = 5
        self.face_not_detected_alert_active = False
        self.yawn_alert_start_time = None
        self.yawn_alert_duration_thresh = 1
        self.blink_start_time = None
        self.blink_alert_active = False
        self.blink_duration_thresh = 15
        self.blink_alert_display_time = None
        self.blink_alert_min_duration = 3

        # Variables for detecting eyes closed for 3 consecutive seconds
        self.eyes_closed_start_time = None
        self.eyes_closed_duration_thresh = 3
        self.eyes_closed_alert_active = False

        # Facial landmark indices for eyes and mouth
        self.L_start, self.L_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.R_start, self.R_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.M_start, self.M_end = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        # Initialize dlib's face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.landmark_predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Variables for calculating frames per second (FPS)
        self.last_time = time.time()
        self.current_time = time.time()
        self.fps = 0

    def play_alert_sound(self, loop=False):
        # Function to play alert sound
        pygame.mixer.music.load("alert_sound.mp3")
        pygame.mixer.music.play(-1 if loop else 0)

    def stop_alert_sound(self):
        # Function to stop alert sound
        pygame.mixer.music.stop()

    def calculate_EAR(self, eye):
        # Calculate Eye Aspect Ratio (EAR)
        y1 = dist.euclidean(eye[1], eye[5])
        y2 = dist.euclidean(eye[2], eye[4])
        x1 = dist.euclidean(eye[0], eye[3])
        EAR = (y1 + y2) / x1
        return EAR

    def calculate_MAR(self, mouth):
        # Calculate Mouth Aspect Ratio (MAR)
        y1 = dist.euclidean(mouth[2], mouth[10])
        y2 = dist.euclidean(mouth[4], mouth[8])
        x1 = dist.euclidean(mouth[0], mouth[6])
        MAR = (y1 + y2) / (2 * x1)
        return MAR

    def mark_eyeLandmark(self, frame, eyes):
        # Draw lines between eye landmarks
        for eye in eyes:
            pt1, pt2 = (eye[1], eye[5])
            pt3, pt4 = (eye[0], eye[3])
            cv2.line(frame, pt1, pt2, (200, 0, 0), 2)
            cv2.line(frame, pt3, pt4, (200, 0, 0), 2)
        return frame

    def mark_mouthLandmark(self, frame, mouth):
        # Draw lines between mouth landmarks
        for i in range(0, len(mouth) - 1):
            cv2.line(frame, mouth[i], mouth[i + 1], (0, 0, 200), 2)
        return frame

    def adjust_brightness(self, image, factor):
        # Adjust the brightness of the image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], factor)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def adjust_contrast(self, image, factor):
        # Adjust the contrast of the image
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.multiply(lab[:, :, 0], factor)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def process_frame(self, frame):
        # Resize and process the frame for facial landmark detection
        frame = imutils.resize(frame, width=640)
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.equalizeHist(img_gray)

        frame_low_brightness = self.adjust_brightness(frame, 0.5)
        frame_high_brightness = self.adjust_brightness(frame, 1.5)
        frame_low_contrast = self.adjust_contrast(frame, 0.5)
        frame_high_contrast = self.adjust_contrast(frame, 1.5)

        test_frame = frame

        faces = self.detector(test_frame)

        if len(faces) == 0:
            # If no face is detected, start the timer
            if self.face_not_detected_start_time is None:
                self.face_not_detected_start_time = time.time()
            else:
                elapsed_time = time.time() - self.face_not_detected_start_time
                if elapsed_time > self.face_not_detected_duration_thresh:
                    cv2.putText(test_frame, "Distraction", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 255), 1)
                    if not self.face_not_detected_alert_active:
                        self.play_alert_sound(loop=True)
                        self.face_not_detected_alert_active = True
        else:
            # Reset the timer if a face is detected
            if self.face_not_detected_start_time is not None:
                self.face_not_detected_start_time = None
            if self.face_not_detected_alert_active:
                self.stop_alert_sound()
                self.face_not_detected_alert_active = False

            for face in faces:
                shape = self.landmark_predict(test_frame, face)
                shape = face_utils.shape_to_np(shape)

                for lm in shape:
                    cv2.circle(test_frame, tuple(lm), 3, (10, 2, 200))

                lefteye = shape[self.L_start: self.L_end]
                righteye = shape[self.R_start: self.R_end]
                mouth = shape[self.M_start: self.M_end]

                left_EAR = self.calculate_EAR(lefteye)
                right_EAR = self.calculate_EAR(righteye)
                avg_EAR = (left_EAR + right_EAR) / 2
                test_frame = self.mark_eyeLandmark(test_frame, [lefteye, righteye])

                mouth_MAR = self.calculate_MAR(mouth)
                test_frame = self.mark_mouthLandmark(test_frame, mouth)

                cv2.putText(test_frame, f"EAR: {avg_EAR:.2f}", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 255), 1)
                cv2.putText(test_frame, f"MAR: {mouth_MAR:.2f}", (10, 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 255), 1)
                cv2.putText(test_frame, f"FPS: {self.fps:.2f}", (10, 90), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 255), 1)

                eye_visible = all(lefteye[:, 1] > 0) and all(righteye[:, 1] > 0)

                if avg_EAR < self.blink_thresh and eye_visible:
                    if self.count_frame >= self.succ_frame:
                        self.blink_detected = True
                        self.blink_count += 1
                    else:
                        self.count_frame += 1

                    # Check if eyes are closed for 3 consecutive seconds
                    if self.eyes_closed_start_time is None:
                        self.eyes_closed_start_time = time.time()
                    else:
                        elapsed_time_eyes_closed = time.time() - self.eyes_closed_start_time
                        if elapsed_time_eyes_closed >= self.eyes_closed_duration_thresh:
                            cv2.putText(test_frame, "Drowsiness", (10, 150), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
                            if not self.eyes_closed_alert_active:
                                self.play_alert_sound(loop=True)
                                self.eyes_closed_alert_active = True
                else:
                    self.count_frame = 0
                    self.eyes_closed_start_time = None
                    if self.eyes_closed_alert_active:
                        self.stop_alert_sound()
                        self.eyes_closed_alert_active = False

                if mouth_MAR > self.mouth_thresh:
                    self.yawn_frame += 1
                    if self.yawn_frame >= self.yawn_thresh:
                        self.yawn_detected = True
                        self.yawn_count += 1
                        self.yawn_alert_start_time = time.time()
                else:
                    self.yawn_frame = 0

                if self.blink_detected:
                    cv2.putText(test_frame, "Blink Detected", (10, 110), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
                    self.blink_detected = False
                if self.yawn_detected:
                    cv2.putText(test_frame, "Yawning Detected", (10, 130), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
                    self.play_alert_sound()
                    self.yawn_detected = False

        if self.yawn_alert_start_time is not None:
            elapsed_time_since_yawn = time.time() - self.yawn_alert_start_time
            if elapsed_time_since_yawn > self.yawn_alert_duration_thresh:
                self.stop_alert_sound()
                self.yawn_alert_start_time = None

        self.last_time = self.current_time
        self.current_time = time.time()
        self.fps = 1.0 / (self.current_time - self.last_time)

        return test_frame

class App:
    def __init__(self, root, detector):
        self.root = root
        self.detector = detector
        self.cam = None
        self.running = False

        # Frame for displaying the video feed
        self.video_frame = Frame(root)
        self.video_frame.pack()

        # Frame for control buttons
        self.controls_frame = Frame(root)
        self.controls_frame.pack()

        # Start button to start the video feed
        self.start_button = tk.Button(self.controls_frame, text="Start", command=self.start)
        self.start_button.pack(side=tk.LEFT)

        # Stop button to stop the video feed
        self.stop_button = tk.Button(self.controls_frame, text="Stop", command=self.stop)
        self.stop_button.pack(side=tk.LEFT)

        # Labels to display the number of blinks and yawns
        self.blinks_label = tk.Label(self.controls_frame, text="Blink Frames: 0")
        self.blinks_label.pack(side=tk.LEFT)
        self.yawns_label = tk.Label(self.controls_frame, text="Yawn Frames: 0")
        self.yawns_label.pack(side=tk.LEFT)

    def start(self):
        # Start the video feed
        if not self.running:
            self.cam = cv2.VideoCapture(0)
            self.running = True
            self.update_frame()

    def update_frame(self):
        # Update the video frame and process it
        if self.running:
            ret, frame = self.cam.read()
            if ret:
                processed_frame = self.detector.process_frame(frame)
                cv2.imshow("Facial Landmark Detection", processed_frame)
                self.blinks_label.config(text=f"Blink Frames: {self.detector.blink_count}")
                self.yawns_label.config(text=f"Yawn Frames: {self.detector.yawn_count}")
                self.root.after(10, self.update_frame)

    def stop(self):
        # Stop the video feed
        if self.running:
            self.running = False
            self.cam.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Main application entry point
    root = tk.Tk()
    root.title("Drowsiness Detector")
    detector = DrowsinessDetector()
    app = App(root, detector)
    root.mainloop()
