#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main program to control the game using chair orientation and gestures.
Uses the "chair_control" script to manage inputs sent to the game
and the "hand_commands" script to detect the positions of the person on the chair.

The camera is positioned in portrait mode to have a larger vertical field of view.

"new_matrix_state" is a callback that receives the phone angle and simulates joystick rotation.
In the main loop, an image is captured, landmarks are detected, and the person's position is determined.

"""

import cv2
import mediapipe as mp
import time
import threading
from multiprocessing import Value
from oscpy.server import OSCThreadServer

from chair_control import (play_accel, press_boost, press_object, press_rescue, 
                           counter_drift, orient_to_steer, joystick_play)

from hand_commands import detect_pose


# A dead zone of 0 allows the user to have faster feedback on the exact rotation of the chair.
joystick_deadzone = 0.0
    
def new_matrix_state(*matrix_state):
    global drift_state
    global current_steer_command  

    phone_orient = matrix_state[1]
    steer = orient_to_steer(phone_orient, joystick_deadzone)
    current_steer_command.value = steer
    
    # If not drifting
    if drift_state == 0:
        joystick_play(steer)
        
    # Otherwise, drifting is handled by the "drift_thread" thread.
    

# Using the rotation matrix sent by OSC    
osc = OSCThreadServer()
sock = osc.listen(address='0.0.0.0', port=8000, default=True)
osc.bind(b'/gyrosc/rmatrix', new_matrix_state)

drift_thread = None
drift_state = 0 # -1 = drift left; 0 = no drift; 1 = drift right

drift_commands_queue = []
current_steer_command = Value('d', 0.0)
last_drift_time = time.time()

last_pose_label = None

cap = cv2.VideoCapture(0)
# The image size is reduced to improve landmark detection performance and reduce position detection latency.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# mp_pose.Pose performs gesture detection.
with mp_pose.Pose(
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2,
    smooth_landmarks=True,
    model_complexity=0) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      continue

    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if not results.pose_landmarks:
        continue
    
    # Get the person's landmarks
    landmarks = results.pose_landmarks.landmark
    
    # Deduce their position
    pose_label = detect_pose(landmarks)
    
    # If not drifting and the drift thread has finished
    if drift_state != 0 and not drift_thread.is_alive():
        # Reset variables related to drifting
        drift_thread = None
        drift_state = 0
        last_drift_time = time.time()
    
    # If the user changes position
    if last_pose_label is None or pose_label != last_pose_label:
        # Execute the input associated with the new position
        
        if pose_label == "HANDS_SHOULDER":
            play_accel(accel_state=0)
            
        elif pose_label == "HANDS_KNEE":
            play_accel(accel_state=1)
        
        elif pose_label == "HANDS_HIPS":
            press_boost(0.5)
            play_accel(accel_state=1)
        
        elif pose_label == "HANDS_KNEE_CROSS":
            press_object()
            
        elif pose_label == "HANDS_SHOULDER_CROSS":
            press_rescue()
    
        elif (pose_label in ["RIGHT_HAND_ABOVE_KNEE", "LEFT_HAND_ABOVE_KNEE"] and
              len(drift_commands_queue) < 1) and drift_thread is None:
            # Drift commands are added to a queue
            # This allows chaining drifts more easily
            drift_commands_queue.append((pose_label=="LEFT_HAND_ABOVE_KNEE",
                                         pose_label=="RIGHT_HAND_ABOVE_KNEE"))
            
        last_pose_label = pose_label
    
    # If there is a drift command in the queue and not currently drifting
    # A minimum of 0.6s is required before drifting again after the end of a drift
    if len(drift_commands_queue) > 0 and (
            drift_state == 0 and (time.time() - last_drift_time) > 0.6):
        # Execute the drift
        
        left_drift_cmd, right_drift_cmd = drift_commands_queue.pop(0)
        
        # Drifts are executed in a thread to avoid disrupting the main program loop
        
        if left_drift_cmd:
            drift_thread = threading.Thread(
                target=counter_drift,
                name="Drift thread",
                args=(-1, current_steer_command)
            )
            drift_thread.start()
            drift_state = -1
            
        elif right_drift_cmd:
            drift_thread = threading.Thread(
                target=counter_drift,
                name="Drift thread",
                args=(1, current_steer_command)
            )
            drift_thread.start()
            drift_state = 1
    
    # Draw landmarks on the image as well as the detected position
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    image = cv2.flip(image, 1)
    cv2.putText(image, f"Pose: {pose_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    image = cv2.resize(image, None, fx=1, fy=1)
    
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(1) & 0xFF == 27:
      break 
  
cap.release()
cv2.destroyAllWindows()
