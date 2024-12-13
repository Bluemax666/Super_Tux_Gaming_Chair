#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The "detect_pose" function detects the position of the person on the chair using landmarks.
List of detected positions: ["HANDS_KNEE", "HANDS_HIPS", "HANDS_KNEE_CROSS", "LEFT_HAND_ABOVE_KNEE",
                               "RIGHT_HAND_ABOVE_KNEE", "HANDS_SHOULDER", "HANDS_SHOULDER_CROSS", "UNKNOWN_POSE"]
"""

# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#configurations_options
import cv2
import mediapipe as mp
import numpy as np


def detect_pose(landmarks):
    # Define the positions
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_knee = landmarks[25]
    right_knee = landmarks[26]
    left_pinky = landmarks[17]
    right_pinky = landmarks[18]
    left_elbow = landmarks[13]
    right_elbow = landmarks[14]
    left_hip = landmarks[23]
    right_hip = landmarks[24]

    # Calculate distances
    left_pinky_to_left_shoulder = np.linalg.norm([left_pinky.x - left_shoulder.x, left_pinky.y - left_shoulder.y])
    right_pinky_to_right_shoulder = np.linalg.norm([right_pinky.x - right_shoulder.x, right_pinky.y - right_shoulder.y])
    
    left_pinky_to_left_knee = np.linalg.norm([left_pinky.x - left_knee.x, left_pinky.y - left_knee.y])
    right_pinky_to_right_knee = np.linalg.norm([right_pinky.x - right_knee.x, right_pinky.y - right_knee.y])
    
    left_pinky_to_right_shoulder = np.linalg.norm([left_pinky.x - right_shoulder.x, left_pinky.y - right_shoulder.y])
    right_pinky_to_left_shoulder = np.linalg.norm([right_pinky.x - left_shoulder.x, right_pinky.y - left_shoulder.y])
    
    left_pinky_to_right_knee = np.linalg.norm([left_pinky.x - right_knee.x, left_pinky.y - right_knee.y])
    right_pinky_to_left_knee = np.linalg.norm([right_pinky.x - left_knee.x, right_pinky.y - left_knee.y])
    
    # Calculate distances for HANDS_HIPS
    left_hand_to_left_hip = np.linalg.norm([left_pinky.x - left_hip.x, left_pinky.y - left_hip.y]) # left_pinky.z - left_hip.z])
    right_hand_to_right_hip = np.linalg.norm([right_pinky.x - right_hip.x, right_pinky.y - right_hip.y]) #, right_pinky.z - right_hip.z])
    
    # Calculate vertical (y-axis only) distances for above knee check
    left_pinky_to_left_knee_y = abs(left_pinky.y - left_knee.y)
    right_pinky_to_right_knee_y = abs(right_pinky.y - right_knee.y)

    # Define threshold distances for determining shoulder and knee positions
    threshold_shoulder_dist = 0.20  # Allowable distance from shoulder for "HANDS_SHOULDER"
    threshold_above_knee = 0.20    # Threshold for "above knee" poses
    threshold_above_knee_diff = 0.11
    threshold_hips_dist = 0.125 # Threshold distance for HANDS_HIPS detection
    
    # Determine pose
    left_pinky_to_nearest_shoulder = min(left_pinky_to_left_shoulder, left_pinky_to_right_shoulder)
    right_pinky_to_nearest_shoulder = min(right_pinky_to_right_shoulder, right_pinky_to_left_shoulder)

    
    if left_elbow.y < left_shoulder.y and right_elbow.y < right_shoulder.y:
        return "HANDS_SHOULDER_CROSS"
    
    # Check for HANDS_SHOULDER or HANDS_SHOULDER_CROSS
    if (left_pinky_to_nearest_shoulder < left_pinky_to_left_knee and 
        right_pinky_to_nearest_shoulder < right_pinky_to_right_knee and
        left_pinky_to_nearest_shoulder < threshold_shoulder_dist and
        right_pinky_to_nearest_shoulder < threshold_shoulder_dist):
        
        # Determine if it's a crossed position
        if (left_pinky_to_right_shoulder < left_pinky_to_left_shoulder or 
            right_pinky_to_left_shoulder < right_pinky_to_right_shoulder):
            return "HANDS_SHOULDER_CROSS"
        else:
            return "HANDS_SHOULDER"

    
    elif (right_pinky_to_right_knee_y > threshold_above_knee and 
          left_pinky_to_left_knee_y < threshold_above_knee and 
          right_pinky_to_right_knee_y - left_pinky_to_left_knee_y > threshold_above_knee_diff):
        return "RIGHT_HAND_ABOVE_KNEE"
    
    elif (left_pinky_to_left_knee_y > threshold_above_knee and 
          right_pinky_to_right_knee_y < threshold_above_knee and 
          left_pinky_to_left_knee_y - right_pinky_to_right_knee_y > threshold_above_knee_diff):
        return "LEFT_HAND_ABOVE_KNEE"
    
    elif (left_pinky_to_left_knee < left_pinky_to_left_shoulder and 
          right_pinky_to_right_knee < right_pinky_to_right_shoulder):
        
        if (left_pinky_to_right_knee < left_pinky_to_left_knee or 
            right_pinky_to_left_knee < right_pinky_to_right_knee):
            return "HANDS_KNEE_CROSS"
        else:
            if left_hand_to_left_hip < threshold_hips_dist and right_hand_to_right_hip < threshold_hips_dist:
                return "HANDS_HIPS"
            else:
                return "HANDS_KNEE"
    
    return "UNKNOWN_POSE"


if __name__ == '__main__':
    # Allows testing only the detection of positions
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    
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
        
        landmarks = results.pose_landmarks.landmark
        
        pose_label = detect_pose(landmarks)
        print("Detected pose:", pose_label)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        cv2.putText(image, f"Pose: {pose_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(1) & 0xFF == 27:
          break
      
    cap.release()
    cv2.destroyAllWindows()
