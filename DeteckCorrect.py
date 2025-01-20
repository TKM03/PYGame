import cv2
import mediapipe as mp
import numpy as np
import math
import textwrap

# Load the reference image
reference_image = cv2.imread('reference_pose.jpg')
reference_height, reference_width, _ = reference_image.shape

# Initialize MediaPipe for pose tracking
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)  # Adjust model_complexity
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

def check_pushup_pose(landmarks, frame):
    suggestions = []

    # Get coordinates
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    # Calculate angles
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
    right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
    back_angle = calculate_angle(left_shoulder, left_hip, left_ankle)

    # Check elbow angles
    if left_elbow_angle > 110 or right_elbow_angle > 110:
        suggestions.append("Suggestion : Lower your body more")
        suggestions.append("(elbow angle too wide)")
    elif left_elbow_angle < 70 or right_elbow_angle < 70:
        suggestions.append("Raise your body more")
        suggestions.append("(elbow angle too narrow)")

    # Check shoulder angles
    if left_shoulder_angle < 30 or right_shoulder_angle < 30:
        suggestions.append("Keep your arms more perpendicular")
        suggestions.append("to your body (shoulder angle too narrow)")

    # Check back angle
    if back_angle < 160:
        suggestions.append("Keep your back straight")
        suggestions.append("(back angle too acute)")

    return suggestions

def draw_text(frame, text_lines, position, font, font_scale, color, thickness):
    y = position[1]
    for line in text_lines:
        cv2.putText(frame, line, (position[0], y), font, font_scale, color, thickness)
        y += int(font_scale * 30)  # Adjust line height based on font scale

try:
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Pose Tracking', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Pose Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            landmarks = results.pose_landmarks.landmark
            suggestions = check_pushup_pose(landmarks, frame)

            if suggestions:
                draw_text(frame, suggestions, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                draw_text(frame, ["Correct Push-Up Pose"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Pose Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
