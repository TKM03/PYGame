import cv2
import mediapipe as mp

class Sprite:
    def __init__(self, name):
        self.name = name
        self.x = 0
        self.y = 0
        self.costume = ""
        self.size = 1
        self.visible = False

    def switchcostume(self, costume):
        self.costume = costume

    def setsize(self, size):
        self.size = size

    def setx(self, x):
        self.x = int(x)

    def sety(self, y):
        self.y = int(y)

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def draw(self, frame):
        if self.visible:
            radius = int(self.size / 2)
            cv2.circle(frame, (self.x, self.y), radius, (0, 255, 0), -1)


def draw_line(frame, sprite1, sprite2):
    if sprite1.visible and sprite2.visible:
        cv2.line(frame, (sprite1.x, sprite1.y), (sprite2.x, sprite2.y), (255, 0, 0), 2)

# Create Sprite objects for key body parts
nose = Sprite('Nose')
left_shoulder = Sprite('Left Shoulder')
right_shoulder = Sprite('Right Shoulder')
left_elbow = Sprite('Left Elbow')
right_elbow = Sprite('Right Elbow')
left_wrist = Sprite('Left Wrist')
right_wrist = Sprite('Right Wrist')
left_hip = Sprite('Left Hip')
right_hip = Sprite('Right Hip')
left_knee = Sprite('Left Knee')
right_knee = Sprite('Right Knee')
left_ankle = Sprite('Left Ankle')
right_ankle = Sprite('Right Ankle')

# Initialize MediaPipe for pose tracking
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Set costumes and sizes for each body part sprite
body_parts = [nose, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, 
              left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]

for part in body_parts:
    part.switchcostume("circle")
    part.setsize(10)

# Start video capture
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        if result.pose_landmarks:
            # Get coordinates for each body part
            landmarks = result.pose_landmarks.landmark

            # Update position and show each body part sprite
            nose.setx(landmarks[mp_pose.PoseLandmark.NOSE].x * frame.shape[1])
            nose.sety(landmarks[mp_pose.PoseLandmark.NOSE].y * frame.shape[0])
            nose.show()

            left_shoulder.setx(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1])
            left_shoulder.sety(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0])
            left_shoulder.show()

            right_shoulder.setx(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1])
            right_shoulder.sety(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0])
            right_shoulder.show()

            left_elbow.setx(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * frame.shape[1])
            left_elbow.sety(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * frame.shape[0])
            left_elbow.show()

            right_elbow.setx(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame.shape[1])
            right_elbow.sety(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame.shape[0])
            right_elbow.show()

            left_wrist.setx(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1])
            left_wrist.sety(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0])
            left_wrist.show()

            right_wrist.setx(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1])
            right_wrist.sety(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0])
            right_wrist.show()

            left_hip.setx(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * frame.shape[1])
            left_hip.sety(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * frame.shape[0])
            left_hip.show()

            right_hip.setx(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * frame.shape[1])
            right_hip.sety(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * frame.shape[0])
            right_hip.show()

            left_knee.setx(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * frame.shape[1])
            left_knee.sety(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * frame.shape[0])
            left_knee.show()

            right_knee.setx(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * frame.shape[1])
            right_knee.sety(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * frame.shape[0])
            right_knee.show()

            left_ankle.setx(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * frame.shape[1])
            left_ankle.sety(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * frame.shape[0])
            left_ankle.show()

            right_ankle.setx(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * frame.shape[1])
            right_ankle.sety(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame.shape[0])
            right_ankle.show()

        else:
            # Hide all body part sprites if no body is detected
            for part in body_parts:
                part.hide()

        # Draw lines between key body parts to form the skeleton
        draw_line(frame, left_shoulder, right_shoulder)
        draw_line(frame, left_shoulder, left_elbow)
        draw_line(frame, left_elbow, left_wrist)
        draw_line(frame, right_shoulder, right_elbow)
        draw_line(frame, right_elbow, right_wrist)
        draw_line(frame, left_shoulder, left_hip)
        draw_line(frame, right_shoulder, right_hip)
        draw_line(frame, left_hip, right_hip)
        draw_line(frame, left_hip, left_knee)
        draw_line(frame, left_knee, left_ankle)
        draw_line(frame, right_hip, right_knee)
        draw_line(frame, right_knee, right_ankle)

        # Draw each sprite on the frame
        for part in body_parts:
            part.draw(frame)

        # Display the frame
        cv2.imshow('Pose Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
