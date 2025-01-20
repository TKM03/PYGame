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

    def draw(self, frame, multi_hand_landmarks):
        if self.visible:
            # Draw a filled circle for the sprite
            radius = int(self.size / 2)
            cv2.circle(frame, (self.x, self.y), radius, (0, 255, 0), -1)
            
            # Draw lines connecting the finger landmarks
            if multi_hand_landmarks:
                for hand_landmarks in multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Detect thumbs up gesture
                    thumb_pos = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_pos = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_pos = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring_pos = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky_pos = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                    thumb_tip = thumb_pos.y
                    index_tip = index_pos.y
                    middle_tip = middle_pos.y
                    ring_tip = ring_pos.y
                    pinky_tip = pinky_pos.y

                    # Check if thumb is above all other fingers along y-axis and other fingers are extended
                    if (thumb_tip < index_tip and 
                        thumb_tip < middle_tip and 
                        thumb_tip < ring_tip and 
                        thumb_tip < pinky_tip and
                        abs(index_tip - index_pos.y) < 0.02 and 
                        abs(middle_tip - middle_pos.y) < 0.02 and 
                        abs(ring_tip - ring_pos.y) < 0.02 and 
                        abs(pinky_tip - pinky_pos.y) < 0.02):
                        cv2.putText(frame, 'Thumbs Up!', (self.x - 50, self.y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Detect okay gesture
                    thumb_x, thumb_y = int(thumb_pos.x * frame.shape[1]), int(thumb_pos.y * frame.shape[0])
                    index_x, index_y = int(index_pos.x * frame.shape[1]), int(index_pos.y * frame.shape[0])
                    distance_thumb_index = ((thumb_x - index_x)**2 + (thumb_y - index_y)**2)**0.5

                    if (distance_thumb_index < 0.1 * frame.shape[1] and
                        abs(middle_tip - middle_pos.y) < 0.02 and 
                        abs(ring_tip - ring_pos.y) < 0.02 and 
                        abs(pinky_tip - pinky_pos.y) < 0.02):
                        cv2.putText(frame, 'Okay!', (self.x - 50, self.y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Create Sprite object for the thumb
thumb = Sprite('Thumb')

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

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
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Update position and show thumb sprite
                thumb_pos = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb.setx(thumb_pos.x * frame.shape[1])
                thumb.sety(thumb_pos.y * frame.shape[0])
                thumb.show()

                index_pos = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_pos = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_pos = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_pos = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                # Draw circles for other fingers
                index_x, index_y = int(index_pos.x * frame.shape[1]), int(index_pos.y * frame.shape[0])
                cv2.circle(frame, (index_x, index_y), 5, (255, 0, 0), -1)  # Blue dot for index finger
                middle_x, middle_y = int(middle_pos.x * frame.shape[1]), int(middle_pos.y * frame.shape[0])
                cv2.circle(frame, (middle_x, middle_y), 5, (0, 255, 0), -1)  # Green dot for middle finger
                ring_x, ring_y = int(ring_pos.x * frame.shape[1]), int(ring_pos.y * frame.shape[0])
                cv2.circle(frame, (ring_x, ring_y), 5, (0, 0, 255), -1)  # Red dot for ring finger
                pinky_x, pinky_y = int(pinky_pos.x * frame.shape[1]), int(pinky_pos.y * frame.shape[0])
                cv2.circle(frame, (pinky_x, pinky_y), 5, (255, 255, 0), -1)  # Yellow dot for pinky finger

        else:
            # Hide thumb sprite if no hand is detected
            thumb.hide()

        # Draw thumb sprite and detect gestures
        thumb.draw(frame, result.multi_hand_landmarks)

        # Display the frame
        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()