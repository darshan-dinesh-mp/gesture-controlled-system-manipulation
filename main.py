import cv2
import mediapipe as mp
import pyautogui
import os

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Screen dimensions for mouse movement
screen_width, screen_height = pyautogui.size()

# Capture video feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and process frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            # Determine hand type (Left or Right)
            hand_type = hand_handedness.classification[0].label  # 'Left' or 'Right'

            # Right Hand for Mouse Control
            if hand_type == 'Right':
                # Mouse movement using index fingertip
                index_finger = hand_landmarks.landmark[8]
                x = int(index_finger.x * screen_width)
                y = int(index_finger.y * screen_height)
                pyautogui.moveTo(x, y, duration=0.1)

                # Left-click: Pinch (Thumb and Index finger touch)
                thumb_tip = hand_landmarks.landmark[4]
                distance = ((thumb_tip.x - index_finger.x)**2 + (thumb_tip.y - index_finger.y)**2)**0.5
                if distance < 0.02:  # Adjust threshold as needed
                    pyautogui.click()

                # Right-click: Thumb and Middle finger touch
                middle_finger = hand_landmarks.landmark[12]
                distance = ((thumb_tip.x - middle_finger.x)**2 + (thumb_tip.y - middle_finger.y)**2)**0.5
                if distance < 0.02:  # Adjust threshold as needed
                    pyautogui.rightClick()

                # Click and drag: Hold pinch (Thumb and Index finger)
                if distance < 0.02:
                    pyautogui.mouseDown()
                else:
                    pyautogui.mouseUp()

            # Left Hand for System Control
            elif hand_type == 'Left':
                # Close current window (Close Palm = Fist Gesture)
                thumb_tip = hand_landmarks.landmark[4]
                thumb_base = hand_landmarks.landmark[3]

                is_fist = (
                    hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y and  # Index finger curled
                    hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and  # Middle finger curled
                    hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and  # Ring finger curled
                    hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y and  # Pinky finger curled
                    abs(thumb_tip.x - thumb_base.x) < 0.03  # Thumb close to palm
                )

                if is_fist:
                    pyautogui.hotkey('alt', 'f4')

                # Open palm does nothing (no action required here)

                # Open Chrome (Peace sign gesture)
                if (hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and  # Index finger extended
                        hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y and  # Middle finger extended
                        hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and  # Ring finger bent
                        hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y and  # Pinky finger bent
                        hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x):  # Thumb tucked in
                    os.system("start chrome")

            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Hand Gesture Control", frame)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
