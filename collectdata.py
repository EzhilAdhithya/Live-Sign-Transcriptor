import cv2
import os
import streamlit as st
import numpy as np
from function import mediapipe_detection, draw_landmarks, extract_keypoints
import mediapipe as mp
from PIL import Image

cap = cv2.VideoCapture(0)

directory = os.path.join(os.path.dirname(__file__), 'Image')

st.title("Hand Sign Detection")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read image as OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    st.image(frame, channels="BGR", caption="Uploaded Image")

    # Hand detection
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as hands:
        image, results = mediapipe_detection(frame, hands)
        draw_landmarks(image, results)
        st.image(image, channels="BGR", caption="Hand Landmarks")

        if results.multi_hand_landmarks:
            st.success("Hand detected!")
            keypoints = extract_keypoints(results)
            st.write("Keypoints:", keypoints)
        else:
            st.warning("No hand detected in the image.")

while True:
    _, frame = cap.read()
    
    count = {
        'a': len(os.listdir(os.path.join(directory, 'A'))),
        'b': len(os.listdir(os.path.join(directory, 'B'))),
        'c': len(os.listdir(os.path.join(directory, 'C'))),
    }


    row = frame.shape[1]
    col = frame.shape[0]

    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)

    cv2.imshow('data', frame)
    cv2.imshow('ROI', frame[40:400, 0:300])

    frame = frame[40:400, 0:300]

    interrupt = cv2.waitKey(10)

    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(os.path.join(directory, 'A', str(count['a']) + '.png'), frame)
        
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(os.path.join(directory, 'B', str(count['b']) + '.png'), frame)
        
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(os.path.join(directory, 'C', str(count['c']) + '.png'), frame)

cap.release()
cv2.destroyAllWindows()


