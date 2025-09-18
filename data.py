from function import *
import cv2
from function import draw_landmarks
import os

for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok = True)

with mp_hands.Hands(
    model_complexity = 0,
    min_detection_confidence = 0.3,
    min_tracking_confidence = 0.3
)as hands:

    for action in actions:
        action_dir = os.path.join('Image', action)
        if not os.path.exists(action_dir):
            print(f"Directory not found: {action_dir}")
            continue
        image_files = [f for f in os.listdir(action_dir) if f.endswith('.png')]
        for image_file in image_files:
            img_path = os.path.join(action_dir, image_file)
            print(f"Looking for: {img_path}")
            frame = cv2.imread(img_path)
            
            if frame is None:
                print('warning: Image not found')
                continue
            image, results = mediapipe_detection(frame, hands)
            
            if results.multi_hand_landmarks:
                print(f'Hands detected for {action} for sequence {sequence} frame {image_file}')
            else:
                print(f'Hands not found {action} for sequence {sequence} frame {image_file}')
            
            draw_landmarks(image, results)
            
            message = f'collecting frames for {action}, video {sequence}'
            cv2.putText(image, message,(15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            cv2.imshow('openCV feed', image )
            
            keypoints = extract_keypoints(results) 
            npy_path = os.path.join(DATA_PATH, action, str(sequence), image_file)
            np.save(npy_path, keypoints)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
    cv2.destroyAllWindows()
            