from function import*
from keras.models import model_from_json
from keras.utils import to_categorical
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import json

import os

# Load model architecture from JSON
with open('model.json', 'r') as json_file:
    model_json = json_file.read().strip()

model = model_from_json(model_json)
model.load_weights('model5.h5')

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

colors = []
for i in range(0,20):
    colors.append((245, 117, 16))
    
def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        if prob > threshold:
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=4, 
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        
        cropped_frame = frame[40:400, 0:300]
        
        frame = cv2.rectangle(frame, (0,0), (300, 400), (255,255,255), 2)
        image, results = mediapipe_detection(cropped_frame, hands)
        
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        try:
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                print(f"Raw probabilities: {res}, Predicted class: {actions[np.argmax(res)]}")
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(str(res[np.argmax(res)]*100))
                    else:
                        sentence.append(actions[np.argmax(res)])
                        accuracy.append(str(res[np.argmax(res)]*100))
            if len(sentence)>1:
                sentence = sentence[-1:]
                accuracy = accuracy[-1:]
        except Exception as e:
            pass
        
        cv2.rectangle(frame, (0,0), (300,400), (245, 117, 16), 2)
        cv2.putText(frame, 'output: ' + ' '.join(sentence) + ' '.join(accuracy), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2, cv2.LINE_AA)
        cv2.imshow('open cv feed', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()