from function import*
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, GlobalAveragePooling1D
from keras.callbacks import TensorBoard
import json

import numpy as np
import os

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num) + '.npy')
            
            if os.path.exists(npy_path):
                res = np.load(npy_path)
                window.append(res)
            else:
                print(f"Warning: Missing data file {npy_path}")
                window.append(np.zeros((63,)))
        sequences.append(window)
        labels.append(label_map[action])
        
X = np.array(sequences)
y = to_categorical(np.array(labels)).astype(int)

print(f"Data shape: X={X.shape}, y={y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y, random_state=42)

logs_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=logs_dir) 

model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Starting training...")
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback], validation_data=(X_test, y_test))
model.summary()

model_json = model.to_json()

with open('model.json', 'w') as f:
    f.write(model_json)

model.save('model5.h5')
print("Model saved")
