import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

ACTIONS = ['hello', 'thanks', 'yes', 'no', 'iloveyou']
SEQUENCE_LENGTH = 30
THRESHOLD = 0.7
SMOOTHING_WINDOW = 10
SMOOTHING_AGREE = 5

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

@st.cache_resource
def get_model():
    return load_model('sign_model.h5')

model = get_model()

def extract_landmarks(results):
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            handedness = results.multi_handedness[idx].classification[0].label
            if handedness == 'Left':
                lh = coords
            else:
                rh = coords
    return np.concatenate([lh, rh])

class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5,
                                     max_num_hands=2)
        self.sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.predictions = deque(maxlen=SMOOTHING_WINDOW)
        self.sentence = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image = cv2.flip(img, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        keypoints = extract_landmarks(results)
        self.sequence.append(keypoints)

        if len(self.sequence) == SEQUENCE_LENGTH:
            res = model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            pred_idx = int(np.argmax(res))
            confidence = float(res[pred_idx])
            self.predictions.append(pred_idx)

            if confidence > THRESHOLD:
                if self.predictions.count(pred_idx) >= SMOOTHING_AGREE:
                    word = ACTIONS[pred_idx]
                    if len(self.sentence) == 0 or self.sentence[-1] != word:
                        self.sentence.append(word)
                    if len(self.sentence) > 5:
                        self.sentence = self.sentence[-5:]

            cv2.putText(image, f'{ACTIONS[pred_idx]} ({confidence:.2f})',
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(self.sentence), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

st.title("Real-Time Sign Language Recognition")
st.write("Allow camera access and perform a trained sign.")

webrtc_streamer(key="sign-recognition", video_processor_factory=SignProcessor)