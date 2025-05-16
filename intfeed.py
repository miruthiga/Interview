import cv2
import numpy as np
import queue
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from transformers import pipeline
from deepface import DeepFace  

# Load OpenCV's Pretrained Haar Cascade for Eye Detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load NLP Model
nlp_feedback = pipeline("text-generation", model="t5-small")

# Load Vosk Model (Ensure correct extracted path)
vosk_model = Model( r"C:\Users\hp\Desktop\model\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15")  
rec = KaldiRecognizer(vosk_model, 16000)
rec.SetWords(True)
audio_queue = queue.Queue()

# Audio Callback Function
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(bytes(indata))

# Function to Detect Eye Position
def detect_eye_direction(eyes, frame_width):
    if len(eyes) == 2:
        left_eye_x, _, _, _ = eyes[0]  # X-coordinate of left eye
        right_eye_x, _, _, _ = eyes[1]  # X-coordinate of right eye

        mid_x = frame_width // 2

        if left_eye_x > mid_x and right_eye_x > mid_x:
            return "Looking Right"
        elif left_eye_x < mid_x and right_eye_x < mid_x:
            return "Looking Left"
        else:
            return "Looking Center"
    return "Unknown"

# Start Audio Recording
with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                        channels=1, callback=audio_callback):
    cap = cv2.VideoCapture(0)  # Open camera

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        gaze_direction = detect_eye_direction(eyes, frame.shape[1])

        # Process Facial Emotion using DeepFace
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion_text = analysis[0]['dominant_emotion']
        except:
            emotion_text = "Neutral"

        # Process Audio Speech Recognition
        while not audio_queue.empty():
            data = audio_queue.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text_response = result.get("text", "")
                print(f"Recognized Speech: {text_response}")

                # Generate AI Feedback
                feedback = nlp_feedback(f"Provide interview feedback for: {text_response}", max_length=50)[0]['generated_text']
                print(f"AI Feedback: {feedback}")

            # Clear queue to avoid overflow
            audio_queue.queue.clear()

        # Display Output on Screen
        cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {emotion_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Interview Feedback System", frame)

        # Exit on 'q' Key Press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
