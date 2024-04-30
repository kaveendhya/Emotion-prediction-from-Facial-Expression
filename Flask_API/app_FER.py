import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import tempfile
import os

app = Flask(__name__)

# Initialize the emotion labels
emotion_labels = ['HAPPY', 'SAD', 'NEUTRAL', 'FEAR', 'DISGUST', 'ANGRY']

# Define model paths for both male and female models (replace with your actual paths)
male_model_path = 'C:/Users/Chamodhi/Untitled Folder 5/API/best_model_cifar10_maletrtest.hdf5'
female_model_path = 'C:/Users/Chamodhi/Untitled Folder 5/API/best_model_cifar10_femaleconfu.hdf5'

# Load emotion prediction models based on gender
def load_emotion_model(gender):
    if gender.lower() == 'male':
        return tf.keras.models.load_model(male_model_path)
    elif gender.lower() == 'female':
        return tf.keras.models.load_model(female_model_path)
    else:
        raise ValueError('Invalid gender specified')

# Function to preprocess a single frame
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (48, 48))
    normalized_frame = resized_frame / 255.0
    preprocessed_frame = np.expand_dims(normalized_frame, axis=-1)
    return preprocessed_frame

# Function to make predictions on a single frame
def predict_emotion(frame, emotion_model):
    preprocessed_frame = preprocess_frame(frame)
    emotions = emotion_model.predict(np.array([preprocessed_frame]))
    return emotions[0]

@app.route('/process_video', methods=['POST'])
def process_video():
    try:
        # Initialize the face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Check if a video file and gender are included in the request
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'})
        if 'gender' not in request.form:
            return jsonify({'error': 'Gender not provided'})

        video_file = request.files['video']
        gender = request.form['gender']

        # Save the uploaded video to a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_video_path = os.path.join(temp_dir, 'uploaded_video.mp4')
        video_file.save(temp_video_path)

        # Load the appropriate emotion prediction model based on gender
        emotion_model = load_emotion_model(gender)

        # Initialize an empty list to store emotion percentages for each frame
        frame_emotions = []

        # Read the video frames using OpenCV
        cap = cv2.VideoCapture(temp_video_path)
        
# #         # Face extraction rate
#         face_extraction_rate = 3  # Extract faces every 3 frames (adjust as needed)
#         frame_counter = 0

#         Get the total number of frames and frames per second (fps)

        
        # Get the total number of frames and frames per second (fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Calculate the video duration in seconds
        video_duration = total_frames / fps

        # Determine the frame extraction rate based on video duration
        if video_duration <= 15:
            face_extraction_rate = 10
        elif video_duration <= 30:
            face_extraction_rate = 20
        else:
            face_extraction_rate = 25

        frame_counter = 0


        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1

            if frame_counter % face_extraction_rate == 0:
                # Detect faces in the frame
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                face_rectangles = [(x, y, x + w, y + h) for (x, y, w, h) in faces]

                for (x, y, x2, y2) in face_rectangles:
                    face = frame[y:y2, x:x2]

                    # Make predictions for the current face
                    emotions = predict_emotion(face, emotion_model)

                    # Append the emotion percentages to the list
                    frame_emotions.append(emotions)

        # Release the video capture object
        cap.release()

        # Calculate the average emotion percentages over all frames
        total_emotions = np.sum(frame_emotions, axis=0)
        total_sum = np.sum(total_emotions)
        average_emotions = (total_emotions / total_sum) *100

        # Return the average emotion percentages as JSON response
        result = {label: percentage.item() for label, percentage in zip(emotion_labels, average_emotions)}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

