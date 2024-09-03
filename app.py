from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your pre-trained model
model_path = r"I:\StressDetection\best_model_checkpointer.h5"
model = tf.keras.models.load_model(model_path, compile=False)
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

# Class names for prediction
class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# Function to preprocess image
def preprocess_image(image):
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if image is in color
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image  # Image is already in grayscale
    resized_image = cv2.resize(gray_image, (48, 48))
    normalized_image = resized_image.astype(np.float32) / 255.0
    expanded_image = np.expand_dims(normalized_image, axis=-1)
    return np.expand_dims(expanded_image, axis=0)


# Video streaming generator function
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            processed_image = preprocess_image(roi_gray)
            prediction = model.predict(processed_image)
            predicted_class_index = np.argmax(prediction[0])
            predicted_class_name = class_names[predicted_class_index]
            if predicted_class_name in ["happy", "neutral",]:
                is_stress = "Happy :)"
                box_color = (0,255,0)
            else:
                is_stress = "Stress"
                box_color = (0,0,255)

            # Draw rectangle around the face and put the prediction text
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(frame, is_stress, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
