import cv2
# import numpy
import numpy as np
import tensorflow as tf
# from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

# Load your pre-trained model
model_path = r"C:\Users\ismai\PycharmProjects\StressDetection\best_model_checkpointer.h5"
model = tf.keras.models.load_model(model_path, compile=False)  # Replace with your model path
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

# Set video capture object
cap = cv2.VideoCapture(0)

frame_count = 0  # Initialize frame count

while True:
    # Capture a frame
    ret, frame = cap.read()
    print(ret)
    # Check if captured successfully
    if not ret:
        print("Error capturing frame")
        break

    # Preprocess the frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Save the grayscale image to a file
    cv2.imwrite(r"C:\Users\ismai\PycharmProjects\StressDetection\Captures\orig.png", image)
    print("Grayscale image saved!")

    imagePath = r"C:\Users\ismai\PycharmProjects\StressDetection\Captures\orig.png"
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    print("[INFO] Found {0} Faces.".format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        print("[INFO] Object found. Saving locally.")
        cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)

    status = cv2.imwrite(r'C:\Users\ismai\PycharmProjects\StressDetection\Captures\faces_detected.jpg', roi_color)
    print("[INFO] Image faces_detected.jpg written to filesystem: ", status)
    roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

    print(roi_color.shape)
    roi_color = cv2.resize(roi_color, (48, 48))  # Adjust resize based on your model's input size
    # img_array = image.img_to_array(img)
    roi_color = roi_color.astype(np.float64)
    roi_color /= 255.0
    roi_color = np.expand_dims(roi_color, axis=0)

    # Make prediction
    predictions = model.predict(roi_color)
    predicted_class_index = np.argmax(predictions[0])
    class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    # Display prediction on the frame
    predicted_class_name = class_names[predicted_class_index]
    print("You are :", predicted_class_name)
    print("Image saved!")
    break

# Release capture object    `
cap.release()
cv2.destroyAllWindows()
