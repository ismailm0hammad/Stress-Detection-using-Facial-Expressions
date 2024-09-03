import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import cv2

# Load your pre-trained model
model_path = r"C:\Users\ismai\PycharmProjects\StressDetection\best_model_checkpointer.h5"
model = tf.keras.models.load_model(model_path, compile=False)
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

# Define the path to your validation data
validation_data_path = r'I:\images\validation'

# Image size and batch size
IMG_SIZE = (48, 48)
BATCH_SIZE = 32

# Create a dataset from the directory
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    validation_data_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',  # Use 'categorical' for multi-class classification
    shuffle=False  # Set to False to maintain order for accurate evaluation
)

# Get the class labels
class_names = validation_dataset.class_names

# Convert images to grayscale
def to_grayscale(image):
    return tf.image.rgb_to_grayscale(image)

# Normalize the dataset
def preprocess(image, label):
    image = to_grayscale(image)  # Convert to grayscale
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    return image, label

validation_dataset = validation_dataset.map(preprocess)

# Evaluate the model
y_true = np.concatenate([y for x, y in validation_dataset], axis=0)
y_pred = model.predict(validation_dataset)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_true, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Generate a detailed classification report
report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
print(report)

# Define function for image preprocessing
def preprocess_image(image_path, target_size=(48, 48)):
    """
    Preprocesses an image for emotion classification.
    Args:
        image_path: Path to the image file.
        target_size: Target size for resizing the image.
    Returns:
        A preprocessed image as a NumPy array.
    """
    # Read the image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image
    resized_image = cv2.resize(gray_image, dsize=target_size, interpolation=cv2.INTER_AREA)
    # Normalize pixel values
    normalized_image = resized_image / 255.0
    # Reshape to add a batch dimension (optional, depending on model)
    normalized_image = np.expand_dims(normalized_image, axis=-1)  # Add channel dimension
    return normalized_image

# Define function to predict emotion
def predict_emotion(model, image):
    """
    Predicts the emotion from an image using the model.
    Args:
        model: The loaded model.
        image: The preprocessed image.
    Returns:
        The predicted emotion class.
    """
    # Make prediction using your model
    predictions = model.predict(np.expand_dims(image, axis=0))  # Add batch dimension if needed
    predicted_class = np.argmax(predictions)
    # Replace with your class labels
    class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    return class_names[predicted_class]

# Example usage
image_paths = {
    "Angry": r"I:\images\validation\angry\65.jpg",
    "Disgust": r"I:\images\validation\disgust\533.jpg",
    "Fear": r"I:\images\validation\fear\21.jpg",
    "Happy": r"I:\images\validation\happy\30.jpg",
    "Neutral": r"I:\images\validation\neutral\44.jpg",
    "Sad": r"I:\images\validation\sad\20.jpg",
    "Surprise": r"I:\images\validation\surprise\51.jpg"
}

for emotion, image_path in image_paths.items():
    preprocessed_image = preprocess_image(image_path)
    predicted_emotion = predict_emotion(model, preprocessed_image)
    print(f"Image Path: {image_path}, Predicted emotion: {predicted_emotion}")
