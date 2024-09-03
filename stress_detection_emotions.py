



train_path = r'I:\DLProject\images\stress_emotions\train'
val_path = r'I:\DLProject\images\stress_emotions\validation'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_generator = ImageDataGenerator(
    rescale=1. / 255,  # Keep this for normalization
    rotation_range=15,  # Rotate images randomly up to 15 degrees
    width_shift_range=0.1,  # Shift images horizontally up to 10% of the width
    height_shift_range=0.1,  # Shift images vertically up to 10% of the height
    horizontal_flip=True,  # Flip images horizontally randomly
    zoom_range=0.1  # Zoom in/out up to 10% randomly
)

train_ds = image_generator.flow_from_directory(
    train_path,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_ds = image_generator.flow_from_directory(
    val_path,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical'
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define model
model = Sequential()

# Input layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))

# Convolutional layers with regularization
# model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer='l2'))
model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.2))

# Flatten layer
model.add(Flatten())

# Dense layers with regularization
model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.3))

# Output layer
model.add(Dense(5, activation='softmax'))  # 5 for stress_angry, stress_sad, stress_fear, no_stress_happy, and no_stress_neutral

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
# ... (use the generated train_generator and validation_generator here)
history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    verbose=1  # Set to 2 for more detailed training progress
)

import csv

# Specify relevant metrics you want to save
relevant_metrics = ['loss', 'accuracy', 'val_loss', 'val_accuracy']

# Open CSV file in write mode
with open('history_relevant_metrics.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header row with metric names
    writer.writerow(relevant_metrics)

    # Iterate through each epoch in the history
    for epoch in range(len(history.history)):
        # Create a row with relevant metrics for the current epoch
        epoch_data = [history.history[metric][epoch] for metric in relevant_metrics]
        writer.writerow(epoch_data)

# save the model
model.save('stress_detection_emotions.keras')
