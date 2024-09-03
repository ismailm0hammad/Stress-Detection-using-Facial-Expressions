import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorboard.program import TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

train_path = r'I:\DLProject\images\images\train'
val_path = r'I:\DLProject\images\images\validation'
MODEL_PATH = r'C:\Users\Sree\PycharmProjects\StressDetection\best_model_checkpointer.h5'

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
    batch_size=64,
    class_mode='categorical',
    subset='training'
)

val_ds = image_generator.flow_from_directory(
    val_path,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical'
)

# Define input shape
input_shape = (48, 48, 1)  # Adjust based on your image size
num_features = 64
num_labels = 7
batch_size = 64
epochs = 100
width, height = 48, 48

model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1),
                 kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2 * 2 * 2 * num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2 * 2 * num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2 * num_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

retrain_model = load_model('best_model_checkpointer.h5')
# result = retrain_model.evaluate(val_ds,steps=16)
# print('Val loss and accuracy = ',result)
# tensorboard = TensorBoard()

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
1
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')

checkpointer = ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True)

history = retrain_model.fit(train_ds,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=val_ds,
                            callbacks=[lr_reducer, early_stopper, checkpointer])

# Specify relevant metrics you want to save
relevant_metrics = ['loss', 'accuracy', 'val_loss', 'val_accuracy']

# Open CSV file in write mode
with open('history_relevant_metrics_medium.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header row with metric names
    writer.writerow(relevant_metrics)

    # Iterate through each epoch in the history
    for epoch in range(len(history.history)):
        # Create a row with relevant metrics for the current epoch
        epoch_data = [history.history[metric][epoch] for metric in relevant_metrics]
        writer.writerow(epoch_data)

# save the model
retrain_model.save('stress_detection_medium.keras')
# Saving the  model to  use it later on
fer_json = retrain_model.to_json()
with open("Stress_Model_medium.json", "w") as json_file:
    json_file.write(fer_json)
retrain_model.save_weights("Stress_Model_medium.h5")
