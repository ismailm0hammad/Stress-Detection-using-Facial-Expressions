import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

# Load your pre-trained model
model_path = r"C:\Users\ismai\PycharmProjects\StressDetection\best_model_checkpointer.h5"
model = tf.keras.models.load_model(model_path, compile=False)
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

model.load_weights(model_path)

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

rectangle_bgr = (255, 255, 255)
img = np.zeros((500, 500))
txt = "HI Dude"
(txt_wid, txt_hgt) = cv2.getTextSize(txt, font, font_scale, thickness=1)[0]
txt_offset_x = 10
txt_offset_y = img.shape[0] - 25
box_cords = ((txt_offset_x, txt_offset_y), (txt_offset_x + txt_wid + 2, txt_offset_y - txt_hgt - 2))
cv2.rectangle(img, box_cords[0], box_cords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, txt, (txt_offset_x, txt_offset_y), font, font_scale, color=(0, 0, 0), thickness=1)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot Open webcam")

while True:
    ret, frm = cap.read()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    image = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(image, 1.3, 4,minSize=(30,30))
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = image[y:y + h, x:x + w]
        # facess = faceCascade.detectMultiScale(roi_gray)
        # if len(facess) == 0:
        #     print("Face not detected")
        # else:
        #     for (ex, wy, wu, hi) in facess:
        #         face_roi = roi_color[wy:wy + hi, ex:ex + wu]
    # roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, (48, 48))  # Adjust resize based on your model's input size
    # img_array = image.img_to_array(img)
    roi_gray = roi_gray.astype(np.float64)
    roi_gray /= 255.0
    roi_gray = np.expand_dims(roi_gray, axis=0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    print(roi_gray.shape)

    prediction = model.predict(roi_gray)

    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN

    predicted_class_index = np.argmax(prediction[0])
    class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    # Display prediction on the frame
    predicted_class_name = class_names[predicted_class_index]
    x1, y1, w1, h1 = 0, 0, 175, 175
    cv2.rectangle(frm, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
    cv2.putText(frm, predicted_class_name, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)
    cv2.putText(frm, predicted_class_name, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
    cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 0, 255))
    cv2.imshow("Face Emotion Recognition", frm)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
