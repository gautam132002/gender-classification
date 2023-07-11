import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Set TensorFlow log level to suppress warnings
from keras.models import load_model
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Colors for printing
yellow = "\033[93m"
green = "\033[92m"
red = "\033[91m"
bold = "\033[1m"
end_color = "\033[0m"

print(yellow + bold + """
select an option
[1] input a video
[2] live feed from the camera
""" + end_color)

ch = input(green + bold + "$" + end_color)

if ch == "1":
    cam_index = input(green + bold + "enter the path to the video $" + end_color)
elif ch == "2":
    cam_index = int(input(green + bold + "enter your camera index ex 0, 1, 2, etc. $" + end_color))
else:
    print(red + bold + "ERR: invalid input, aborting..." + end_color)

np.set_printoptions(suppress=True)

model = load_model("gender_classification.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(cam_index)

while True:
    ret, image = camera.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_image = image[y:y + h, x:x + w]
        face_image = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_AREA)

        # Preprocess the face image
        face_image = np.asarray(face_image, dtype=np.float32).reshape(1, 224, 224, 3)
        face_image = (face_image / 127.5) - 1

        # Predict the gender using the model
        prediction = model.predict(face_image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Write the prediction over the box
        label = class_name[2:] + ": " + str(np.round(confidence_score * 100))[:-2] + "%"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Webcam Image", image)

    # Listen to the keyboard for presses
    keyboard_input = cv2.waitKey(1)

    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
