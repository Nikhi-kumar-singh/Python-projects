import cv2
import numpy as np
import matplotlib.pyplot as plt

data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_data_nomask = []
face_data_with_mask = []

# capturing the image without mask
capture = cv2.VideoCapture(0)
while True:
    flag, img = capture.read()
    if flag:
        faces = data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50, 50))
            if len(face_data_nomask) < 20:
                print(len(face_data_nomask))
                face_data_nomask.append(face)
        cv2.imshow("images", img)
        if cv2.waitKey(2) == 27 or len(face_data_nomask) >= 20:
            break
np.save('no_mask.npy', face_data_nomask)

# capturing the images with mask
capture = cv2.VideoCapture(0)
while True:
    flag, img = capture.read()
    if flag:
        faces = data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50, 50))
            if len(face_data_with_mask) < 20:
                print(len(face_data_with_mask))
                face_data_with_mask.append(face)
        cv2.imshow("images", img)
        if cv2.waitKey(2) == 27 or len(face_data_with_mask) >= 20:
            break
np.save("with_mask.npy", face_data_with_mask)

capture.release()
cv2.destroyAllWindows()

# Corrected code for displaying the image using Matplotlib
plt.imshow(face_data_with_mask[0])
plt.show()
