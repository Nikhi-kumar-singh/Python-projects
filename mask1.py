import cv2
import matplotlib.pyplot as plt
import numpy as np


# img = cv2.imread("maa.jpg")
# img1 = cv2.imread("family1.jpg")

# plt.imshow(img)
# plt.show()

# cv2.imshow("maa image",img)
# cv2.waitKey(1000)
# cv2.destroyAllWindows()

data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# print(data.detectMultiScale(img))
# print(data.detectMultiScale(img1))

# cv2.rectangle(img,(x,y),(w,h),(b,g,r),border_thickness)

# while True:
#     faces = data.detectMultiScale(img1)
#     for x,y,w,h in faces:
#         cv2.rectangle(img1, (x, y), (x + w, y + h), (255, 0, 255), 2)
#     cv2.imshow("images",img1)
#     if cv2.waitKey(2)==27:
#          break
# cv2.destroyAllWindows()


face_data_nomask = []
face_data_with_mask = []

# if we want to
capture = cv2.VideoCapture(0)
while True:
    flag, img = capture.read()
    if flag:
        faces = data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50, 50))
            if len(face_data_nomask) < 201:
                print(len(face_data_nomask))
                face_data_nomask.append(face)
        cv2.imshow("images", img)
        if cv2.waitKey(2) == 27 or len(face_data_nomask) >= 200:
            break

np.save('no_mask.npy', face_data_nomask)

capture = cv2.VideoCapture(0)
while True:
    flag, img = capture.read()
    if flag:
        faces = data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50, 50))
            if len(face_data_with_mask) < 201:
                print(len(face_data_with_mask))
                face_data_with_mask.append(face)
        cv2.imshow("images", img)
        if cv2.waitKey(2) == 27 or len(face_data_with_mask) >= 200:
            break

np.save("with_mask.npy", face_data_with_mask)

