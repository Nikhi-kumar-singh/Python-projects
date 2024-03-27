import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

mask = np.load('with_mask.npy')
no_mask = np.load('no_mask.npy')

# Reshape the images
mask = mask.reshape(20, -1)  # -1 means automatically infer the size
no_mask = no_mask.reshape(20, -1)

# Combine the data
x = np.vstack((mask, no_mask))

# Create labels
labels = np.zeros(x.shape[0])
labels[20:] = 1.0

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2)

# Apply PCA
pca = PCA(n_components=3)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Create and train the SVM model
model = SVC()
model.fit(x_train_pca, y_train)


names = {0: "no mask", 1: "with mask"}
data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
capture = cv2.VideoCapture(0)
data1 = []
while True:
    flag, img = capture.read()
    if flag:
        faces = data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50, 50))
            face = face.reshape(1, -1)  # Reshape to match the model input format
            face = pca.transform(face)  # Comment out PCA transformation for inference
            pred = model.predict(face)[0]
            n = names[int(pred)]
            print(n)
        cv2.imshow("images", img)
        if cv2.waitKey(2) == 27:
            break

# Delay before releasing resources
cv2.waitKey(0)
capture.release()
cv2.destroyAllWindows()
