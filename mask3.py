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

# Make predictions
predictions = model.predict(x_test_pca)
# print(predictions)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
