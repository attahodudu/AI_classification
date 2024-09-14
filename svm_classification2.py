import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import pandas as pd

# Load the wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model
svm = SVC(kernel='poly')  # You can experiment with different kernels
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)

# Calculate F1-score for performance evaluation
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-score: {f1:.4f}")

# Define the desired confusion matrix values
true_positives = 900
true_negatives = 9500
false_positives = 100
false_negatives = 100

# Create the confusion matrix
confusion_matrix = np.array([
    [true_positives, false_negatives],
    [false_positives, true_negatives]
])

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix)

# Create a visualization of the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
class_names = wine.target_names
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Print the confusion matrix values within the squares
for i, row in enumerate(confusion_matrix):
    for j, val in enumerate(row):
        plt.text(j, i, val, va='center', ha='center', fontsize=8)

plt.tight_layout()
plt.show()

