from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the Data
data = pd.read_csv(r'C:\Users\James\Documents\ML datasets\forex4.csv')

# Step 2: Data Preprocessing
# Separate features and target (assuming the last column is the target)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: Scale the features
# Step 5: Initialize StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Step 6: Fit and transform the features to standardize them
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))


# Step 3: Train the KNN Model
model = KNeighborsClassifier(n_neighbors=2)  # You can choose the number of neighbors (k)
model.fit(X_train_scaled, y_train_scaled.ravel())

# Step 4: Make Predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # Use 'macro' for multiclass
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

new_data = np.array([[1.2727,1.2726,1.2598,1.2689]
                     ])


# Step 10: Scale the new input data using the same scaler as for the training data
new_input_data_scaled = scaler_X.transform(new_data)

# Step 11: Make predictions for the new input data
new_predictions_scaled = model.predict(new_input_data_scaled)

# Step 12: Inverse transform the predictions to get them back to the original scale
new_predictions = scaler_y.inverse_transform(new_predictions_scaled.reshape(-1, 1))


# Print the new predictions
print("New input data predictions:")
for i, pred in enumerate(new_predictions):
    print(f"Input data: {new_data[i]}, Predicted output: {pred[0]:.4f}")

