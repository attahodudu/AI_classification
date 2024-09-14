from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load the dataset
data = pd.read_csv(r'C:\Users\James\Documents\ML datasets\covid.csv', encoding='latin1')

# Step 2: Inspect and clean the dataset
print(data.head())
print(data.info())
print(data.isnull().sum())
data = data.fillna(0)  # Fill NaNs with 0

# Step 3: Separate features and target
X = data.drop('ICU', axis=1)  # Replace 'target' with your actual target column name
y = data['ICU']

# Verify shapes
print(X.shape)
print(y.shape)
# Step 4: Encode categorical variables if any
# If you have categorical features, you need to encode them. Here's an example:
# label_encoder = LabelEncoder()
# X['categorical_column'] = label_encoder.fit_transform(X['categorical_column'])

imputer = SimpleImputer(strategy='mean')  # You can also use 'median', 'most_frequent', or 'constant'
X = imputer.fit_transform(X)

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# If you need to standardize the features, you can do it here
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Step 6: Initialize and train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 7: Make predictions and evaluate
y_pred = model.predict(X_test)

#metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')


# Define new input data (only feature values, no target column)
new_data = np.array([[2,1,2,2,3,1,2020,1,2,55,3,1,0,0,0,0,0,0,0,0,0,3]])

# Apply the same scaling as used for the training data
new_data_scaled = scaler.transform(new_data)

# Make predictions with the new data
new_predictions = model.predict(new_data_scaled)

print("New data prediction:", new_predictions)
