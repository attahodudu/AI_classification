##this code has failed the test! although the metrics are good but the dataset doesn't work for logistic regression
##even if categories are included, it still can't work with these complex numbers

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the dataset
data = pd.read_csv(r'C:\Users\James\Documents\ML datasets\forex4.csv')

# Step 2: Inspect and clean the dataset
print(data.head())
data.isnull().sum()

# Step 3: Separate features and target
X = data.drop('direction', axis=1)  # Replace 'target' with your actual target column name
y = data['direction']

# Step 4: Encode the target variable if it is categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 5: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train the logistic regression model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions and evaluate
y_pred = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')


#Step 14: Plot ROC curve and calculate AUC for resampled model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (Resampled)')
plt.legend()
plt.grid(True)
plt.show()


#new data input prompt
def get_new_data():
    # Prompt the user to enter the data
    user_input = input("Enter prev_ath,prev_atl,open_price e.g 1.2443,1.2459,1.2378): ")
    
    # Split the input string into a list of strings
    input_list = user_input.split(',')
    
    # Convert the list of strings to a list of floats
    input_floats = [float(i) for i in input_list]
    
    # Convert the list of floats to a numpy array and reshape it
    new_data = np.array([input_floats])
    
    return new_data

# Call the function to get new data
new_data = get_new_data()

# Print the new data to verify
print("New data entered:", new_data)


# # step 18: Make predictions>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_prediction = model.predict(new_data)
new_prediction_proba = model.predict_proba(new_data)

#print("Prediction:", new_prediction)

print("New input data predictions:")
for i, pred in enumerate(new_prediction):
    print(f"Input data: {new_data[i]}, Predicted output: {pred:.2f}")

print("Prediction Probability:", new_prediction_proba)

