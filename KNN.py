
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the Data
data = pd.read_csv(r'C:\Users\James\Documents\ML datasets\gbpusd2.csv')

# Step 2: Data Preprocessing
# Separate features and target (assuming the last column is the target)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# step 4: Specify the categorical column>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
categorical_column = ["week", "day"]  # Replace with your actual column names

# Check if the categorical columns exist in the DataFrame
for col in categorical_column:
    if col not in data.columns:
        raise ValueError(f"Column '{col}' does not exist in the DataFrame")

# step 5: One-hot encode the categorical column>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_data = encoder.fit_transform(data[categorical_column])

# Create the DataFrame from encoded data with appropriate column names
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_column))

# Combine the original data with the encoded columns (excluding the original categorical column)
data = data.drop(columns=categorical_column)
data = pd.concat([data, encoded_df], axis=1)


#step 6: Split the data into features (X) and target (y)>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
X = data.drop("direction", axis=1)  # features
y = data["direction"]  # target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Train the KNN Model
model = KNeighborsClassifier(n_neighbors=1)  # You can choose the number of neighbors (k)
model.fit(X_train, y_train)

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




# # step 13:Prepare your new data as a 2D array>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_data = np.array([[35,1,1.3229,1.3086,1.3208]
                     ])

new_data_df = pd.DataFrame(new_data, columns=["week","day","prev_ath","prev_atl","current_open"])

# # step 14: Select the same categorical columns for encoding>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_categorical_data = new_data_df[categorical_column]

# # step 15: Encode the categorical columns>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_encoded_data = encoder.transform(new_categorical_data)

# # step 16: Drop original categorical columns and concatenate encoded columns>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_data_df = new_data_df.drop(columns=categorical_column)
new_data_df = pd.concat([new_data_df.reset_index(drop=True), pd.DataFrame(new_encoded_data, columns=encoder.get_feature_names_out(categorical_column))], axis=1)

# # step 17: Scale the features>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_data_scaled = scaler.transform(new_data_df)

# # step 18: Make predictions>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_prediction = model.predict(new_data_scaled)
new_prediction_proba = model.predict_proba(new_data_scaled)


print("New input data predictions:")
for i, pred in enumerate(new_prediction):
    print(f"Input data: {new_data[i]}, Predicted output: {pred:.2f}")

print("Prediction Probability:", new_prediction_proba)


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

