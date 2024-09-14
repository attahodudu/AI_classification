import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


#Step 1: Load your data>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
data = pd.read_csv(r'C:\Users\James\Documents\ML datasets\Bank Customer Churn.csv')

#step 2: data cleaning>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Drop rows with missing values (if any)
data = data.dropna()

# Check for duplicates and remove them
data = data.drop_duplicates()

# Step 3: Separate the features and target variable>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Assuming the last column is the target variable and the rest are features
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# step 4: Specify the categorical column>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
categorical_column = ["country", "gender", "active_member"]  # Replace with your actual column names

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

# Display the first few rows of the processed DataFrame
print(data.head())


#step 6: Split the data into features (X) and target (y)>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
X = data.drop("churn", axis=1)  # features
y = data["churn"]  # target

## step 7: Split the data into training and testing sets>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # step 8: Normalize or scale the features>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# # step 9: Define the logistic regression model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
model = LogisticRegression()
# # step 10: fit or train the model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
model.fit(X_train, y_train)

## step 11: make predictions>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
y_pred = model.predict(X_test)


## step 12: Calculate evaluation metrics>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')



# # step 13:Prepare your new data as a 2D array>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_data = np.array([[15634603, 614, "France", "Female", 42, 2, 0, 0, 1, 1, 1013.88],
                     [15634604, 616, "Spain", "Female", 42, 2, 0, 0, 0, 0, 10138.88],
                     [15634605, 616, "Germany", "male", 42, 2, 0, 0, 0, 1, 1013248.88],
                     [15634606, 716, "France", "male", 42, 2, 0, 0, 1, 0, 1048.88],
                     [15634607, 613, "Spain", "Female", 42, 2, 0, 0, 0, 0, 1348.88],
                     [15634603, 612, "France", "male", 32, 2, 0, 0, 1, 1, 101348.88],
                     [15634603, 61, "France", "male", 40, 3, 0, 0, 1, 1, 101348.88]])

# Convert new data to DataFrame
new_data_df = pd.DataFrame(new_data, columns=["customer_id", "credit_score", "country", "gender", "age", "tenure", "balance", "products_number", "credit_card", "active_member", "estimated_salary"])

# # step 14: Select the same categorical columns for encoding>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_categorical_data = new_data_df[categorical_column]

# # step 15: Encode the categorical columns>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_encoded_data = encoder.transform(new_categorical_data)

# # step 16: Drop original categorical columns and concatenate encoded columns>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_data_df = new_data_df.drop(columns=categorical_column)
new_data_df = pd.concat([new_data_df.reset_index(drop=True), pd.DataFrame(new_encoded_data, columns=encoder.get_feature_names_out(categorical_column))], axis=1)

# Ensure the columns are in the same order as the training data
new_data_df = new_data_df[model.feature_names_in_]

# # step 17: Scale the features>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_data_scaled = scaler.transform(new_data_df)

# # step 18: Make predictions>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_prediction = model.predict(new_data_scaled)
new_prediction_proba = model.predict_proba(new_data_scaled)

#print("Prediction:", new_prediction)

print("New input data predictions:")
for i, pred in enumerate(new_prediction):
    print(f"Input data: {new_data[i]}, Predicted output: {pred:.2f}")

print("Prediction Probability:", new_prediction_proba)

