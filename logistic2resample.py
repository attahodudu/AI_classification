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
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report


#Step 1: Load your data>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
data = pd.read_csv(r'C:\Users\James\Documents\ML datasets\covid.csv')

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
categorical_column = ["USMER","MEDICAL_UNIT","SEX","PATIENT_TYPE","DAY","MONTH","YEAR","INTUBED","PNEUMONIA","PREGNANT","DIABETES","COPD","ASTHMA","INMSUPR","HYPERTENSION","OTHER_DISEASE","CARDIOVASCULAR","OBESITY","RENAL_CHRONIC","TOBACCO","CLASIFFICATION_FINAL"]  # Replace with your actual column names

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
X = data.drop("ICU", axis=1)  # features
y = data["ICU"]  # target

## step 7: Split the data into training and testing sets>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# # step 8: Normalize or scale the features>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Step 9: Apply SMOTE to balance the classes>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# # step 10: Define the logistic regression model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
model = LogisticRegression()
#step 11: fit or train the model on the resampled data>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Assuming X_train and y_train are pandas Series or DataFrames
y_train = y_train.fillna(y_train.mean())  # Impute with mean
y_train = y_train.astype(float)  # Ensure numeric data type

# Handle NaN in X_train as before
X_train_imputed = X_train.fillna(X_train.mean())



model.fit(X_train_resampled, y_train_resampled)
feature_order = X_train.columns.tolist()

# step 12: make predictions on the test set>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

y_pred_resampled = model.predict(X_test_scaled)
y_pred_proba_resampled = model.predict_proba(X_test_scaled)[:, 1]

# step 13:Calculate and print the new evaluation metrics>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

accuracy_resampled = accuracy_score(y_test, y_pred_resampled)
precision_resampled = precision_score(y_test, y_pred_resampled)
recall_resampled = recall_score(y_test, y_pred_resampled)
f1_resampled = f1_score(y_test, y_pred_resampled)

print(f'Accuracy (Resampled): {accuracy_resampled:.2f}')
print(f'Precision (Resampled): {precision_resampled:.2f}')
print(f'Recall (Resampled): {recall_resampled:.2f}')
print(f'F1-score (Resampled): {f1_resampled:.2f}')

# Detailed classification report
print(classification_report(y_test, y_pred_resampled))

#Step 14: Plot ROC curve and calculate AUC for resampled model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
fpr_resampled, tpr_resampled, _ = roc_curve(y_test, y_pred_proba_resampled)
roc_auc_resampled = auc(fpr_resampled, tpr_resampled)

plt.figure(figsize=(8, 6))
plt.plot(fpr_resampled, tpr_resampled, color='blue', label=f'ROC curve (area = {roc_auc_resampled:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (Resampled)')
plt.legend()
plt.grid(True)
plt.show()

# step 15:Prepare your new data as a 2D array>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_data = np.array([[2,1,1,1,1,1,220,3,1,65,2,0,0,0,0,1,0,0,0,0,0,3],
                     [2,1,2,1,2,1,220,3,1,72,3,0,0,0,0,1,0,0,1,1,0,5]
                     ])
# Convert new data to DataFrame
new_data_df = pd.DataFrame(new_data, columns=["USMER","MEDICAL_UNIT","SEX","PATIENT_TYPE","DAY","MONTH","YEAR","INTUBED","PNEUMONIA","PREGNANT","DIABETES","COPD","ASTHMA","INMSUPR","HYPERTENSION","OTHER_DISEASE","CARDIOVASCULAR","OBESITY","RENAL_CHRONIC","TOBACCO","CLASIFFICATION_FINAL"])

#step 16: Select the same categorical columns for encoding>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_categorical_data = new_data_df[categorical_column]

#step 17: Encode the categorical columns>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_encoded_data = encoder.transform(new_categorical_data)

#step 18: Drop original categorical columns and concatenate encoded columns>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_data_df = new_data_df.drop(columns=categorical_column)
new_data_df = pd.concat([new_data_df.reset_index(drop=True), pd.DataFrame(new_encoded_data, columns=encoder.get_feature_names_out(categorical_column))], axis=1)

#Step 19: Ensure the column order matches the training data>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_data_df = new_data_df[feature_order]

#step 20: Scale the features>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_data_scaled = scaler.transform(new_data_df)

#step 21: Make predictions and print>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
new_prediction = model.predict(new_data_scaled)
new_prediction_proba = model.predict_proba(new_data_scaled)

#print("Prediction:", new_prediction)

print("New input data predictions:")
for i, pred in enumerate(new_prediction):
    print(f"Input data: {new_data[i]}, Predicted output: {pred:.2f}")

print("Prediction Probability:", new_prediction_proba)
