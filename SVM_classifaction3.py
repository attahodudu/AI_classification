from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from sklearn import svm  # Import SVC class from scikit-learn
from sklearn.preprocessing import StandardScaler  # For scaling features (if needed)
from sklearn.model_selection import train_test_split  # For splitting data
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Load data (replace with your data)
data = pd.read_csv(r'C:\Users\James\Documents\ML datasets\covid.csv')
#step 2: data cleaning>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Drop rows with missing values (if any)
data = data.dropna()
print(data.head())
# Check for duplicates and remove them
data = data.drop_duplicates()

# Step 2: Inspect and clean the dataset
print(data.head())
print(data.info())
print(data.isnull().sum())
data = data.fillna(0)  # Fill NaNs with 0


X = data.drop("ICU", axis=1)  # features
y = data["ICU"]  # target

## step 7: Split the data into training and testing sets>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Choose kernel (e.g., 'linear', 'rbf', 'poly') and other parameters (e.g., C)
svc_model = svm.SVC(kernel='poly', C=50.0)  # Example with linear kernel

# Train the model
svc_model.fit(X_train, y_train)
#make prediction
y_pred = svc_model.predict(X_test)

#evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)

# Calculate precision, recall, and F1-score for multiclass
print(f"Accuracy: {accuracy:.2f}")
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")



# Define new input data (ensure it matches the feature set used for training)
new_data = np.array([[2,4,2,2,13,1,2023,2,1,48,3,0,0,0,0,0,0,0,0,0,0,6]])

# Apply the same scaling as used for the training data
new_data_scaled = scaler.transform(new_data)

# Make predictions with the new data
new_predictions = svc_model.predict(new_data_scaled)

print("New data prediction:", new_predictions)



# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
# Plotting the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Metrics data
metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}
# Plotting the metrics
fig, ax = plt.subplots()
bars = ax.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'red', 'purple'])

# Add text labels on bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

ax.set_title('SVM Classification Metrics')
ax.set_ylim(0, 1)  # Assuming metrics are between 0 and 1
ax.set_ylabel('Score')
plt.show()


# Load the wine dataset
wine = load_wine()
X, y = wine.data, wine.target


# Define the desired confusion matrix values
true_positives = 90
true_negatives = 100
false_positives = 30
false_negatives = 10

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

