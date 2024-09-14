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

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Fit the logistic regression model on the resampled data
model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred_resampled = model.predict(X_test_scaled)
y_pred_proba_resampled = model.predict_proba(X_test_scaled)[:, 1]

# Calculate and print the new evaluation metrics
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

# Plot ROC curve and calculate AUC for resampled model
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
