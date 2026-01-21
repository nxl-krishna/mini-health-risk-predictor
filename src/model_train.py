# src/model_train.py
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_prep import load_and_prep_data

# 1. Load Prepared Data
X_train, X_test, y_train, y_test, scaler = load_and_prep_data('../data/diabetes.csv')

# 2. Train Model
# Using 100 trees (n_estimators) and limiting depth to prevent overfitting
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 3. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualizing Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'], 
            yticklabels=['No Diabetes', 'Diabetes'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
# plt.show() # Uncomment to view plot

# 4. Save Model and Scaler
# We must save the scaler too! New user input must be scaled 
# using the EXACT same mean/std deviation as the training data.
artifacts = {
    'model': model,
    'scaler': scaler
}

with open('../model.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("Model and Scaler saved to model.pkl")