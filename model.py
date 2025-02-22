# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import warnings
# warnings.filterwarnings('ignore')


# __author__ = "SqueezyCoders"

# data = pd.read_csv("heartDiseaseDataSet_with_Refined_FABP3.csv")

# #age,gender,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target,FABP3

# X = data[["age", "gender", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "FABP3"]]
# y = data["target"]



# # Initializing 

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# rf_classifier.fit(X_train, y_train)

# y_pred = rf_classifier.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred)


# print(f"Accuracy: {accuracy:.2f}")
# print("\nClassification Report:\n", classification_rep)



# #Testing

# # Sampling
# sample = X_test.iloc[0:1]
# # Predicting on samples
# prediction = rf_classifier.predict(sample) 


# sample_dict = sample.iloc[0].to_dict()
# print(f"\nSample Heart Patients: {sample_dict}")
# print(f"Predicted Survival: {'HeartAttack' if prediction[0] == 1 else 'No Heart Attack'}")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

__author__ = "SqueezyCoders"

# Load the dataset
data = pd.read_csv("heartDiseaseDataSet_with_Refined_FABP3.csv")

# Define the features and target variable
X = data[["age", "gender", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "FABP3"]]
y = data["target"]

# Convert categorical variables to numerical if necessary
X['gender'] = X['gender'].astype('category').cat.codes

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search to find the best hyperparameters
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model
best_rf_classifier = grid_search.best_estimator_

# Fit the best model on the training data
best_rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = best_rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)

# Testing
# Sampling
sample = X_test.iloc[0:1]

# Predicting on samples
prediction = best_rf_classifier.predict(sample)

sample_dict = sample.iloc[0].to_dict()
print(f"\nSample Heart Patients: {sample_dict}")
print(f"Predicted Survival: {'HeartAttack' if prediction[0] == 1 else 'No Heart Attack'}")
