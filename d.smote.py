# SVM
# Import necessary libraries 
import pandas as pd 
from sklearn.preprocessing import StandardScaler, LabelEncoder  
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay  
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
import matplotlib.pyplot as plt  
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

print("Processing data with two classes to build SVM models, applying feature scaling, one-hot encoding, and hyperparameter tuning.")

# Load and filter the dataset
df = pd.read_csv("datafile/heart_disease_uci.csv")
df = df[df['dataset'] == "Cleveland"]  
df = df[df['age'] != 28]

# Convert target values: keep 0 as 0 and group 1-4 into class 1
df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

# Example of getting counts for the 'num' column
counts = df['num'].value_counts()
print("counts for the 'num' column : \n", counts)

# Drop rows with missing values
df = df.dropna()  
    
# One-hot encode categorical variables and drop the first level
df = pd.get_dummies(df, drop_first=True)
    
# Define features (X) and the target variable (y)
X = df.drop('num', axis=1)  
y = df['num']  

# Split data into training and testing sets 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter tuning for SVM
param_grid = {
    'C': [0.1,1.095,2.09,3.085,4.08,5.075,6.07,7.065,8.06,9.055,10.05,11.045,12.04,13.035,14.03,15.025,16.02,17.015,18.01,19.005,20.0],  
    'kernel': ['linear', 'sigmoid', 'poly', 'rbf'], 
    'gamma': [0.05, 0.0875, 0.125, 0.1625, 0.2] 
}

# Case 1: Without SMOTE
print("\n### Case 1: Training and Testing Without SMOTE ###")

# Scale the features
scaler_no_smote = StandardScaler()
X_train_scaled_no_smote = scaler_no_smote.fit_transform(X_train)
X_test_scaled_no_smote = scaler_no_smote.transform(X_test)

# Hyperparameter tuning using GridSearchCV
grid_search_no_smote = GridSearchCV(SVC(), param_grid, cv=5)
grid_search_no_smote.fit(X_train_scaled_no_smote, y_train)

# Best model without SMOTE
best_svm_no_smote = grid_search_no_smote.best_estimator_

# Predict on the test set
y_pred_no_smote = best_svm_no_smote.predict(X_test_scaled_no_smote)

# Evaluate the model
accuracy_no_smote = accuracy_score(y_test, y_pred_no_smote)
print(f"Accuracy without SMOTE: {accuracy_no_smote:.2f}")
print("Best parameters without SMOTE:", grid_search_no_smote.best_params_)

# Confusion matrix without SMOTE
cm_no_smote = confusion_matrix(y_test, y_pred_no_smote)
ConfusionMatrixDisplay(confusion_matrix=cm_no_smote).plot(cmap='Blues')
plt.title('Confusion Matrix without SMOTE (SVM)')
plt.show()

# Case 2: With SMOTE
print("\n### Case 2: Training and Testing With SMOTE ###")

print("Class distribution before SMOTE:")
print(pd.Series(y_train).value_counts())

# Apply SMOTE to balance the training dataset
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# Scale the features
scaler_smote = StandardScaler()
X_train_scaled_smote = scaler_smote.fit_transform(X_train_smote)
X_test_scaled_smote = scaler_smote.transform(X_test)

# Hyperparameter tuning using GridSearchCV
grid_search_smote = GridSearchCV(SVC(), param_grid, cv=5)
grid_search_smote.fit(X_train_scaled_smote, y_train_smote)

# Best model with SMOTE
best_svm_smote = grid_search_smote.best_estimator_

# Predict on the test set
y_pred_smote = best_svm_smote.predict(X_test_scaled_smote)

# Evaluate the model
accuracy_smote = accuracy_score(y_test, y_pred_smote)
print(f"Accuracy with SMOTE: {accuracy_smote:.2f}")
print("Best parameters with SMOTE:", grid_search_smote.best_params_)

# Confusion matrix with SMOTE
cm_smote = confusion_matrix(y_test, y_pred_smote)
ConfusionMatrixDisplay(confusion_matrix=cm_smote).plot(cmap='Blues')
plt.title('Confusion Matrix with SMOTE (SVM)')
plt.show()



# Logistic regression (LR)
# Import necessary libraries 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

print("Processing data with two classes to build LR models, applying feature scaling, one-hot encoding, and hyperparameter tuning.")

# Load and filter the dataset
df = pd.read_csv("datafile/heart_disease_uci.csv")
df = df[df['dataset'] == "Cleveland"]
df = df[df['age'] != 28]

# Convert target values: keep 0 as 0 and group 1-4 into class 1
df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

# Example of getting counts for the 'num' column
counts = df['num'].value_counts()
print("counts for the 'num' column : \n", counts)

# Drop rows with missing values
df = df.dropna()
    
# One-hot encode categorical variables and drop the first level
df = pd.get_dummies(df, drop_first=True)
    
# Define features (X) and the target variable (y)
X = df.drop('num', axis=1)  
y = df['num'] 

# Split data into training and testing sets 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter tuning parameters for Logistic Regression
param_grid = [
    {'penalty': ['l1','l2'], 'C': [0.1,1.095,2.09,3.085,4.08,5.075,6.07,7.065,8.06,9.055,10.05,11.045,12.04,13.035,14.03,15.025,16.02,17.015,18.01,19.005,20.0], 'solver': ['liblinear']},  
    {'penalty': ['l2'], 'C': [0.1,1.095,2.09,3.085,4.08,5.075,6.07,7.065,8.06,9.055,10.05,11.045,12.04,13.035,14.03,15.025,16.02,17.015,18.01,19.005,20.0], 'solver': ['lbfgs', 'sag']} 
]

# Case 1: Without SMOTE
print("\n### Case 1: Training and Testing Without SMOTE ###")

# Scale the features
scaler_no_smote = StandardScaler()
X_train_scaled_no_smote = scaler_no_smote.fit_transform(X_train)
X_test_scaled_no_smote = scaler_no_smote.transform(X_test)

# Hyperparameter tuning using GridSearchCV
grid_search_no_smote = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid_search_no_smote.fit(X_train_scaled_no_smote, y_train)

# Best model without SMOTE
best_lr_no_smote = grid_search_no_smote.best_estimator_

# Predict on the test set
y_pred_no_smote = best_lr_no_smote.predict(X_test_scaled_no_smote)

# Evaluate the model
accuracy_no_smote = accuracy_score(y_test, y_pred_no_smote)
print(f"Accuracy without SMOTE: {accuracy_no_smote:.2f}")
print("Best parameters without SMOTE:", grid_search_no_smote.best_params_)

# Confusion matrix without SMOTE
cm_no_smote = confusion_matrix(y_test, y_pred_no_smote)
ConfusionMatrixDisplay(confusion_matrix=cm_no_smote).plot(cmap='Blues')
plt.title('Confusion Matrix without SMOTE (LR)')
plt.show()

# Case 2: With SMOTE
print("\n### Case 2: Training and Testing With SMOTE ###")

# Apply SMOTE to balance the training dataset
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Scale the features
scaler_smote = StandardScaler()
X_train_scaled_smote = scaler_smote.fit_transform(X_train_smote)
X_test_scaled_smote = scaler_smote.transform(X_test)

# Hyperparameter tuning using GridSearchCV
grid_search_smote = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid_search_smote.fit(X_train_scaled_smote, y_train_smote)

# Best model with SMOTE
best_lr_smote = grid_search_smote.best_estimator_

# Predict on the test set
y_pred_smote = best_lr_smote.predict(X_test_scaled_smote)

# Evaluate the model
accuracy_smote = accuracy_score(y_test, y_pred_smote)
print(f"Accuracy with SMOTE: {accuracy_smote:.2f}")
print("Best parameters with SMOTE:", grid_search_smote.best_params_)

# Confusion matrix with SMOTE
cm_smote = confusion_matrix(y_test, y_pred_smote)
ConfusionMatrixDisplay(confusion_matrix=cm_smote).plot(cmap='Blues')
plt.title('Confusion Matrix with SMOTE (LR)')
plt.show()


# KNN
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

print("Processing data with two classes to build KNN models, applying feature scaling, one-hot encoding, and hyperparameter tuning.")

# Load and filter the dataset
df = pd.read_csv("datafile/heart_disease_uci.csv")
df = df[df['dataset'] == "Cleveland"]
df = df[df['age'] != 28]

# Convert target values: keep 0 as 0 and group 1-4 into class 1
df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

# Example of getting counts for the 'num' column
counts = df['num'].value_counts()
print("Counts for the 'num' column:\n", counts)

# Drop rows with missing values
df = df.dropna()

# One-hot encode categorical variables and drop the first level
df = pd.get_dummies(df, drop_first=True)

# Define features (X) and the target variable (y)
X = df.drop('num', axis=1)  
y = df['num'] 

# Split data into training and testing sets 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter tuning for KNN
param_grid = {
    'n_neighbors': [1, 2, 3, 4, 5 ,6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 
    'weights': ['uniform', 'distance'] 
}

# Case 1: Without SMOTE
print("\n### Case 1: Training and Testing Without SMOTE ###")

# Scale the features
scaler_no_smote = StandardScaler()
X_train_scaled_no_smote = scaler_no_smote.fit_transform(X_train)
X_test_scaled_no_smote = scaler_no_smote.transform(X_test)

# Hyperparameter tuning using GridSearchCV
grid_search_no_smote = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search_no_smote.fit(X_train_scaled_no_smote, y_train)

# Best model without SMOTE
best_knn_no_smote = grid_search_no_smote.best_estimator_

# Predict on the test set
y_pred_no_smote = best_knn_no_smote.predict(X_test_scaled_no_smote)

# Evaluate the model
accuracy_no_smote = accuracy_score(y_test, y_pred_no_smote)
print(f"Accuracy without SMOTE: {accuracy_no_smote:.2f}")
print("Best parameters without SMOTE:", grid_search_no_smote.best_params_)

# Confusion matrix without SMOTE
cm_no_smote = confusion_matrix(y_test, y_pred_no_smote)
ConfusionMatrixDisplay(confusion_matrix=cm_no_smote).plot(cmap='Blues')
plt.title('Confusion Matrix without SMOTE (KNN)')
plt.show()

# Case 2: With SMOTE
print("\n### Case 2: Training and Testing With SMOTE ###")

# Apply SMOTE to balance the training dataset
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Scale the features
scaler_smote = StandardScaler()
X_train_scaled_smote = scaler_smote.fit_transform(X_train_smote)
X_test_scaled_smote = scaler_smote.transform(X_test)

# Hyperparameter tuning using GridSearchCV
grid_search_smote = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search_smote.fit(X_train_scaled_smote, y_train_smote)

# Best model with SMOTE
best_knn_smote = grid_search_smote.best_estimator_

# Predict on the test set
y_pred_smote = best_knn_smote.predict(X_test_scaled_smote)

# Evaluate the model
accuracy_smote = accuracy_score(y_test, y_pred_smote)
print(f"Accuracy with SMOTE: {accuracy_smote:.2f}")
print("Best parameters with SMOTE:", grid_search_smote.best_params_)

# Confusion matrix with SMOTE
cm_smote = confusion_matrix(y_test, y_pred_smote)
ConfusionMatrixDisplay(confusion_matrix=cm_smote).plot(cmap='Blues')
plt.title('Confusion Matrix with SMOTE (KNN)')
plt.show()

