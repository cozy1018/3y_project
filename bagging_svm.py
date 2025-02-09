# (3) Bagging - SVM

# Import necessary libraries 
import pandas as pd 
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import accuracy_score  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.svm import SVC  
import matplotlib.pyplot as plt  
from imblearn.over_sampling import SMOTE  
from sklearn.ensemble import BaggingClassifier  

print("Processing data with two classes to build SVM models, applying feature scaling, one-hot encoding and hyperparameter tuning.")

# Load and filter the dataset
df = pd.read_csv("datafile/heart_disease_uci.csv")
df = df[df['dataset'] == "Cleveland"]  
df = df[df['age'] != 28]  

# Convert target values: keep 0 as 0 and group 1-4 into class 1
df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)  
df['num'] = df['num'].astype('str')

# Example of getting counts for the 'num' column
counts = df['num'].value_counts()  

# Drop rows with missing values
df = df.dropna()  

# One-hot encode categorical variables and drop the first level
df = pd.get_dummies(df, drop_first=True)
    
# Define features (X) and the target variable (y)
X = df.drop('num_1', axis=1)  
X = X.drop('id', axis=1)  
y = df['num_1']  

# Split data into training and testing sets 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Case 1: Without SMOTE
print("\n### Case 1: Training and Testing Without SMOTE ###")

# Scale the features
scaler_no_smote = StandardScaler()
X_train_scaled_no_smote = scaler_no_smote.fit_transform(X_train)
X_test_scaled_no_smote = scaler_no_smote.transform(X_test)

# Initialize SVM and Bagging Models
svm_model = SVC(random_state=42)

# Define BaggingClassifier with SVM
bagging_model = BaggingClassifier(
    estimator=svm_model,
    bootstrap=True,  # Use Bagging
    random_state=42
)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 150, 200],  # Number of models to train
    'max_samples': [0.8, 0.9, 1.0],  # Fraction of training samples
}

# Perform grid search
grid_search = GridSearchCV(bagging_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled_no_smote, y_train)

# Print best parameters and accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled_no_smote)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)


###############################
###############################
print("Processing data with two classes to build SVM models, applying feature scaling, SMOTE, one-hot encoding, and hyperparameter tuning.")
# Case 2: With SMOTE
print("\n### Case 2: Training and Testing With SMOTE ###")

# Load and filter the dataset
df = pd.read_csv("datafile/heart_disease_uci.csv")
df = df[df['dataset'] == "Cleveland"]  
df = df[df['age'] != 28]  

# Convert target values: keep 0 as 0 and group 1-4 into class 1
df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)  
df['num'] = df['num'].astype('str')

# Example of getting counts for the 'num' column
counts = df['num'].value_counts()  

# Drop rows with missing values
df = df.dropna()  

# One-hot encode categorical variables and drop the first level
df = pd.get_dummies(df, drop_first=True)
    
# Define features (X) and the target variable (y)
X = df.drop('num_1', axis=1)  
X = X.drop('id', axis=1)  
y = df['num_1']  

# Split data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply SMOTE to balance the dataset
print("\n### Applying SMOTE to Balance Classes ###")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Scale the features after SMOTE
scaler_smote = StandardScaler()
X_train_scaled_smote = scaler_smote.fit_transform(X_train_smote)
X_test_scaled_smote = scaler_smote.transform(X_test)

# Initialize SVM and Bagging Models
svm_model = SVC(random_state=42)

# Define BaggingClassifier with SVM
bagging_model = BaggingClassifier(
    estimator=svm_model,
    bootstrap=True,  # Use Bagging
    random_state=42
)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 150, 200],  # Number of models to train
    'max_samples': [0.8, 0.9, 1.0],  # Fraction of training samples
}

# Perform grid search
grid_search = GridSearchCV(bagging_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled_smote, y_train_smote)

# Print best parameters and accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled_smote)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)
