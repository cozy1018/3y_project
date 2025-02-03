#### Results presented in the paper.
# Testing - Logistic regression (LR) - Without SMOTE
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
df['num'] = df['num'].astype('str')

# Example of getting counts for the 'num' column
counts = df['num'].value_counts()
print("counts for the 'num' column : \n", counts)

# Drop rows with missing values
df = df.dropna()

# List of selected features to be used from the dataframe
features = ['id', 'age', 'sex', 'cp', 'chol', 'restecg', 'oldpeak', 'slope', 'ca', 'thal', 'num']
df = df[features]

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
grid_search_no_smote = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), param_grid, cv=5)
grid_search_no_smote.fit(X_train_scaled_no_smote, y_train)
    
# Best model without SMOTE
best_lr_no_smote = grid_search_no_smote.best_estimator_
    
# Predict on the test set
y_pred_no_smote = best_lr_no_smote.predict(X_test_scaled_no_smote)
    
# Evaluate the model
accuracy_no_smote = accuracy_score(y_test, y_pred_no_smote)
print(f"Accuracy without SMOTE: {accuracy_no_smote:.2f}")
print("Best parameters without SMOTE:", grid_search_no_smote.best_params_)


###########################################
###########################################


# Testing - Logistic regression (LR) - With SMOTE
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
df['num'] = df['num'].astype('str')

# Example of getting counts for the 'num' column
counts = df['num'].value_counts()
print("counts for the 'num' column : \n", counts)

# Drop rows with missing values
df = df.dropna()

# List of selected features to be used from the dataframe
features = ['id', 'age', 'sex', 'cp', 'chol', 'restecg', 'oldpeak', 'slope', 'ca', 'thal', 'num']
df = df[features]
    
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

# Hyperparameter tuning parameters for Logistic Regression
param_grid = [
    {'penalty': ['l1','l2'], 'C': [0.1,1.095,2.09,3.085,4.08,5.075,6.07,7.065,8.06,9.055,10.05,11.045,12.04,13.035,14.03,15.025,16.02,17.015,18.01,19.005,20.0], 'solver': ['liblinear']},  
    {'penalty': ['l2'], 'C': [0.1,1.095,2.09,3.085,4.08,5.075,6.07,7.065,8.06,9.055,10.05,11.045,12.04,13.035,14.03,15.025,16.02,17.015,18.01,19.005,20.0], 'solver': ['lbfgs', 'sag']} 
]

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
grid_search_smote = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), param_grid, cv=5)
grid_search_smote.fit(X_train_scaled_smote, y_train_smote)

# Best model with SMOTE
best_lr_smote = grid_search_smote.best_estimator_

# Predict on the test set
y_pred_smote = best_lr_smote.predict(X_test_scaled_smote)

# Evaluate the model
accuracy_smote = accuracy_score(y_test, y_pred_smote)
print(f"Accuracy with SMOTE: {accuracy_smote:.2f}")
print("Best parameters with SMOTE:", grid_search_smote.best_params_)
