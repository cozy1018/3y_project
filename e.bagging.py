# Import necessary libraries 
import pandas as pd 
from sklearn.preprocessing import StandardScaler, LabelEncoder  
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay  
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
import matplotlib.pyplot as plt  
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import BaggingClassifier

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

# Case 1: Without SMOTE
print("\n### Case 1: Training and Testing Without SMOTE ###")

# Scale the features
scaler_no_smote = StandardScaler()
X_train_scaled_no_smote = scaler_no_smote.fit_transform(X_train)
X_test_scaled_no_smote = scaler_no_smote.transform(X_test)

# Initialize SVM and Bagging Models
svm_model = SVC(random_state = 42)

# Define BaggingClassifier with SVM
bagging_model = BaggingClassifier(estimator=svm_model,
                                bootstrap=True, # Use Bagging
                                random_state=42
                               )

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],        # Number of models to train
    'max_samples': [0.8, 0.85, 0.9, 0.95, 1.0],      # Fraction of training samples
}

# Perform grid search
grid_search = GridSearchCV(bagging_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled_no_smote, y_train)

# Accessing the cv_results_ attribute
results = pd.DataFrame(grid_search.cv_results_)

# Display the relevant columns: parameters, mean test score, and rank
results_display = results[['param_n_estimators', 'param_max_samples', 'mean_test_score', 'rank_test_score', 'params']]

# Print the table
print(results_display)

# Print best parameters and accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled_no_smote)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)
