#### Results presented in the paper (Naïve Bayes)
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

print("Processing data with two classes to build Naïve Bayes models, applying feature scaling, and hyperparameter tuning.")

# Load and filter the dataset
df = pd.read_csv("datafile/heart_disease_uci.csv")
df = df[df['dataset'] == "Cleveland"]
df = df[df['age'] != 28]

# Convert target values: keep 0 as 0 and group 1-4 into class 1
df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
df['num'] = df['num'].astype(str)

# Drop rows with missing values
df = df.dropna()

# Mapping categorical columns to numeric values
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
df['cp'] = df['cp'].map({'typical angina': 1, 'atypical angina': 2, 'non-anginal': 3, 'asymptomatic': 4})
df['fbs'] = df['fbs'].map({True: 1, False: 0})
df['restecg'] = df['restecg'].map({'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2})
df['exang'] = df['exang'].map({True: 1, False: 0})
df['slope'] = df['slope'].map({'upsloping': 1, 'flat': 2, 'downsloping': 3})
df['thal'] = df['thal'].map({'normal': 3, 'fixed defect': 6, 'reversable defect': 7})

# List of selected features
features = ['sex', 'cp', 'thalch', 'exang', 'oldpeak', 'ca', 'num']
df = df[features]

# Define features (X) and target variable (y)
X = df.drop('num', axis=1)
y = df['num']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the hyperparameter grid for Naïve Bayes
param_grid = {
    'var_smoothing': [10**(-i) for i in range(12, 6, -1)]  # 1e-12 to 1e-6
}

# Case 1: Without SMOTE
print("\n### Case 1: Training and Testing Without SMOTE ###")

# Scale the features
scaler_no_smote = StandardScaler()
X_train_scaled_no_smote = scaler_no_smote.fit_transform(X_train)
X_test_scaled_no_smote = scaler_no_smote.transform(X_test)

# Hyperparameter tuning using GridSearchCV
grid_search_no_smote = GridSearchCV(GaussianNB(), param_grid, cv=10)
grid_search_no_smote.fit(X_train_scaled_no_smote, y_train)

# Best model without SMOTE
best_nb_no_smote = grid_search_no_smote.best_estimator_

# Predict on the test set
y_pred_no_smote = best_nb_no_smote.predict(X_test_scaled_no_smote)

# Evaluate the model
accuracy_no_smote = accuracy_score(y_test, y_pred_no_smote)
print(f"Accuracy without SMOTE: {accuracy_no_smote:.4f}")
print("Best parameters without SMOTE:", grid_search_no_smote.best_params_)



########################################
########################################
# Naïve Bayes - with SMOTE
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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
grid_search_smote = GridSearchCV(GaussianNB(), param_grid, cv=10)
grid_search_smote.fit(X_train_scaled_smote, y_train_smote)

# Best model with SMOTE
best_nb_smote = grid_search_smote.best_estimator_

# Predict on the test set
y_pred_smote = best_nb_smote.predict(X_test_scaled_smote)

# Evaluate the model
accuracy_smote = accuracy_score(y_test, y_pred_smote)
print(f"Accuracy with SMOTE: {accuracy_smote:.4f}")
print("Best parameters with SMOTE:", grid_search_smote.best_params_)