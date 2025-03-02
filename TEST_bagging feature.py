# Bagging - LR, SVM, Naïve Bayes with Paper Features (No One-Hot Encoding, Fixing Categorical Variables)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE

print("Processing data with two classes to build ensemble models (LR, SVM, NB) using Bagging, without One-Hot Encoding, fixing categorical variables.")

# Load and filter the dataset
df = pd.read_csv("datafile/heart_disease_uci.csv")
df = df[df['dataset'] == "Cleveland"]
df = df[df['age'] != 28]

# Convert target values: keep 0 as 0 and group 1-4 into class 1
df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
df['num'] = df['num'].astype(str)

# Print class distribution
counts = df['num'].value_counts()
print("Counts for the 'num' column: \n", counts)

# Drop rows with missing values
df = df.dropna()

# Convert categorical variables to numerical codes
categorical_columns = ['sex', 'cp', 'restecg', 'slope', 'thal', 'fbs', 'exang']  # Add relevant categorical features
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype('category').cat.codes

# Define feature sets for each model based on the paper
features_lr = ['id', 'age', 'sex', 'cp', 'chol', 'restecg', 'oldpeak', 'slope', 'ca', 'thal', 'num']
features_svm = ['id', 'age', 'sex', 'cp', 'chol', 'fbs', 'exang', 'oldpeak', 'slope', 'ca', 'num']
features_nb = ['id', 'sex', 'cp', 'thal', 'exang', 'oldpeak', 'ca', 'num']

# Extract selected features from the dataset
df_lr = df[features_lr].copy()
df_svm = df[features_svm].copy()
df_nb = df[features_nb].copy()

# Define target variable
y = df['num']

# Split data into training and testing sets (80% training, 20% testing)
rs = 42
X_train_lr, X_test_lr, y_train, y_test = train_test_split(df_lr.drop(columns=['id', 'num']), y, test_size=0.2, random_state=rs)
X_train_svm, X_test_svm, _, _ = train_test_split(df_svm.drop(columns=['id', 'num']), y, test_size=0.2, random_state=rs)
X_train_nb, X_test_nb, _, _ = train_test_split(df_nb.drop(columns=['id', 'num']), y, test_size=0.2, random_state=rs)

# Scale the features (except Naïve Bayes, which doesn’t require scaling)
scaler = StandardScaler()
X_train_lr_scaled = scaler.fit_transform(X_train_lr)
X_test_lr_scaled = scaler.transform(X_test_lr)

X_train_svm_scaled = scaler.fit_transform(X_train_svm)
X_test_svm_scaled = scaler.transform(X_test_svm)

# Define base classifiers
svm = SVC(random_state=rs, probability=True)
lr = LogisticRegression(random_state=rs, max_iter=1000)
nb = GaussianNB()

# Train individual models
lr.fit(X_train_lr_scaled, y_train)
svm.fit(X_train_svm_scaled, y_train)
nb.fit(X_train_nb, y_train)  # No scaling for Naïve Bayes

# Create a Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('lr', lr), ('svm', svm), ('nb', nb)], 
    voting='hard'
)

# Use BaggingClassifier with the VotingClassifier
bagging_voting = BaggingClassifier(
    estimator=voting_clf,
    random_state=rs
)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],  
    'max_samples': [0.8, 0.9, 1.0],  
}

# Case 1: Without SMOTE
print("\n### Case 1: Training and Testing Without SMOTE ###")
grid_search_no_smote = GridSearchCV(bagging_voting, param_grid, cv=5, scoring='accuracy')
grid_search_no_smote.fit(X_train_lr_scaled, y_train)

# Best model without SMOTE
best_model_no_smote = grid_search_no_smote.best_estimator_

# Predict on the test set
y_pred_no_smote = best_model_no_smote.predict(X_test_lr_scaled)

# Evaluate performance
accuracy_no_smote = accuracy_score(y_test, y_pred_no_smote)
print(f"Accuracy without SMOTE: {accuracy_no_smote:.2f}")
print("Best parameters without SMOTE:", grid_search_no_smote.best_params_)

###########################################
###########################################

# Case 2: With SMOTE
print("\n### Case 2: Training and Testing With SMOTE ###")

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_lr, y_train)

# Scale the features after SMOTE
X_train_scaled_smote = scaler.fit_transform(X_train_smote)
X_test_scaled_smote = scaler.transform(X_test_lr)

# Grid search for SMOTE case
grid_search_smote = GridSearchCV(bagging_voting, param_grid, cv=5, scoring='accuracy')
grid_search_smote.fit(X_train_scaled_smote, y_train_smote)

# Best model with SMOTE
best_model_smote = grid_search_smote.best_estimator_

# Predict on the test set
y_pred_smote = best_model_smote.predict(X_test_scaled_smote)

# Evaluate performance
accuracy_smote = accuracy_score(y_test, y_pred_smote)
print(f"Accuracy with SMOTE: {accuracy_smote:.2f}")
print("Best parameters with SMOTE:", grid_search_smote.best_params_)
