# Without SMOTE
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance

# Load and preprocess the dataset
df = pd.read_csv("datafile/heart_disease_uci.csv")
df = df[df['dataset'] == "Cleveland"]
df = df[df['age'] != 28]

df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
df['num'] = df['num'].astype('str')
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)

# Define features (X) and the target variable (y)
X = df.drop('num_1', axis=1)
X = X.drop('id', axis=1)
y = df['num_1']
rs = 49

# Function to identify and remove the least important feature
def remove_least_important_feature(model, X_train, X_test, y_train, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=rs)
    feature_importances = result.importances_mean
    least_important_index = np.argmin(feature_importances)
    return least_important_index

# Function to train and evaluate Naïve Bayes with grid search and permutation importance
def train_and_evaluate_nb_with_gridsearch(X, y, cv=5):
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rs
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    num_features = []
    accuracies = []
    features_used = []
    test_accuracies = []  
    all_features_used = []  

    remaining_features = list(X.columns)

    # Define hyperparameter grid for Naïve Bayes
    param_grid = {
        'var_smoothing': [10**(-i) for i in range(12, 6, -1)]  # 1e-12 to 1e-6
    }

    while len(remaining_features) > 0:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X[remaining_features], y, test_size=0.2, random_state=rs
        )

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_temp = X_train_scaled
        X_test_temp = X_test_scaled
    
        # Using Naïve Bayes model with GridSearchCV for hyperparameter tuning
        model = GaussianNB()
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv)
        grid_search.fit(X_train_temp, y_train)

        # Get best parameters from grid search
        best_params = grid_search.best_params_
        print(f"Best parameters for this iteration: {best_params}")  

        # Predict and evaluate accuracy using cross-validation
        best_model = grid_search.best_estimator_
        cv_accuracy = grid_search.best_score_
        y_pred = best_model.predict(X_test_temp)
        test_accuracy = accuracy_score(y_test, y_pred)

        # Store results
        num_features.append(len(remaining_features))
        accuracies.append(cv_accuracy)
        test_accuracies.append(test_accuracy)
        features_used.append(list(remaining_features))  
        all_features_used.append(list(remaining_features))  

        # Identify the least important feature using permutation importance
        least_important_index = remove_least_important_feature(best_model, X_train_temp, X_test_temp, y_train, y_test)
        
        # Remove the least important feature
        del remaining_features[least_important_index]
        X_train_temp = np.delete(X_train_temp, least_important_index, axis=1)
        X_test_temp = np.delete(X_test_temp, least_important_index, axis=1)

    return num_features, accuracies, features_used, test_accuracies, all_features_used


# Without SMOTE (as requested)
num_features_no_smote, accuracies_no_smote, features_used_no_smote, test_accuracies_no_smote, all_features_no_smote = train_and_evaluate_nb_with_gridsearch(
    X, y
)

# Plot accuracy vs. number of features for the scenario without SMOTE
plt.figure(figsize=(8, 6))
plt.plot(num_features_no_smote, test_accuracies_no_smote, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Features')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs Number of Features (Naïve Bayes with GridSearchCV, no SMOTE)')
plt.grid()
plt.show()

# Print results for the case without SMOTE
print("Final Results without SMOTE:")
for n, acc, features, test_acc in zip(num_features_no_smote, accuracies_no_smote, features_used_no_smote, test_accuracies_no_smote):
    print(f"Number of features: {n}, Cross-Validation Accuracy: {acc:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Features used: {features}")

# Print all features used in each iteration
print("\nFeatures used in each iteration:")
for i, features in enumerate(all_features_no_smote, 1):
    print(f"Iteration {i}: {features}")
