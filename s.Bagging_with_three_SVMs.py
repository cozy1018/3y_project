import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
def load_preprocess_data(csv_file, random_state):
    df = pd.read_csv(csv_file)
    df = df[(df['dataset'] == "Cleveland") & (df['age'] != 28)]
    df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1).astype('str')
    df = pd.get_dummies(df.dropna(), drop_first=True)
    X = df.drop(['num_1', 'id'], axis=1)
    y = df['num_1']
    return train_test_split(X, y, test_size=0.2, random_state=random_state)

# Load dataset
X_train, X_test, y_train, y_test = load_preprocess_data("datafile/heart_disease_uci.csv", random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the base estimator (Logistic Regression)
base_estimator = LogisticRegression(max_iter=1000, random_state=42)

# Train BaggingClassifier with 3 base estimators 
bagging_clf = BaggingClassifier(base_estimator, n_estimators=3, max_samples=0.8, random_state=42)
bagging_clf.fit(X_train_scaled, y_train)

# Get individual predictions from each base estimator
individual_preds = np.array([est.predict(X_test_scaled) for est in bagging_clf.estimators_])

# Get the final prediction from the bagging classifier
final_preds = bagging_clf.predict(X_test_scaled)

# Convert to DataFrame for better visualization
results_df = pd.DataFrame({
    "Estimator_1": individual_preds[0],
    "Estimator_2": individual_preds[1],
    "Estimator_3": individual_preds[2],
    "Final_Bagging": final_preds,
    "Actual": y_test.values
})

# Show results for first 10 samples
print(results_df.head(10))

# Calculate accuracy for individual estimators and bagging
acc_est_1 = accuracy_score(y_test, individual_preds[0])
acc_est_2 = accuracy_score(y_test, individual_preds[1])
acc_est_3 = accuracy_score(y_test, individual_preds[2])
acc_bagging = accuracy_score(y_test, final_preds)

print(f"Estimator 1 Accuracy: {acc_est_1:.2f}")
print(f"Estimator 2 Accuracy: {acc_est_2:.2f}")
print(f"Estimator 3 Accuracy: {acc_est_3:.2f}")
print(f"Final Bagging Accuracy: {acc_bagging:.2f}")