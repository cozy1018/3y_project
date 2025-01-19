
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier

# Load and preprocess dataset
def load_and_preprocess_data():
    # Load the dataset and filter for Cleveland dataset
    df = pd.read_csv("datafile/heart_disease_uci.csv")
    df = df[df['dataset'] == "Cleveland"]
    df = df[df['age'] != 28]  # Remove rows with age 28
    df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)  # Convert target to binary
    df = df.dropna()  # Drop rows with missing values
    df = pd.get_dummies(df, drop_first=True)  # One-hot encode categorical features

    X = df.drop('num', axis=1)
    y = df['num']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the model
def train_and_evaluate(X_train, X_test, y_train, y_test, model, smote=False):
    if smote:
        # Apply SMOTE for oversampling the minority class
        smote_instance = SMOTE(random_state=42)
        X_train, y_train = smote_instance.fit_resample(X_train, y_train)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create an ensemble model using voting classifier
    ensemble_model = VotingClassifier(
        estimators=[('svm_bagging', model[0]),  # Bagged SVM model
                    ('knn_bagging', model[1]),  # Bagged KNN model
                    ('lr_bagging', model[2])],  # Bagged Logistic Regression model
        voting='hard'  # Majority voting
    )

    # Train the ensemble model
    ensemble_model.fit(X_train_scaled, y_train)

    # Make predictions and evaluate accuracy
    y_pred = ensemble_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
    plt.show()

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Define base models for BaggingClassifier
svm_model = SVC(random_state=42)
knn_model = KNeighborsClassifier()
lr_model = LogisticRegression(random_state=42)

# Set parameters for BaggingClassifier
n_estimators_value = 1
max_samples_value = 1.0
random_state_value = 42

# Create BaggingClassifier instances for each base model
bagging_svm = BaggingClassifier(svm_model, n_estimators=n_estimators_value, max_samples=max_samples_value, random_state=random_state_value)
bagging_knn = BaggingClassifier(knn_model, n_estimators=n_estimators_value, max_samples=max_samples_value, random_state=random_state_value)
bagging_lr = BaggingClassifier(lr_model, n_estimators=n_estimators_value, max_samples=max_samples_value, random_state=random_state_value)

# Group the models into a list
model = [bagging_svm, bagging_knn, bagging_lr]

# SVM, KNN, and LR with Bagging (without SMOTE)
print("### SVM, KNN, and LR with Bagging (without SMOTE) ###")
train_and_evaluate(X_train, X_test, y_train, y_test, model, smote=False)

# SVM, KNN, and LR with Bagging (with SMOTE)
print("### SVM, KNN, and LR with Bagging (with SMOTE) ###")
train_and_evaluate(X_train, X_test, y_train, y_test, model, smote=True)
