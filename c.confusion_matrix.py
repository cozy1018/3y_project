################
################
################
# From Step1 to Step2:
# Import necessary libraries
import pandas as pd 
from sklearn.preprocessing import StandardScaler, LabelEncoder  
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay  
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
import matplotlib.pyplot as plt  

# Define configurations for different classification tasks
# List of target classes: multi-class (5) and binary (2)
num_of_classes = [5, 2]  
# Output file names for predictions
file_names = ["svm_predictions_5", "svm_predictions_2"]  
# Define class type 
class_labels = ["Five Classes", "Two Classes"]

# Loop through the classification configurations
for i in range(len(num_of_classes)):
    print("The case for classes =", num_of_classes[i])
    
    # Load and filter the dataset
    df = pd.read_csv("datafile/heart_disease_uci.csv")
    df = df[df['dataset'] == "Cleveland"]  
    df = df[df['age'] != 28]

    # Adjust target column for binary classification if required
    if num_of_classes[i] == 2:
        # Convert target values: keep 0 as 0 and group 1-4 into class 1
        df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

    # Drop rows with missing values
    df = df.dropna()  
    
    # Encode categorical features into numerical format
    label_encoder = LabelEncoder()  
    categorical_columns = df.select_dtypes(include=['object']).columns  
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])  

    # Define features (X) and the target variable (y)
    X = df.drop('num', axis=1) 
    y = df['num']  

    # Split data into training and testing sets 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the features for consistent input to the SVM
    scaler = StandardScaler()  
    X_train_scaled = scaler.fit_transform(X_train)  
    X_test_scaled = scaler.transform(X_test) 

    # Train the SVM model
    svm_model = SVC(kernel='poly')  
    svm_model.fit(X_train_scaled, y_train)  

    # Make predictions on the test set
    y_pred = svm_model.predict(X_test_scaled)  

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred) 
    print(f"Accuracy for {num_of_classes[i]} classes: {accuracy:.2f}")
    
    # Save predictions to a CSV file
    # Create a DataFrame with actual and predicted values
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    # Save predictions to file
    results_df.to_csv('{}.csv'.format(file_names[i]), index=False)  

    # Generate and visualise the confusion matrix
    cm = confusion_matrix(y_test, y_pred)  
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')  
    plt.title('Confusion Matrix ({})'.format(class_labels[i]))
    plt.show()
    
################
################
################
# From Step2 to Step3:
# Import necessary libraries 
import pandas as pd 
from sklearn.preprocessing import StandardScaler, LabelEncoder  
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay  
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
import matplotlib.pyplot as plt  

print("Processing the case with 2 classes, applying feature scaling, and using one-hot encoding")

# Load and filter the dataset
df = pd.read_csv("datafile/heart_disease_uci.csv")
df = df[df['dataset'] == "Cleveland"]  
df = df[df['age'] != 28]

# Convert target values: keep 0 as 0 and group 1-4 into class 1
df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

# Drop rows with missing values
df = df.dropna()  
    
# One-hot encode categorical variables and drop the first level
df = pd.get_dummies(df, drop_first=True)
    
# Define features (X) and the target variable (y)
X = df.drop('num', axis=1)  # Features: all columns except the target column 'num'
y = df['num']  # Target: 'num' column

# Split data into training and testing sets 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features for consistent input to the SVM
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test) 
    
# Train the SVM model
svm_model = SVC(kernel='poly')  
svm_model.fit(X_train_scaled, y_train)  

# Make predictions on the test set
y_pred = svm_model.predict(X_test_scaled)  

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred) 
print(f"Accuracy for 2 classes: {accuracy:.2f}")

# Generate and visualise the confusion matrix
cm = confusion_matrix(y_test, y_pred)  
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')  
plt.title('Confusion Matrix (Step3)')
plt.show()

################
################
################
# From Step3 to Step4:
# Import necessary libraries 
import pandas as pd 
from sklearn.preprocessing import StandardScaler, LabelEncoder  
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay  
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
import matplotlib.pyplot as plt  
from sklearn.model_selection import GridSearchCV

print("Processing the case with 2 classes, applying feature scaling, one-hot encoding, and hyperparameter tuning")

# Load and filter the dataset
df = pd.read_csv("datafile/heart_disease_uci.csv")
df = df[df['dataset'] == "Cleveland"]  
df = df[df['age'] != 28]

# Convert target values: keep 0 as 0 and group 1-4 into class 1
df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

# Drop rows with missing values
df = df.dropna()  
    
# One-hot encode categorical variables and drop the first level
df = pd.get_dummies(df, drop_first=True)
    
# Define features (X) and the target variable (y)
X = df.drop('num', axis=1)  # Features: all columns except the target column 'num'
y = df['num']  # Target: 'num' column

# Split data into training and testing sets 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features for consistent input to the SVM
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test) 

# Hyperparameter tuning for SVM
param_grid = {
    'C': [0.1, 1, 5, 10, 20],  # Regularization parameter
    'kernel': ['linear', 'sigmoid', 'poly', 'rbf'],  # Kernel types
    'gamma': [0.05, 0.1, 0.2]  # Kernel coefficient for 'rbf', 'sigmoid', 'poly'
}

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Get the best model from the grid search
best_svm = grid_search.best_estimator_

# Predict on the test set using the best model
y_pred = best_svm.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of SVM with Hyperparameter Tuning: {accuracy:.2f}")
print("Best parameters found:", grid_search.best_params_)

# Generate and visualise the confusion matrix
cm = confusion_matrix(y_test, y_pred)  
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
plt.title('Confusion Matrix (Step4)')
plt.show()

################
################
################
# Assess the accuracy of predicting whether someone is likely to have a heart attack for male and female groups
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("datafile/heart_disease_uci.csv")

print("\nAssess the accuracy of predicting whether someone is likely to have a heart attack for male and female groups")

# Drop any rows with missing values
df = df.dropna()

# Filter the dataframe
df = df[df['dataset'] == "Cleveland"]
df = df[df['age'] != 28]

# Data preprocessing 
df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

# One-hot encode categorical variables and drop the first level
df = pd.get_dummies(df, drop_first=True)

# Define features and target 
X = df.drop('num', axis=1)
y = df['num']

# Split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the test set into male and female subsets
male_X_test = X_test[X_test['sex_Male'] == 1]
female_X_test = X_test[X_test['sex_Male'] == 0]

male_y_test = y_test[X_test['sex_Male'] == 1]
female_y_test = y_test[X_test['sex_Male'] == 0]

# Scale features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM model training 
svm_model = SVC(C = 1, gamma = 0.05, kernel = 'linear')
svm_model.fit(X_train, y_train)

# Predict on the test set  
y_pred = svm_model.predict(X_test)

# Calculate and print accuracy 
accuracy = accuracy_score(y_test, y_pred)
print("SVM")
print(f"Overall accuracy: {accuracy:.2f}")

# Generate and visualise the confusion matrix
cm = confusion_matrix(y_test, y_pred)  
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
plt.title('Confusion Matrix (Overall)')
plt.show()

# Evaluate accuracy for male and female subsets
for gender in ['Male', 'Female']:
    # Subset and scale test data by gender
    if gender == "Male":
        male_X_test = scaler.transform(male_X_test)
        gender_test_data = male_X_test
        gender_test_labels = male_y_test
    else:
        female_X_test = scaler.transform(female_X_test)
        gender_test_data = female_X_test
        gender_test_labels = female_y_test

    # Make predictions
    predictions = svm_model.predict(gender_test_data)
    
    # Calculate and print accuracy for each gender subset 
    gender_accuracy = accuracy_score(gender_test_labels, predictions)
    print(f"\nPerformance for {gender} subset:")
    print(f"Accuracy: {gender_accuracy:.2f}")

    # Generate and visualise the confusion matrix
    cm = confusion_matrix(gender_test_labels, predictions)  
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
    plt.title('Confusion Matrix ({})'.format(gender))
    plt.show()