# Case 1: Cleveland data, Case 2: Cleveland data and hungary data, Case 3: all data
# Import necessary libraries 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

print("Performance comparison across three different cases.")

# Load dataset
df = pd.read_csv("datafile/heart_disease_uci.csv")

# Check initial shape and drop any rows with missing values
print("Original shape:", df.shape)
df = df.dropna()
print("Shape after dropping NA:", df.shape)

print(df['dataset'].value_counts())

# Print the counts of each unique value in the 'dataset' column
#print(df['dataset'].value_counts())

# Data preprocessing 
# Create the 'target' column 
df['target'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

# Drop the original 'num' column
df = df.drop(columns=['num'])

# Rename 'target' to 'num'
df = df.rename(columns={'target': 'num'})

# Print the counts of each unique value in the 'num' column
print(df['num'].value_counts())

# Define different datasets 
Cleveland_data = df[df['dataset'] == "Cleveland"]
Cleveland_Hungary_data = df[df['dataset'].isin(["Cleveland", "Hungary"])]
all_data = df

# Define a function to train and evaluate a model
def train_and_evaluate(df, dataset_label):
    # Print the counts of each unique value in the 'dataset' column
    #print(df['dataset'].value_counts())
    #print(df.shape)

    # One-hot encode categorical variables and drop the first level
    df = pd.get_dummies(df, drop_first=True)

    # Define features and target 
    X = df.drop('num', axis=1)
    y = df['num']

    # Split data into training and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Hyperparameter tuning for SVM
    param_grid = {
        'C': [0.1, 1, 5, 10, 20],  # Regularization parameter
        'kernel': ['linear', 'sigmoid', 'poly', 'rbf'],  # Kernel types
        'gamma': [0.05, 0.1, 0.2]  # Kernel coefficient for 'rbf', 'sigmoid', 'poly'
    }

    # Use GridSearchCV to perform cross-validation and find the best parameters
    print("========")
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, verbose=1)
    grid_search.fit(X_train, y_train)

    # Get the best model from the grid search
    best_svm = grid_search.best_estimator_

    # Predict on the test set using the best model
    y_pred = best_svm.predict(X_test)

    # Evaluate the model's performance 
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {dataset_label} model: {accuracy:.2f}")
    print("Best parameters found:", grid_search.best_params_)

# Train and evaluate models for different datasets 
train_and_evaluate(Cleveland_data, 'Cleveland data')
train_and_evaluate(Cleveland_Hungary_data, 'Cleveland and Hungary data') 
train_and_evaluate(all_data, 'All data')


# Hyperparameter tuning
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

print("\nHyperparameter tuning")

# Load dataset
df = pd.read_csv("datafile/heart_disease_uci.csv")

# Check initial shape and drop any rows with missing values
print("Original shape:", df.shape)
df = df.dropna()
print("Shape after dropping NA:", df.shape)

# Print the counts of each unique value in the 'dataset' column
#print(df['dataset'].value_counts())

# Filter the dataframe
df = df[df['dataset'] == "Cleveland"]

# Plot the distribution of values in the 'num' column
plt.figure(figsize=(8, 6))
df['num'].value_counts().sort_index().plot(kind='bar')

# Add labels and title
plt.xlabel('Num Values')
plt.ylabel('Frequency')
plt.title('Distribution of num Column before data preprocessing')
plt.xticks(rotation=0)

# Show the plot
plt.show()

# Data preprocessing 
# Create the 'target' column 
df['target'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
#df.to_csv("test.csv")

# Drop the original 'num' column
df = df.drop(columns=['num'])

# Rename 'target' to 'num'
df = df.rename(columns={'target': 'num'})

# Show distribution of num column 
#print(df['num'].value_counts())

# Plot the distribution of values in the 'num' column
plt.figure(figsize=(8, 6))
df['num'].value_counts().sort_index().plot(kind='bar')

# Add labels and title
plt.xlabel('Num Values')
plt.ylabel('Frequency')
plt.title('Distribution of num Column after data preprocessing')
plt.xticks(rotation=0)

# Show the plot
plt.show()

# One-hot encode categorical variables and drop the first level
df = pd.get_dummies(df, drop_first=True)

# Define features and target 
X = df.drop('num', axis=1)
y = df['num']

# Split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning for SVM
param_grid = {
    'C': [0.1, 1, 5, 10, 20],  # Regularization parameter
    'kernel': ['linear', 'sigmoid', 'poly', 'rbf'],  # Kernel types
    'gamma': [0.05, 0.1, 0.2]  # Kernel coefficient for 'rbf', 'sigmoid', 'poly'
}

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_svm = grid_search.best_estimator_

# Predict on the test set using the best model
y_pred = best_svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of SVM with Hyperparameter Tuning: {accuracy:.2f}")
print("Best parameters found:", grid_search.best_params_)


# Plotting the histogram of 'trestbps' (Resting Blood Pressure) - drop missing data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("\nOutlier detection and histogram of 'trestbps' - drop missing data and dataset =='Cleveland' ")

# Load dataset
df = pd.read_csv("datafile/heart_disease_uci.csv")

# Drop any rows with missing values
df = df.dropna()

# Filter the dataframe
df = df[df['dataset'] == "Cleveland"]

# Box plot to visually check for outliers in the 'trestbps' column 
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['oldpeak'])  
plt.title('Box Plot for oldpeak')
plt.ylabel('oldpeak')
plt.show()

# Plotting the histogram of 'trestbps'
plt.hist(df['oldpeak'], bins=20)

# Adding a title and labels
plt.title('Histogram of oldpeak')
plt.xlabel('oldpeak')
plt.ylabel('Frequency')

# Show the plot
plt.show()


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

# Check initial shape and drop any rows with missing values
print("Original shape:", df.shape)
df = df.dropna()
print("Shape after dropping NA:", df.shape)

# Filter the dataframe
df = df[df['dataset'] == "Cleveland"]

# Print the counts of each unique value in the 'sex' column
#print(df['sex'].value_counts())

# Data preprocessing 
# Create the 'target' column 
df['target'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

# Drop the original 'num' column
df = df.drop(columns=['num'])

# Rename 'target' to 'num'
df = df.rename(columns={'target': 'num'})

# Show distribution of num column 
#print(df['num'].value_counts())

# One-hot encode categorical variables and drop the first level
df = pd.get_dummies(df, drop_first=True)

# Define features and target 
X = df.drop('num', axis=1)
y = df['num']

# Print total counts before splitting
total_male = (df['sex_Male'] == 1).sum()
total_female = (df['sex_Male'] == 0).sum()
print("\nTotal dataset distribution:")
print(f"Number of male samples: {total_male}")
print(f"Number of female samples: {total_female}")
print(f"Male percentage: {(total_male/(total_male+total_female)*100):.2f}%")
print(f"Female percentage: {(total_female/(total_male+total_female)*100):.2f}%")


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
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict on the test set  
y_pred = svm.predict(X_test)

# Calculate and print accuracy 
accuracy = accuracy_score(y_test, y_pred)
print("SVM")
print(f"Overall accuracy: {accuracy:.2f}")

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
    predictions = svm.predict(gender_test_data)
    
    #print(predictions)
    #print(gender_test_labels)
    
    # Calculate and print accuracy for each gender subset 
    gender_accuracy = accuracy_score(gender_test_labels, predictions)
    print(f"\nPerformance for {gender} subset:")
    print(f"Accuracy: {gender_accuracy:.2f}")