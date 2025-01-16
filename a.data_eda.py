#####################
##################### Feature importance [for all features]
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("datafile/heart_disease_uci.csv")

print(df.shape)
df = df.dropna()
print(df.shape)

scaler = StandardScaler()
df[['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']] = scaler.fit_transform(df[['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']])

# Get categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

label_encoder = LabelEncoder()

for column in categorical_columns:
    # Convert categorical data into numerical
    df[column] = label_encoder.fit_transform(df[column])

X = df.drop('num', axis=1)
y = df['num']

# Random Forest model for feature importance
rf = RandomForestRegressor()
rf.fit(X, y)

# Get feature importance from the trained Random Forest
importances = rf.feature_importances_

# Create a DataFrame to hold the feature names and their corresponding importance scores
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

# Sort the DataFrame by importance in descending order
sorted_feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Display sorted feature importance
print(sorted_feature_importance)    
    
     
#####################
##################### Feature importance [for numerical data]
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("heart_disease_uci.csv")
print(df.shape)
df = df.dropna()
print(df.shape)

# Get numerical columns
numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

# Get categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

print(numerical_columns)
print(categorical_columns)

# Step 1: Calculate the correlation matrix
correlation_matrix = df[numerical_columns].corr()

# Step 2: Plot the heatmap
plt.figure(figsize=(8, 6))  # Set the size of the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Title of the heatmap
plt.title('Correlation Heatmap')

# Show the heatmap
plt.show()



#####################
##################### EDA
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# Load the dataset from a CSV file into a DataFrame
df = pd.read_csv("heart_disease_uci.csv")

# Print the shape of the DataFrame (number of rows and columns)
print(df.shape)

# Display the first 5 rows of the DataFrame 
print(df.head())

# Print the column names of the DataFrame
print(df.columns)

# Display information about the DataFrame
print(df.info())

# Check missing values 
null_values = df.isnull().sum()
print(null_values)

# Check the minimum and maximum values in the column 
print(df['age'].min(), df['age'].max())

# Plot a distribution to see the distribution of ages in the dataset  
sns.histplot(df['age'])

# Showing the distribution of age with colours distinguishing between different values in the 'sex' column
fig = px.histogram(data_frame=df, x='age', color='sex')
fig.show()

### Check for duplicate rows [?]

# Count the occurrences of each value in the 'sex' column
sex_counts = df['sex'].value_counts()
print(sex_counts)

# Plot a pie chart
fig = px.pie(df, names = 'sex', color = 'sex')
fig.show()

# x-axis represents the 'dataset' column values
# bars colored by the values in the 'sex' column 
fig = px.bar(df, x='dataset', color='sex')
fig.show()
print(df.groupby('sex')['dataset'].value_counts())

# Create a count plot to show the frequency of each 'cp' value, colored by 'num'
plt.figure(figsize=(10, 6))
sns.countplot(x='cp', hue='num', data=df)

# Set title and labels for better readability
plt.xlabel('Chest Pain Type')
plt.ylabel('Amount')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# Create a count plot to show the distribution of the 'num' column
plt.figure(figsize=(8, 6))
sns.countplot(x='num', data=df)
plt.title('Distribution of num')
plt.xlabel('num')
plt.ylabel('Amount')
plt.show()


#####################
#####################
##################### Data preprocessing
##################### Outlier Detection: We aim to remove any outliers that are not reasonable. 
##################### trestbps = 0 could be an outlier
# Box plot to visually check for outliers in the 'trestbps' column 
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['trestbps'])  
plt.title('Box Plot for trestbps (Resting Blood Pressure)')
plt.ylabel('trestbps (Resting Blood Pressure)')
plt.show()


#####################
#####################
##################### Hyper-parameter tuning
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("heart_disease_uci.csv")

print(df.shape)
df = df.dropna()
print(df.shape)

scaler = StandardScaler()
df[['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']] = scaler.fit_transform(df[['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']])


df = pd.get_dummies(df, drop_first=True)

print(df.head(5))

"""# **Logistic Regression**"""

from sklearn.model_selection import train_test_split

X = df.drop('num', axis=1)
y = df['num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape[0]

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

"""#**KNN**"""
"""
from sklearn.neighbors import KNeighborsClassifier

knn1 = KNeighborsClassifier(n_neighbors=2)

knn1.fit(X, y)

knn1.predict(X)

k = 5

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap= 'Blues')
plt.title('KNN Confusion Matrix')
plt.show()
"""

"""#**SVM**"""

from sklearn.svm import SVC

X_trainsvm, X_testsvm, y_trainsvm, y_testsvm = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel='linear')
svm.fit(X_trainsvm, y_trainsvm)

y_pred = svm.predict(X_testsvm)

accuracy = accuracy_score(y_testsvm, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Define the parameter grid
param_grid = {
    'degree': [2, 3, 4],  # degree of polynomial 
}

# Create the SVC model
svc = SVC()

# Set up the GridSearchCV
grid_search = GridSearchCV(svc, param_grid, cv=5)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print the best parameters and the corresponding score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validation Score:", grid_search.best_score_)

# Evaluate on the test set
test_score = grid_search.score(X_test, y_test)
print("Test Set Score:", test_score)


#####################
##################### Data preprocessing 
##################### Replace the missing data 
##################### Replace the missing categorical data with the most frequent value
##################### Replace the missing numerical data with the mean value
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

df = pd.read_csv("heart_disease_uci.csv")
print(df.shape)

for column in df.columns:
    if df[column].dtype == 'object':
        imputer = SimpleImputer(strategy='most_frequent')
        df[column]= imputer.fit_transform(df[[column]]).ravel()
    else:
        if df[column].isnull().sum() > 0:
            imputer_num = SimpleImputer(strategy='mean')
            df[column] = imputer_num.fit_transform(df[[column]]).ravel()

scaler = StandardScaler()
df[['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']] = scaler.fit_transform(df[['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']])

df = pd.get_dummies(df, drop_first=True)

print(df.head(5))

"""# **Logistic Regression**"""

from sklearn.model_selection import train_test_split

X = df.drop('num', axis=1)
y = df['num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape[0]

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

"""#**KNN**"""
"""
from sklearn.neighbors import KNeighborsClassifier

knn1 = KNeighborsClassifier(n_neighbors=2)

knn1.fit(X, y)

knn1.predict(X)

k = 5

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap= 'Blues')
plt.title('KNN Confusion Matrix')
plt.show()
"""

"""#**SVM**"""

from sklearn.svm import SVC

X_trainsvm, X_testsvm, y_trainsvm, y_testsvm = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel='linear')
svm.fit(X_trainsvm, y_trainsvm)

y_pred = svm.predict(X_testsvm)

accuracy = accuracy_score(y_testsvm, y_pred)
print(f"Accuracy: {accuracy:.2f}")
