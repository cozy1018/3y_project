# SVM, LR, NB, Bagging and Fuzzy (Without smote) [optimal]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, VotingClassifier

###############################################
# PART 1: SVM, LR, NB and Bagging
###############################################

# Load and preprocess data for train-test experiments
def load_preprocess_data(csv_file, random_state):
    df = pd.read_csv(csv_file)
    df = df[(df['dataset'] == "Cleveland") & (df['age'] != 28)]
    df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1).astype('str')
    df = pd.get_dummies(df.dropna(), drop_first=True)
    X = df.drop(['num_1', 'id'], axis=1)
    y = df['num_1']
    return train_test_split(X, y, test_size=0.2, random_state=random_state)

# Scale features
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

# Grid search helper
def grid_search(model, param_grid, X_train, y_train):
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

# Evaluate and display model performance
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test)).plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()
    return tn, fp, fn, tp, acc
    
# Fuzzy ranking functions
def generateRank1(score, class_no):
    rank = np.zeros([class_no, 1])
    for i in range(class_no):
        rank[i] = 1 - np.exp(-((score[i] - 1) ** 2) / 2.0)
    return rank

def generateRank2(score, class_no):
    rank = np.zeros([class_no, 1])
    for i in range(class_no):
        rank[i] = 1 - np.tanh(((score[i] - 1) ** 2) / 2)
    return rank

def doFusion(res1, res2, res3, labels, class_no):
    fused_preds = []
    cnt = 0
    for i in range(len(res1)):
        rank1 = generateRank1(res1[i], class_no) * generateRank2(res1[i], class_no)
        rank2 = generateRank1(res2[i], class_no) * generateRank2(res2[i], class_no)
        rank3 = generateRank1(res3[i], class_no) * generateRank2(res3[i], class_no)
        rankSum = rank1 + rank2 + rank3
        scoreSum = 1 - (res1[i] + res2[i] + res3[i]) / 3
        # fusedScore = (rankSum.T) * scoreSum  # Not used in decision-making below
        cls = np.argmin(rankSum)
        if cls < class_no and cls == labels[i]:
            cnt += 1
        fused_preds.append(cls)
    #print("Fusion accuracy:", cnt / len(res1))
    return fused_preds
    
# Initialise a dictionary to store the metrics for each model and random_state
metrics = {
    'random_state': [],
    'svm': [],
    'lr': [],
    'nb': [],
    'bagging': [],
    'fuzzy': []
}

# Initialise an empty list to store DataFrames for each iteration
df_list = []

# Loop through random_state from 40 to 45
for random_state in range(40, 45):
        
    # Load data and split
    X_train, X_test, y_train, y_test = load_preprocess_data("datafile/heart_disease_uci.csv", random_state)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    # SVM Grid Search
    svm_params = {
        'C': [0.1,1.095,2.09,3.085,4.08,5.075,6.07,7.065,8.06,9.055,10.05,11.045,12.04,13.035,14.03,15.025,16.02,17.015,18.01,19.005,20.0],
        'kernel': ['linear', 'sigmoid', 'poly', 'rbf'],
        'gamma': [0.05, 0.0875, 0.125, 0.1625, 0.2]
    }
    best_svm, svm_best_params = grid_search(SVC(random_state=random_state), svm_params, X_train_scaled, y_train)
    best_svm_1, svm_best_params_1 = grid_search(SVC(probability=True, random_state=random_state), svm_params, X_train_scaled, y_train)
    tn_svm, fp_svm, fn_svm, tp_svm, acc_svm = evaluate_model(best_svm, X_test_scaled, y_test, "SVM")
    print(f"SVM Accuracy: {acc_svm:.2f}")
    
    # Logistic Regression Grid Search
    lr_params = [
        {'penalty': ['l1','l2'], 'C': [0.1,1.095,2.09,3.085,4.08,5.075,6.07,7.065,8.06,9.055,10.05,11.045,12.04,13.035,14.03,15.025,16.02,17.015,18.01,19.005,20.0], 'solver': ['liblinear']},
        {'penalty': ['l2'], 'C': [0.1,1.095,2.09,3.085,4.08,5.075,6.07,7.065,8.06,9.055,10.05,11.045,12.04,13.035,14.03,15.025,16.02,17.015,18.01,19.005,20.0], 'solver': ['lbfgs', 'sag']}
    ]
    best_lr, lr_best_params = grid_search(LogisticRegression(max_iter=1000, random_state=random_state), lr_params, X_train_scaled, y_train)
    tn_lr, fp_lr, fn_lr, tp_lr, acc_lr = evaluate_model(best_lr, X_test_scaled, y_test, "Logistic Regression")
    print(f"LR Accuracy: {acc_lr:.2f}")

    # Naïve Bayes Grid Search
    nb_params = {
        'var_smoothing': [10**(-i) for i in range(12, 6, -1)]  # 1e-12 to 1e-6
    }
    best_nb, nb_best_params = grid_search(GaussianNB(), nb_params, X_train_scaled, y_train)
    tn_nb, fp_nb, fn_nb, tp_nb, acc_nb = evaluate_model(best_nb, X_test_scaled, y_test, "Naïve Bayes")
    print(f"NB Accuracy: {acc_nb:.2f}")
    
    # Voting Classifier for Bagging
    voting_clf = VotingClassifier(estimators=[('lr', best_lr), ('svm', best_svm), ('nb', best_nb)], voting='hard')
    
    # Grid search for BaggingClassifier
    bagging_params = {
        'n_estimators': [100, 150, 200],
        'max_samples': [0.8, 0.9, 1.0]
    }
    best_bagging, bagging_best_params = grid_search(
        BaggingClassifier(estimator=voting_clf, random_state=random_state),
        bagging_params,
        X_train_scaled,
        y_train
    )
    tn_bagging, fp_bagging, fn_bagging, tp_bagging, acc_bagging = evaluate_model(best_bagging, X_test_scaled, y_test, "Optimized Bagging (SVM, LR, NB)")
    print(f"Bagging Accuracy: {acc_bagging:.2f}")
    print("Best parameters for Bagging:", bagging_best_params)

###############################################
# PART 2: Fuzzy Rank–Based Ensemble 
###############################################

    # Load and preprocess data for fuzzy ensemble (same filtering)
    df = pd.read_csv("datafile/heart_disease_uci.csv")
    df = df[(df['dataset'] == "Cleveland") & (df['age'] != 28)]
    df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1).astype('str')
    df = pd.get_dummies(df.dropna(), drop_first=True)
    X = df.drop(['num_1', 'id'], axis=1)
    y = df['num_1']
    
    # Set up 5-fold Stratified CV and define models
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    num_of_classes = 2
    svm_model = best_svm_1
    lr_model = best_lr
    nb_model = best_nb
                                  
    # Lists to collect predictions and true labels
    pred_svm, pred_lr, pred_nb, actual = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
        scaler = StandardScaler()
        X_train_cv_scaled = scaler.fit_transform(X_train_cv)
        X_test_cv_scaled = scaler.transform(X_test_cv)
        
        # Train each model
        svm_model.fit(X_train_cv_scaled, y_train_cv)
        lr_model.fit(X_train_cv_scaled, y_train_cv)
        nb_model.fit(X_train_cv_scaled, y_train_cv)
        
        # Get probability predictions
        pred_svm.append(svm_model.predict_proba(X_test_cv_scaled))
        pred_lr.append(lr_model.predict_proba(X_test_cv_scaled))
        pred_nb.append(nb_model.predict_proba(X_test_cv_scaled))
        actual.append(y_test_cv)
    
    # Concatenate predictions and actual labels from all folds
    pred_svm = np.concatenate(pred_svm, axis=0)
    pred_lr = np.concatenate(pred_lr, axis=0)
    pred_nb = np.concatenate(pred_nb, axis=0)
    actual = np.concatenate(actual, axis=0)

    # Apply fuzzy rank–based ensemble fusion
    ensemble_preds = doFusion(pred_svm, pred_lr, pred_nb, actual, num_of_classes)
    acc_fuzzy = accuracy_score(actual, ensemble_preds)
    cm = confusion_matrix(actual, ensemble_preds)
    tn_fuzzy, fp_fuzzy, fn_fuzzy, tp_fuzzy = cm.ravel()
    
    print(f"Accuracy of Ensemble Model (Fuzzy Rank–Based): {acc_fuzzy:.4f}")

    # Store results for the current random_state
    df = pd.DataFrame({
    'svm': [tn_svm, fp_svm, fn_svm, tp_svm, round(acc_svm, 2), random_state],
    'lr': [tn_lr, fp_lr, fn_lr, tp_lr, round(acc_lr, 2), random_state],
    'nb': [tn_nb, fp_nb, fn_nb, tp_nb, round(acc_nb, 2), random_state],
    'bagging': [tn_bagging, fp_bagging, fn_bagging, tp_bagging, round(acc_bagging, 2), random_state],
    'fuzzy': [tn_fuzzy, fp_fuzzy, fn_fuzzy, tp_fuzzy, round(acc_fuzzy, 2), random_state]
    })

    # Set custom row names (index)
    row_names = ['tn', 'fp', 'fn', 'tp', 'acc', 'random_state']
    df.index = row_names

    # Append the DataFrame for this iteration to the list
    df_list.append(df)

# After the loop, concatenate all DataFrames horizontally (along columns)
final_df = pd.concat(df_list, axis=1)

# Display the final DataFrame
print(final_df)

# Save the final DataFrame to a CSV file
final_df.to_csv('results_without_smote_optimal.csv')