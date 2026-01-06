# backend/train_classification_model.py
import pandas as pd
import numpy as np
import joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# === Load Dataset ===
df = pd.read_csv("data/ds_salaries.csv").dropna()
df.drop_duplicates(inplace=True)

# === Preprocess job titles ===
df['job_title'] = df['job_title'].str.lower()
def simplify(title):
    for kw in ['scientist','engineer','analyst','ml','machine']:
        if kw in title: return kw
    if 'manager' in title: return 'manager'
    return 'other'
df['job_title'] = df['job_title'].apply(simplify)

# === Frequency Encoding ===
cat_cols = ['experience_level','employment_type','job_title',
            'company_size','company_location','employee_residence']
for col in cat_cols:
    freq = df[col].value_counts(normalize=True)
    df[col + '_freq'] = df[col].map(freq)

# === Salary Category ===
q1, q2 = df['salary_in_usd'].quantile([0.33, 0.66])
def categorize(s):  
    
    if s <= q1: return "Low"
    elif s <= q2: return "Medium"
    else: return "High"
df['salary_category'] = df['salary_in_usd'].apply(categorize)

# === Features & Target ===
features = [c + '_freq' for c in cat_cols] + ['remote_ratio']
X = df[features]
y = df['salary_category']

# === Classification Models ===
log_pipe = Pipeline([
    ('scale', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=2000, C=5))
])
svm_pipe = Pipeline([
    ('scale', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=5, gamma='auto'))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
log_pipe.fit(X_train, y_train)
pred_log = log_pipe.predict(X_test)
log_acc = accuracy_score(y_test, pred_log)
log_f1 = f1_score(y_test, pred_log, average='weighted')

# SVM
svm_pipe.fit(X_train, y_train)
pred_svm = svm_pipe.predict(X_test)
svm_acc = accuracy_score(y_test, pred_svm)
svm_f1 = f1_score(y_test, pred_svm, average='weighted')

print("=== Logistic Regression ===")
print(f"Accuracy: {log_acc:.3f} | F1: {log_f1:.3f}")
print("\n=== SVM ===")
print(f"Accuracy: {svm_acc:.3f} | F1: {svm_f1:.3f}")

# === Choose Best ===
best_clf = log_pipe if log_acc >= svm_acc else svm_pipe

# === Save model ===
os.makedirs("model", exist_ok=True)
joblib.dump(best_clf, "model/classification_model.pkl")
print("\n✅ Classification model saved at backend/model/classification_model.pkl")
