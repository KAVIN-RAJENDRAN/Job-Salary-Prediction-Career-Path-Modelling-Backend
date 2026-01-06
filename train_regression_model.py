# train_regression_model.py
import pandas as pd
import numpy as np
import joblib, os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

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

# === Log-transform target ===
df['log_salary'] = np.log1p(df['salary_in_usd'])

# === Features & Target ===
features = [c + '_freq' for c in cat_cols] + ['remote_ratio']
X = df[features]
y = df['log_salary']

# === Ridge Regression Model ===
ridge_pipe = Pipeline([
    ('scale', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge())
])

params = {'ridge__alpha':[0.1, 0.5, 1, 5, 10]}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grid = GridSearchCV(ridge_pipe, params, cv=5, scoring='r2')
grid.fit(X_train, y_train)

best_ridge = grid.best_estimator_

# === Evaluate ===
y_pred = np.expm1(best_ridge.predict(X_test))
y_true = np.expm1(y_test)
print("=== Ridge Regression ===")
print("Best α:", grid.best_params_)
print("R²:", round(r2_score(y_true, y_pred),3))
print("MAE:", round(mean_absolute_error(y_true, y_pred),2))

# === Save model ===
os.makedirs("backend/model", exist_ok=True)
joblib.dump(best_ridge, "backend/model/regression_model.pkl")
print("\n✅ Regression model saved successfully at backend/model/regression_model.pkl")
