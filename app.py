# backend/app.py
import os, joblib, pandas as pd, numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "ds_salaries.csv")
REG_MODEL_PATH = os.path.join(BASE_DIR, "model", "regression_model.pkl")
CLF_MODEL_PATH = os.path.join(BASE_DIR, "model", "classification_model.pkl")

app = Flask(__name__)
CORS(app)

# Load models
if not os.path.exists(REG_MODEL_PATH):
    raise FileNotFoundError("Regression model not found. Run train_regression_model.py first.")
if not os.path.exists(CLF_MODEL_PATH):
    raise FileNotFoundError("Classification model not found. Run train_classification_model.py first.")

reg_model = joblib.load(REG_MODEL_PATH)
clf_model = joblib.load(CLF_MODEL_PATH)

# Load original CSV to build frequency maps (must be same file used for training)
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Data CSV not found. Put ds_salaries.csv in backend/data/")
df_full = pd.read_csv(DATA_PATH).dropna()
df_full.drop_duplicates(inplace=True)
df_full['job_title'] = df_full['job_title'].astype(str).str.lower()
def simplify_job_title(title):
    if title is None: return "other"
    s = str(title).lower()
    for kw in ['scientist','engineer','analyst','ml','machine']:
        if kw in s: return kw
    if 'manager' in s: return 'manager'
    return 'other'
df_full['job_title'] = df_full['job_title'].apply(simplify_job_title)

cat_cols = ['experience_level','employment_type','job_title','company_size','company_location','employee_residence']
freq_maps = {}
for col in cat_cols:
    freq_maps[col] = df_full[col].value_counts(normalize=True).to_dict()

def build_features(payload):
    job_title_raw = payload.get("job_title","")
    job_title_s = simplify_job_title(job_title_raw)
    mapping = {
        "experience_level": payload.get("experience_level",""),
        "employment_type": payload.get("employment_type",""),
        "job_title": job_title_s,
        "company_size": payload.get("company_size",""),
        "company_location": payload.get("company_location",""),
        "employee_residence": payload.get("employee_residence","")
    }
    features = []
    for col in cat_cols:
        val = mapping.get(col,"")
        freq = freq_maps.get(col, {}).get(val, 0.0)
        features.append(freq)
    # remote_ratio numeric
    try:
        rr = float(payload.get("remote_ratio", 0))
    except:
        rr = 0.0
    features.append(rr)
    return np.array([features], dtype=float)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Input JSON:
      experience_level, employment_type, job_title, company_size,
      company_location, employee_residence, remote_ratio
    Output: predicted_salary_usd (numeric), salary_category
    """
    try:
        data = request.get_json(force=True)
        X = build_features(data)
        # regression: reg_model predicts log_salary (because training used np.log1p)
        log_salary = reg_model.predict(X)[0]
        salary = float(np.expm1(log_salary))
        category = str(clf_model.predict(X)[0])
        return jsonify({"predicted_salary_usd": round(salary,2), "salary_category": category})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/career-path", methods=["POST"])
def career_path():
    """
    Input JSON:
      current_salary (optional) - numeric. If not provided, user can pass in a predicted salary from /predict
      years (int, default 5)
      annual_growth (float, default 0.12)
    Output: sequence of year-level-salary
    """
    try:
        data = request.get_json(force=True)
        current_salary = float(data.get("current_salary", 100000))
        years = int(data.get("years", 5))
        annual_growth = float(data.get("annual_growth", 0.12))
        levels = ["Entry", "Mid", "Senior", "Executive"]
        seq = []
        salary = current_salary
        for year in range(1, years+1):
            salary = salary * (1 + annual_growth)
            level = levels[min((year-1)//2, len(levels)-1)]
            seq.append({"year": year, "level": level, "salary": round(salary,2)})
        return jsonify({"sequence": seq})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting backend server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
