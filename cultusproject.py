# churn_pipeline.py
# Self-contained churn modeling pipeline that runs even without a provided CSV.
# Save as churn_pipeline.py and run: python churn_pipeline.py

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import matplotlib.pyplot as plt
import joblib
import json
import random

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def generate_synthetic_telecom(n_samples=2000, random_state=42):
    np.random.seed(random_state)
    df = pd.DataFrame()
    df['customerID'] = [f"C{100000+i}" for i in range(n_samples)]
    df['gender'] = np.random.choice(['Male','Female'], size=n_samples)
    df['SeniorCitizen'] = np.random.choice([0,1], p=[0.84,0.16], size=n_samples)
    df['Partner'] = np.random.choice(['Yes','No'], size=n_samples, p=[0.47,0.53])
    df['Dependents'] = np.random.choice(['Yes','No'], size=n_samples, p=[0.27,0.73])
    df['tenure'] = np.clip(np.random.exponential(scale=24, size=n_samples).astype(int), 0, 72)
    df['PhoneService'] = np.random.choice(['Yes','No'], size=n_samples, p=[0.9,0.1])
    df['MultipleLines'] = np.random.choice(['No phone service','No','Yes'], size=n_samples, p=[0.05,0.55,0.4])
    df['InternetService'] = np.random.choice(['DSL','Fiber optic','No'], size=n_samples, p=[0.4,0.45,0.15])
    df['OnlineSecurity'] = np.random.choice(['Yes','No','No internet service'], size=n_samples, p=[0.18,0.67,0.15])
    df['OnlineBackup'] = np.random.choice(['Yes','No','No internet service'], size=n_samples, p=[0.22,0.63,0.15])
    df['DeviceProtection'] = np.random.choice(['Yes','No','No internet service'], size=n_samples, p=[0.18,0.67,0.15])
    df['TechSupport'] = np.random.choice(['Yes','No','No internet service'], size=n_samples, p=[0.15,0.7,0.15])
    df['StreamingTV'] = np.random.choice(['Yes','No','No internet service'], size=n_samples, p=[0.28,0.57,0.15])
    df['StreamingMovies'] = np.random.choice(['Yes','No','No internet service'], size=n_samples, p=[0.28,0.57,0.15])
    df['Contract'] = np.random.choice(['Month-to-month','One year','Two year'], size=n_samples, p=[0.55,0.25,0.20])
    df['PaperlessBilling'] = np.random.choice(['Yes','No'], size=n_samples, p=[0.6,0.4])
    df['PaymentMethod'] = np.random.choice(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'], size=n_samples)
    df['MonthlyCharges'] = np.round(20 + np.random.rand(n_samples)*90 + (df['InternetService']=='Fiber optic')*10,2)
    df['TotalCharges'] = np.round(df['MonthlyCharges'] * (df['tenure'] + np.random.rand(n_samples)),2)
    # churn probability influenced by tenure, contract, monthly charges
    churn_score = (
        0.35*(df['Contract']=='Month-to-month').astype(float) +
        0.2*(df['SeniorCitizen']==1).astype(float) +
        0.25*(df['PaperlessBilling']=='Yes').astype(float) +
        0.0008*(df['MonthlyCharges']) +
        -0.007*(df['tenure'])
    )
    churn_prob = 1/(1+np.exp(-(-1.5 + churn_score*2.5)))
    df['Churn'] = np.where(np.random.rand(n_samples) < churn_prob, 'Yes', 'No')
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

def load_data():
    for fname in ("data.csv", "telecom_churn.csv", "telecom.csv"):
        if os.path.exists(fname):
            print(f"Loading dataset from {fname}")
            return pd.read_csv(fname)
    print("No dataset found. Generating synthetic dataset.")
    return generate_synthetic_telecom(n_samples=2500, random_state=RANDOM_STATE)

df = load_data()
print("Dataset shape:", df.shape)

# detect target
target_col = None
for cand in ['Churn','churn','TARGET','Exited','is_churn']:
    if cand in df.columns:
        target_col = cand
        break
if target_col is None:
    df['Churn'] = np.where(df['tenure'] < 6, 'Yes','No')
    target_col = 'Churn'

df[target_col] = df[target_col].astype(str).map({'Yes':1,'No':0})

ignore_cols = ['customerID'] if 'customerID' in df.columns else []
feature_candidates = [c for c in df.columns if c not in ignore_cols + [target_col]]
feature_candidates = feature_candidates[:40]

X = df[feature_candidates].copy()
y = df[target_col].astype(int).copy()

numeric_feats = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_feats = [c for c in X.columns if c not in numeric_feats]

numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_feats), ('cat', categorical_transformer, categorical_feats)])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE)

models = {
    'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    'HistGB': HistGradientBoostingClassifier(random_state=RANDOM_STATE)
}

param_distributions = {
    'RandomForest': {
        'model__n_estimators': [100, 200, 400],
        'model__max_depth': [5, 8, 12, None],
        'model__min_samples_leaf': [1,2,4],
    },
    'HistGB': {
        'model__max_iter': [100,200,400],
        'model__max_leaf_nodes': [15,31,63,None],
        'model__learning_rate': [0.01,0.05,0.1,0.2]
    }
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
search_results = {}

for name, estimator in models.items():
    pipe = Pipeline([('pre', preprocessor), ('model', estimator)])
    print(f"Tuning {name} ...")
    rs = RandomizedSearchCV(pipe, param_distributions[name], n_iter=6, scoring='roc_auc',
                            n_jobs=-1, cv=cv, random_state=RANDOM_STATE, verbose=0)
    rs.fit(X_train, y_train)
    print(f"Best {name} params:", rs.best_params_)
    print(f"Best CV AUC for {name}: {rs.best_score_:.4f}")
    search_results[name] = rs

best_name = max(search_results.keys(), key=lambda n: search_results[n].best_score_)
best_search = search_results[best_name]
print(f"Selected best model: {best_name} with CV AUC {best_search.best_score_:.4f}")

best_model = best_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_test)[:,1]
test_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test ROC AUC: {test_auc:.4f}")

out_dir = './churn_model_outputs'
os.makedirs(out_dir, exist_ok=True)
joblib.dump(best_model, os.path.join(out_dir, 'best_model.joblib'))
with open(os.path.join(out_dir, 'metadata.json'),'w') as f:
    json.dump({'selected_model': best_name, 'cv_auc': float(best_search.best_score_), 'test_auc': float(test_auc)}, f)

pre = best_model.named_steps['pre']
numeric_feats_after = numeric_feats
feature_names = list(numeric_feats_after)
if len(categorical_feats)>0:
    ohe = pre.named_transformers_['cat'].named_steps['onehot']
    cat_names = ohe.get_feature_names_out(categorical_feats).tolist()
    feature_names.extend(cat_names)

# model importances if available
model_obj = best_model.named_steps['model']
if hasattr(model_obj, 'feature_importances_'):
    try:
        fi = model_obj.feature_importances_
        fi_df = pd.DataFrame({'feature': feature_names, 'importance': fi}).sort_values('importance', ascending=False)
        fi_df.to_csv(os.path.join(out_dir,'feature_importances_model.csv'), index=False)
        print("Saved model-based feature importances.")
    except Exception as e:
        print("Could not extract model.feature_importances_:", e)

perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
perm_df = pd.DataFrame({'feature': feature_names, 'perm_mean': perm.importances_mean, 'perm_std': perm.importances_std}).sort_values('perm_mean', ascending=False)
perm_df.to_csv(os.path.join(out_dir,'feature_importances_permutation.csv'), index=False)
print("Saved permutation importances.")

# Try SHAP
have_shap = False
try:
    import shap
    have_shap = True
    print("SHAP available - computing SHAP summary (may take time).")
    X_test_trans = best_model.named_steps['pre'].transform(X_test)
    model_for_shap = best_model.named_steps['model']
    explainer = shap.Explainer(model_for_shap, masker=shap.maskers.Independent(X_test_trans))
    shap_values = explainer(X_test_trans)
    shap.summary_plot(shap_values, features=X_test_trans, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'shap_summary.png'), dpi=150)
    plt.close()
    print("Saved SHAP summary.")
except Exception as e:
    print("SHAP not available or failed:", e)
    have_shap = False

top_features = perm_df['feature'].tolist()[:6]
with open(os.path.join(out_dir,'top_features.json'),'w') as f:
    json.dump(top_features[:10], f)

# PDPs/fallback plots for top 3 features
def plot_pdp_for_feature(f_name, savepath):
    try:
        fig, ax = plt.subplots(figsize=(6,4))
        # If original feature exists use PDP on raw name
        if f_name in X_test.columns:
            PartialDependenceDisplay.from_estimator(best_model, X_test, [f_name], ax=ax)
        else:
            # fallback: use transformed column index if available
            idxs = [i for i,fn in enumerate(feature_names) if fn==f_name or f_name in fn]
            if not idxs:
                print("PDP: could not find preprocessed feature for", f_name)
                return
            X_trans = best_model.named_steps['pre'].transform(X_test)
            col = X_trans[:,idxs[0]]
            bins = pd.qcut(col, q=10, duplicates='drop')
            df_tmp = pd.DataFrame({'bin': bins, 'pred': best_model.predict_proba(X_test)[:,1]})
            agg = df_tmp.groupby('bin')['pred'].mean()
            agg.plot(ax=ax)
            ax.set_title(f"PDP-like for {f_name}")
        plt.tight_layout()
        plt.savefig(savepath, dpi=150)
        plt.close()
        print("Saved PDP to", savepath)
    except Exception as e:
        print("Failed PDP for", f_name, ":", e)

pdp_dir = os.path.join(out_dir, 'pdps')
os.makedirs(pdp_dir, exist_ok=True)
for i, f in enumerate(top_features[:3]):
    fname = os.path.join(pdp_dir, f"pdp_{i+1}_{f.replace(' ','_').replace('/','_')}.png")
    plot_pdp_for_feature(f, fname)

# Five sample profiles
profiles = X_test.sample(n=5, random_state=RANDOM_STATE).reset_index(drop=True)
profiles_preds = best_model.predict_proba(profiles)[:,1]
profiles_results = profiles.copy()
profiles_results['pred_proba'] = profiles_preds.round(4)
profiles_results['pred_label'] = (profiles_results['pred_proba'] > 0.5).astype(int)
profiles_results.to_csv(os.path.join(out_dir, 'sample_profiles_predictions.csv'), index=False)
print("Saved 5 sample profiles predictions.")

# Local explanations if SHAP present
if have_shap:
    try:
        profiles_trans = best_model.named_steps['pre'].transform(profiles)
        expl = shap.Explainer(best_model.named_steps['model'], masker=shap.maskers.Independent(X_test_trans))
        sv = expl(profiles_trans)
        for i in range(len(profiles)):
            shap.plots.bar(sv[i], show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"profile_shap_{i+1}.png"), dpi=150)
            plt.close()
        print("Saved SHAP local explanations for profiles.")
    except Exception as e:
        print("Failed to produce SHAP local explanations:", e)

# Classification report
y_pred = (y_pred_proba > 0.5).astype(int)
print("\nClassification report (threshold 0.5):")
print(classification_report(y_test, y_pred))

with open(os.path.join(out_dir,'summary.txt'),'w') as f:
    f.write(f"Selected model: {best_name}\\nCV AUC: {best_search.best_score_:.4f}\\nTest AUC: {test_auc:.4f}\\nTop features:\\n")
    for feat in top_features[:10]:
        f.write(f"- {feat}\\n")
print("All outputs saved to:", out_dir)
