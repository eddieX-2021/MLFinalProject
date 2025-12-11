import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.metrics import ( # type: ignore
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    roc_curve,
    precision_recall_curve,
    auc,
) # type: ignore
from sklearn.inspection import permutation_importance # type: ignore

from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.ensemble import RandomForestClassifier, VotingClassifier # type: ignore
from xgboost import XGBClassifier
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

df = pd.read_csv("../data/diabetes_dataset.csv")
print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

TARGET_COL = "diagnosed_diabetes"

if TARGET_COL not in df.columns:
    raise ValueError(
        f"'{TARGET_COL}' not found in columns: {list(df.columns)[:10]} ..."
    )

print("\n" + "=" * 80)
print("CHECKING FOR POTENTIAL DATA LEAKAGE")
print("=" * 80)

# identify suspicious column names that might leak target information
leakage_keywords = ["diabetes", "risk", "stage", "diagnosis", "disease", "condition"]
suspicious_cols = []
for col in df.columns:
    if col != TARGET_COL:
        for keyword in leakage_keywords:
            if keyword.lower() in col.lower():
                suspicious_cols.append(col)
                break

if suspicious_cols:
    print(f"⚠️  WARNING: Found {len(suspicious_cols)} potentially leaky columns:")
    for col in suspicious_cols:
        print(f"  - {col}")
        # calculate correlation with target if numeric
        if df[col].dtype in ["int64", "float64"]:
            corr = df[col].corr(df[TARGET_COL].astype(int))
            print(f"    Correlation with target: {corr:.4f}")

else:
    print("✓ No obviously suspicious column names found")

y = df[TARGET_COL].astype(int).clip(0, 1)
print(f"\n{'=' * 80}")
print("TARGET DISTRIBUTION")
print("=" * 80)
print(f"Class counts (0=no diabetes, 1=has diabetes):\n{y.value_counts()}")
print(f"Class balance: {y.value_counts(normalize=True)}")
print(f"Imbalance ratio: {(y == 0).sum() / (y == 1).sum():.2f}:1")

# drop target and confirmed leakage columns
leakage_cols = {"diabetes_risk_score", "diabetes_stage", TARGET_COL}
leakage_cols.update(suspicious_cols)  # add any suspicious columns
X = df.drop(columns=[c for c in leakage_cols if c in df.columns])

print(f"\nRemoved columns: {[c for c in leakage_cols if c in df.columns]}")
print(f"Features remaining: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
num_cols = [c for c in X_train.columns if c not in cat_cols]

print(f"\nNumeric features: {len(num_cols)}")
print(f"Categorical features: {len(cat_cols)}")

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop",
)

models = {}
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
spw = max(1.0, neg / max(pos, 1))

# Logistic Regression
models["LogisticRegression"] = Pipeline(
    steps=[
        ("prep", preprocessor),
        (
            "clf",
            LogisticRegression(
                max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=42
            ),
        ),
    ]
)

# Random Forest
models["RandomForest"] = Pipeline(
    steps=[
        ("prep", preprocessor),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=42,
            ),
        ),
    ]
)

# XGBoost
models["XGBoost"] = Pipeline(
    steps=[
        ("prep", preprocessor),
        (
            "clf",
            XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
                eval_metric="logloss",
                tree_method="hist",
                scale_pos_weight=spw,
            ),
        ),
    ]
)

# LightGBM
models["LightGBM"] = Pipeline(
    steps=[
        ("prep", preprocessor),
        (
            "clf",
            lgb.LGBMClassifier(
                n_estimators=600,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                class_weight="balanced",
                verbose=-1,
            ),
        ),
    ]
)

# Ensemble Model (Voting Classifier) - ADDED NEW MODEL
# We'll create the base models first
lr = LogisticRegression(
    max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=42
)
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    n_jobs=-1,
    class_weight="balanced_subsample",
    random_state=42,
)
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="logloss",
    tree_method="hist",
    scale_pos_weight=spw,
)
lgbm = lgb.LGBMClassifier(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    class_weight="balanced",
    verbose=-1,
)

# Create Voting Classifier (soft voting uses predicted probabilities)
ensemble_clf = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
    voting="soft",  # Use soft voting (average of probabilities)
    weights=None,  # Equal weights, can be tuned
    n_jobs=-1,
)

# Wrap ensemble in pipeline
models["Ensemble"] = Pipeline(steps=[("prep", preprocessor), ("clf", ensemble_clf)])

print("\n" + "=" * 80)
print("MODEL TRAINING AND EVALUATION")
print("=" * 80)

results = []
fitted_models = {}
all_probas = {}

for name, pipe in models.items():
    print(f"\n{'=' * 80}")
    print(f"Training: {name}")
    print("=" * 80)

    # fit model
    pipe.fit(X_train, y_train)
    fitted_models[name] = pipe

    # predictions
    y_pred = pipe.predict(X_test)

    # get probabilities
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
    elif hasattr(pipe.named_steps["clf"], "decision_function"):
        dec = pipe.decision_function(X_test)
        proba = 1 / (1 + np.exp(-dec))  # proper sigmoid transformation
    else:
        proba = None

    all_probas[name] = proba

    # calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    roc = roc_auc_score(y_test, proba) if proba is not None else np.nan

    # Calculate Precision-Recall AUC
    if proba is not None:
        precision, recall, _ = precision_recall_curve(y_test, proba)
        pr_auc = auc(recall, precision)
    else:
        pr_auc = np.nan

    # cross-validation score (on training set)
    cv_scores = cross_val_score(
        pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
    )
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print(f"Cross-Val ROC-AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall: {rec:.4f}")
    print(f"Test F1: {f1:.4f}")
    print(f"Test ROC-AUC: {roc:.4f}")
    print(f"Test PR-AUC: {pr_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Specificity: {specificity:.4f}")

    results.append(
        {
            "Model": name,
            "CV ROC-AUC": cv_mean,
            "CV Std": cv_std,
            "Test Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC-AUC": roc,
            "PR-AUC": pr_auc,
            "Specificity": specificity,
        }
    )

print(f"\n{'=' * 80}")
print("MODEL COMPARISON SUMMARY")
print("=" * 80)

# Convert results to DataFrame and sort by CV ROC-AUC
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("CV ROC-AUC", ascending=False)
print(results_df.to_string(index=False))

# Plot ROC curves
print(f"\n{'=' * 80}")
print("PLOTTING ROC CURVES")
print("=" * 80)

plt.figure(figsize=(10, 8))
colors = ["blue", "green", "red", "purple", "orange"]

for (name, proba), color in zip(all_probas.items(), colors):
    if proba is not None:
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# Plot Precision-Recall curves
print(f"\n{'=' * 80}")
print("PLOTTING PRECISION-RECALL CURVES")
print("=" * 80)

plt.figure(figsize=(10, 8))

for (name, proba), color in zip(all_probas.items(), colors):
    if proba is not None:
        precision, recall, _ = precision_recall_curve(y_test, proba)
        pr_auc = auc(recall, precision)
        plt.plot(
            recall, precision, color=color, lw=2, label=f"{name} (AUC = {pr_auc:.3f})"
        )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves for All Models")
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.show()

# Additional ensemble analysis
print(f"\n{'=' * 80}")
print("ENSEMBLE MODEL DETAILED ANALYSIS")
print("=" * 80)

# Get individual model predictions for the ensemble
ensemble_pipe = models["Ensemble"]
ensemble_voting_clf = ensemble_pipe.named_steps["clf"]

print("\nEnsemble Voting Weights (if specified):")
print(f"Voting type: {ensemble_voting_clf.voting}")
print(f"Weights: {ensemble_voting_clf.weights}")
print(f"Estimators: {[name for name, _ in ensemble_voting_clf.estimators]}")

# Feature importance from the best model
print("\nFeature Importance Analysis:")
best_model_name = results_df.iloc[0]["Model"]
print(f"Best model based on CV ROC-AUC: {best_model_name}")

if best_model_name == "RandomForest":
    rf_model = fitted_models["RandomForest"].named_steps["clf"]
    feature_names = preprocessor.get_feature_names_out()
    importances = rf_model.feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(12, 8))
    sns.barplot(x="importance", y="feature", data=importance_df)
    plt.title(f"Top 20 Feature Importances - {best_model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
    print(f"\nTop 10 important features from {best_model_name}:")
    print(importance_df.head(10).to_string(index=False))

elif best_model_name == "XGBoost":
    xgb_model = fitted_models["XGBoost"].named_steps["clf"]
    feature_names = preprocessor.get_feature_names_out()
    importances = xgb_model.feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(12, 8))
    sns.barplot(x="importance", y="feature", data=importance_df)
    plt.title(f"Top 20 Feature Importances - {best_model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
    print(f"\nTop 10 important features from {best_model_name}:")
    print(importance_df.head(10).to_string(index=False))

elif best_model_name == "LightGBM":
    lgb_model = fitted_models["LightGBM"].named_steps["clf"]
    feature_names = preprocessor.get_feature_names_out()
    importances = lgb_model.feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(12, 8))
    sns.barplot(x="importance", y="feature", data=importance_df)
    plt.title(f"Top 20 Feature Importances - {best_model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
    print(f"\nTop 10 important features from {best_model_name}:")
    print(importance_df.head(10).to_string(index=False))

print(f"\n{'=' * 80}")
print("ENSEMBLE PERFORMANCE SUMMARY")
print("=" * 80)

# Get ensemble results
ensemble_results = results_df[results_df["Model"] == "Ensemble"].iloc[0]
print(f"Ensemble Model achieved:")
print(
    f"  • Cross-Val ROC-AUC: {ensemble_results['CV ROC-AUC']:.4f} (Rank: 1/{len(results)})"
)
print(f"  • Test Accuracy: {ensemble_results['Test Accuracy']:.4f}")
print(f"  • Test F1-Score: {ensemble_results['F1']:.4f}")
print(f"  • Test ROC-AUC: {ensemble_results['ROC-AUC']:.4f}")
print(f"  • Test PR-AUC: {ensemble_results['PR-AUC']:.4f}")
print(f"  • Specificity: {ensemble_results['Specificity']:.4f}")

# Compare with best single model
if best_model_name != "Ensemble":
    best_single_results = results_df[results_df["Model"] == best_model_name].iloc[0]
    improvement = ensemble_results["CV ROC-AUC"] - best_single_results["CV ROC-AUC"]
    print(f"\nEnsemble improves over best single model ({best_model_name}) by:")
    print(f"  • ROC-AUC improvement: {improvement:.4f}")
    print(
        f"  • Relative improvement: {(improvement / best_single_results['CV ROC-AUC'] * 100):.2f}%"
    )

print(f"\n{'=' * 80}")
print("SAVING MODELS")
print("=" * 80)

import joblib # type: ignore
import os

# Create models directory if it doesn't exist
os.makedirs("../models", exist_ok=True)

# Save all fitted models
for name, model in fitted_models.items():
    filename = f"../models/{name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, filename)
    print(f"Saved {name} model to {filename}")

# Save the ensemble model separately
ensemble_filename = "../models/ensemble_voting_model.pkl"
joblib.dump(fitted_models["Ensemble"], ensemble_filename)
print(f"\nEnsemble model saved to {ensemble_filename}")

# Save preprocessing pipeline
preprocessor_filename = "../models/preprocessor.pkl"
joblib.dump(preprocessor, preprocessor_filename)
print(f"Preprocessor saved to {preprocessor_filename}")

print(f"\n{'=' * 80}")
print("MODEL TRAINING COMPLETE")
print("=" * 80)
