# =============================================================================
# Marine Bioacoustics Species Classification – FINAL VERSION (v3)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

# ── 1. Load Data ─────────────────────────────────────────────────────────────
df = pd.read_csv("SM_1_WMD_Features_and_Labels.csv")

feature_cols = [str(i) for i in range(128)]
X = df[feature_cols].copy()
y = df["SPECIES"]

# ── 2. Handle Missing Values ─────────────────────────────────────────────────
X = X.fillna(X.median())

# ── 3. Encode Labels ─────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# ── 4. Train/Test Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# ── 5. PCA Analysis (determine components) ───────────────────────────────────
scaler_temp = StandardScaler()
X_train_scaled_temp = scaler_temp.fit_transform(X_train)

pca_full = PCA()
pca_full.fit(X_train_scaled_temp)

cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.searchsorted(cumulative_var, 0.95) + 1

print(f"PCA components for 95% variance: {n_95}")

# ── 6. Define Models (with proper pipelines) ─────────────────────────────────
models = {
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1))
    ]),

    "Decision Tree": Pipeline([
        ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=42))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_95)),
        ("clf", KNeighborsClassifier())
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_95)),
        ("clf", SVC(class_weight="balanced", random_state=42))
    ]),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_95)),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs",
                                   class_weight="balanced", random_state=42))
    ])
}

# ── 7. Hyperparameter Tuning (Random Forest) ─────────────────────────────────
param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [None, 20],
    "clf__min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    models["Random Forest"],
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
print("Best RF params:", grid.best_params_)

# Replace RF with tuned version
models["Random Forest"] = best_rf

# ── 8. Cross-validation + Test Evaluation ────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"\n{name}")

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1)
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "test_acc": acc
    }

# ── 9. Best Model ────────────────────────────────────────────────────────────
best_model_name = max(results, key=lambda x: results[x]["test_acc"])
best_model = results[best_model_name]["model"]
y_pred_best = results[best_model_name]["y_pred"]

print(f"\nBest Model: {best_model_name}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=class_names))

# ── 10. Confusion Matrix ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title(f"Confusion Matrix – {best_model_name}")
plt.tight_layout()
plt.show()

# ── 11. Error Analysis ───────────────────────────────────────────────────────
print("\nCommon misclassifications:")
cm_norm = cm / cm.sum(axis=1, keepdims=True)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm_norm[i, j] > 0.1:
            print(f"{class_names[i]} → {class_names[j]}: {cm_norm[i, j]:.2f}")

# ── 12. Feature Importance (Random Forest only) ──────────────────────────────
rf_model = results["Random Forest"]["model"].named_steps["clf"]
importances = rf_model.feature_importances_

top_idx = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(10, 5))
plt.bar(range(20), importances[top_idx])
plt.xticks(range(20), [feature_cols[i] for i in top_idx], rotation=45)
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()

print("\nDone.")