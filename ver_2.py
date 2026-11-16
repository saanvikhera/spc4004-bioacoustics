# =============================================================================
# Marine Bioacoustics Species Classification
# Dataset: SM_1_WMD_Features_and_Labels.csv (Watkins Marine Mammal Database)
# Task: Multi-class classification of 12 marine mammal species from acoustic features
# Final refined version
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

# ── 1. Load & Inspect Data ────────────────────────────────────────────────────
print("=" * 60)
print("MARINE BIOACOUSTICS – SPECIES CLASSIFICATION")
print("=" * 60)

DATA_PATH = "SM_1_WMD_Features_and_Labels.csv"
df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")
print(f"\nClass distribution (SPECIES):\n{df['SPECIES'].value_counts()}")

# ── 2. Define Features and Target ─────────────────────────────────────────────
# Features: 128 acoustic feature columns named '0' to '127'
feature_cols = [str(i) for i in range(128)]
X = df[feature_cols].copy()
y = df["SPECIES"]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Number of classes: {y.nunique()}")
print(f"Missing values in features: {X.isnull().sum().sum()}")

# ── 3. Handle Missing Values ──────────────────────────────────────────────────
# Impute missing values with per-column median (robust to outliers)
for col in feature_cols:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())

print(f"Missing values after imputation: {X.isnull().sum().sum()}")

# ── 4. Encode Target Labels ───────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
print(f"\nEncoded species: {list(class_names)}")

# ── 5. Stratified Train/Test Split ────────────────────────────────────────────
# stratify=y_encoded ensures each split reflects the class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── 6. Feature Scaling ────────────────────────────────────────────────────────
# Essential for distance-based and linear models (KNN, SVM, Logistic Regression)
# Fit ONLY on training data to prevent data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 7. PCA – Exploratory Visualisation ───────────────────────────────────────
print("\n[PCA] Fitting for 2D visualisation...")
pca_2d = PCA(n_components=2, random_state=42)
X_train_pca2 = pca_2d.fit_transform(X_train_scaled)
X_test_pca2 = pca_2d.transform(X_test_scaled)

explained = pca_2d.explained_variance_ratio_
print(f"  PC1 explains {explained[0]*100:.1f}% | PC2 explains {explained[1]*100:.1f}% of variance")

fig, ax = plt.subplots(figsize=(10, 7))
palette = plt.cm.get_cmap("tab10", len(class_names))
for idx, species in enumerate(class_names):
    mask = y_train == idx
    ax.scatter(
        X_train_pca2[mask, 0], X_train_pca2[mask, 1],
        label=species, alpha=0.5, s=12, color=palette(idx)
    )
ax.set_title(
    f"PCA – First Two Components (Training Set)\n"
    f"Variance explained: {sum(explained)*100:.1f}%"
)
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
ax.legend(title="Species", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("fig1_pca_visualisation.png", dpi=150)
plt.show()
print("  Figure saved: fig1_pca_visualisation.png")

# ── 8. PCA – Variance Explained Curve ────────────────────────────────────────
pca_full = PCA(random_state=42)
pca_full.fit(X_train_scaled)
cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.searchsorted(cumulative_var, 0.95) + 1

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, len(cumulative_var) + 1), cumulative_var * 100, color="navy")
ax.axhline(95, color="red", linestyle="--", label="95% threshold")
ax.axvline(n_95, color="orange", linestyle="--", label=f"{n_95} components")
ax.set_xlabel("Number of PCA Components")
ax.set_ylabel("Cumulative Variance Explained (%)")
ax.set_title("PCA – Cumulative Variance Explained")
ax.legend()
plt.tight_layout()
plt.savefig("fig2_pca_variance.png", dpi=150)
plt.show()
print(f"  {n_95} components needed to explain 95% of variance")
print("  Figure saved: fig2_pca_variance.png")

# ── 9. Define Models ──────────────────────────────────────────────────────────
# All models are wrapped in Pipelines where scaling is needed.
# Tree-based models (RF, DT) do NOT need scaling – they use raw features.
models = {
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200,      # More trees = more stable predictions
            max_depth=None,        # Allow full depth; controlled by min_samples_leaf
            min_samples_leaf=2,    # Slight regularisation to reduce overfitting
            class_weight="balanced",  # Compensate for class imbalance
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "Decision Tree": Pipeline([
        ("clf", DecisionTreeClassifier(
            max_depth=15,          # Pruned to reduce overfitting
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        ))
    ]),
    "KNN (k=7)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7, metric="euclidean", n_jobs=-1))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=42))
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42
        ))
    ]),
}

# ── 10. Cross-Validation + Test Evaluation ────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, pipeline in models.items():
    print(f"\n[{name}]")

    # 5-fold cross-validation on training set
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Fit on full training set and evaluate on held-out test set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Macro F1:      {macro_f1:.4f}")

    results[name] = {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "test_acc": test_acc,
        "macro_f1": macro_f1,
        "y_pred": y_pred,
    }

# ── 11. Model Comparison Figure ───────────────────────────────────────────────
names = list(results.keys())
test_accs = [results[n]["test_acc"] for n in names]
cv_means  = [results[n]["cv_mean"]  for n in names]
cv_stds   = [results[n]["cv_std"]   for n in names]

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
bars1 = ax.bar(x - width/2, cv_means, width, yerr=cv_stds, capsize=5,
               label="CV Accuracy (±1 SD)", color="steelblue", edgecolor="black")
bars2 = ax.bar(x + width/2, test_accs, width,
               label="Test Accuracy", color="coral", edgecolor="black")

ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Accuracy")
ax.set_title("Model Comparison – Marine Bioacoustics Species Classification")
ax.legend()

for bar, val in zip(bars1, cv_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8)
for bar, val in zip(bars2, test_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("fig3_model_comparison.png", dpi=150)
plt.show()
print("\nFigure saved: fig3_model_comparison.png")

# ── 12. Classification Report & Confusion Matrix – Best Model ─────────────────
best_model_name = max(results, key=lambda n: results[n]["test_acc"])
print(f"\nBest model: {best_model_name} (Test Acc = {results[best_model_name]['test_acc']:.4f})")

y_pred_best = results[best_model_name]["y_pred"]
print(f"\nClassification Report – {best_model_name}:")
print(classification_report(y_test, y_pred_best, target_names=class_names))

cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(11, 9))
disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
ax.set_title(f"Confusion Matrix – {best_model_name}")
plt.tight_layout()
plt.savefig("fig4_confusion_matrix.png", dpi=150)
plt.show()
print("Figure saved: fig4_confusion_matrix.png")

# ── 13. Feature Importance – Random Forest ────────────────────────────────────
rf_clf = models["Random Forest"].named_steps["clf"]
importances = rf_clf.feature_importances_
top_n = 20
top_idx = np.argsort(importances)[::-1][:top_n]

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(range(top_n), importances[top_idx], color="teal", edgecolor="black")
ax.set_xticks(range(top_n))
ax.set_xticklabels([f"F{feature_cols[i]}" for i in top_idx], rotation=45, ha="right")
ax.set_ylabel("Importance (Mean Decrease in Impurity)")
ax.set_title(f"Top {top_n} Acoustic Feature Importances – Random Forest")
plt.tight_layout()
plt.savefig("fig5_feature_importance.png", dpi=150)
plt.show()
print("Figure saved: fig5_feature_importance.png")

# ── 14. Summary Table ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
summary = pd.DataFrame({
    "Model": names,
    "CV Accuracy (mean)": [f"{results[n]['cv_mean']:.4f}" for n in names],
    "CV Accuracy (std)":  [f"±{results[n]['cv_std']:.4f}" for n in names],
    "Test Accuracy":      [f"{results[n]['test_acc']:.4f}" for n in names],
    "Macro F1":           [f"{results[n]['macro_f1']:.4f}" for n in names],
})
print(summary.to_string(index=False))
print("\nAll figures saved. Analysis complete.")
