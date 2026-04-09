# =============================================================================
# Marine Bioacoustics – FINAL REFINED VERSION (No Leakage + Strong Evaluation)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve
)

warnings.filterwarnings("ignore")

# ── 1. Load Data (NO DATA LEAKAGE) ───────────────────────────────────────────
print("="*60)
print("MARINE BIOACOUSTICS – FINAL PROFESSIONAL ANALYSIS")
print("="*60)

df = pd.read_csv("SM_1_WMD_Features_and_Labels.csv")

feature_cols = [str(i) for i in range(128)]
X = df[feature_cols].copy()
y = df["SPECIES"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# Train-test split BEFORE preprocessing (prevents leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Impute using TRAINING DATA ONLY
train_median = X_train.median()
X_train = X_train.fillna(train_median)
X_test = X_test.fillna(train_median)

# ── 2. PCA Visualisation (Figure 1) ─────────────────────────────────────────
# Separate scaler used ONLY for visualisation
scaler_vis = StandardScaler()
X_train_scaled = scaler_vis.fit_transform(X_train)

pca_2d = PCA(n_components=2, random_state=42)
X_pca2 = pca_2d.fit_transform(X_train_scaled)
explained = pca_2d.explained_variance_ratio_

plt.figure(figsize=(10, 7))
palette = plt.cm.get_cmap("tab10", len(class_names))

for idx, species in enumerate(class_names):
    mask = y_train == idx
    plt.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                label=species, alpha=0.6, s=15, color=palette(idx))

plt.title(
    f"Figure 1: PCA Projection (Limited Class Separation)\n"
    f"PC1 ({explained[0]*100:.1f}%) | PC2 ({explained[1]*100:.1f}%)"
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("fig1_pca_visualisation.png", dpi=300)
plt.show()

print(f"\nPCA Insight: Only {sum(explained)*100:.1f}% variance captured → poor 2D separability")

# ── 3. PCA Variance Curve (Figure 2) ───────────────────────────────────────
pca_full = PCA().fit(X_train_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.searchsorted(cum_var, 0.95) + 1

plt.figure(figsize=(9, 5))
plt.plot(range(1, len(cum_var)+1), cum_var*100)
plt.axhline(95, linestyle="--", label="95% Threshold")
plt.axvline(n_95, linestyle="--", label=f"{n_95} Components")

plt.title("Figure 2: PCA Variance Explained")
plt.xlabel("Components")
plt.ylabel("Cumulative Variance (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig2_pca_variance.png", dpi=300)
plt.show()

# ── 4. Models ──────────────────────────────────────────────────────────────
models = {
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("clf", DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_95)),
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_95)),
        ("clf", SVC(kernel="rbf", C=10, class_weight="balanced", probability=True, random_state=42))
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_95)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))
    ])
}

# ── 5. Evaluation (Accuracy + Macro F1) ─────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\nEvaluating Models...")

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "test_acc": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "y_pred": y_pred
    }

    print(f"{name:20} | CV: {cv_scores.mean():.4f} | Test: {results[name]['test_acc']:.4f} | F1: {results[name]['macro_f1']:.4f}")

# ── 6. Model Comparison Plot (Figure 3) ─────────────────────────────────────
names = list(results.keys())
test_accs = [results[n]["test_acc"] for n in names]
f1_scores = [results[n]["macro_f1"] for n in names]

x = np.arange(len(names))
width = 0.35

plt.figure(figsize=(11,6))
plt.bar(x - width/2, test_accs, width, label="Accuracy")
plt.bar(x + width/2, f1_scores, width, label="Macro F1")

plt.xticks(x, names, rotation=20)
plt.ylabel("Score")
plt.title("Figure 3: Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("fig3_model_comparison.png", dpi=300)
plt.show()

# ── 7. Best Model Selection ─────────────────────────────────────────────────
best_name = max(results, key=lambda x: results[x]["test_acc"])
print(f"\nBest Model: {best_name}")

best_model = models[best_name]
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=class_names))

# ── 8. Normalised Confusion Matrix (Figure 4) ───────────────────────────────
cm = confusion_matrix(y_test, y_pred_best, normalize='true')

plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)

plt.title(f"Figure 4: Normalised Confusion Matrix – {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("fig4_confusion_matrix.png", dpi=300)
plt.show()

# ── 9. Misclassification Analysis ───────────────────────────────────────────
cm_raw = confusion_matrix(y_test, y_pred_best)

print("\nCommon Misclassifications:")
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j:
            rate = cm_raw[i, j] / cm_raw[i].sum()
            if rate > 0.1:
                print(f"{class_names[i]} → {class_names[j]}: {rate:.2f}")

# ── 10. Feature Importance (Figure 5) ───────────────────────────────────────
rf = models["Random Forest"].named_steps["clf"]
importances = rf.feature_importances_
idx = np.argsort(importances)[-20:]

plt.figure(figsize=(10,6))
plt.barh(range(20), importances[idx])
plt.yticks(range(20), [f"F{i}" for i in idx])
plt.title("Figure 5: Feature Importance")
plt.tight_layout()
plt.savefig("fig5_feature_importance.png", dpi=300)
plt.show()

# ── 11. Precision-Recall Curve (Figure 6) ───────────────────────────────────
Y_bin = label_binarize(y_test, classes=range(len(class_names)))
y_score = best_model.predict_proba(X_test)

plt.figure(figsize=(10,7))
for i in range(len(class_names)):
    precision, recall, _ = precision_recall_curve(Y_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, label=class_names[i])

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Figure 6: Precision-Recall Curves")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("fig6_precision_recall.png", dpi=300)
plt.show()

# ── 12. 3D PCA (Figure 7) ───────────────────────────────────────────────────
pca_3d = PCA(n_components=3)
X_pca3 = pca_3d.fit_transform(X_train_scaled)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

for idx, species in enumerate(class_names):
    mask = y_train == idx
    ax.scatter(X_pca3[mask, 0], X_pca3[mask, 1], X_pca3[mask, 2], s=10)

ax.set_title("Figure 7: 3D PCA Visualisation")
plt.tight_layout()
plt.savefig("fig7_pca_3d.png", dpi=300)
plt.show()

print("\nAll figures generated. Analysis complete.")