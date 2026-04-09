# =============================================================================
# Marine Bioacoustics – THE ULTIMATE AESTHETIC EDITION (Final Consolidated)
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
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             f1_score, precision_recall_curve)

warnings.filterwarnings("ignore")

# Set global aesthetic style
sns.set_style("whitegrid")
PINK = "#FF69B4"
BLUE = "#1E90FF"
PURPLE = "#8A2BE2"

# ── 1. Data Preparation (No Leakage) ────────────────────────────────────────
print("="*60)
print("MARINE BIOACOUSTICS – MASTER PERFORMANCE ANALYSIS")
print("="*60)

df = pd.read_csv("SM_1_WMD_Features_and_Labels.csv")
feature_cols = [str(i) for i in range(128)]
X = df[feature_cols].copy()
y = df["SPECIES"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

train_median = X_train.median()
X_train = X_train.fillna(train_median)
X_test = X_test.fillna(train_median)

# ── 2. PCA Visualisation (Fig 1 & 2) ────────────────────────────────────────
scaler_vis = StandardScaler()
X_train_scaled = scaler_vis.fit_transform(X_train)

pca_2d = PCA(n_components=2, random_state=42)
X_pca2 = pca_2d.fit_transform(X_train_scaled)
explained = pca_2d.explained_variance_ratio_

plt.figure(figsize=(10, 7))
# Using a "cool" palette for Pink/Blue/Purple vibes
palette = sns.color_palette("cool", len(class_names))
for idx, species in enumerate(class_names):
    mask = y_train == idx
    plt.scatter(X_pca2[mask, 0], X_pca2[mask, 1], label=species, alpha=0.7, s=20, color=palette[idx])

plt.title(f"Figure 1: PCA Projection Spectrum\nPC1 ({explained[0]*100:.1f}%) | PC2 ({explained[1]*100:.1f}%)", fontweight='bold')
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("fig1_pca.png", dpi=300); plt.show()

pca_full = PCA().fit(X_train_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.searchsorted(cum_var, 0.95) + 1

# ── 3. Optimized Model Pipelines ────────────────────────────────────────────
models = {
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1))
    ]),
    "Decision Tree": Pipeline([
        ("clf", DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("clf", SVC(kernel="rbf", C=10, class_weight="balanced", probability=True, random_state=42))
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("clf", LogisticRegression(max_iter=5000, solver='saga', class_weight="balanced", random_state=42))
    ])
}

# ── 4. Evaluation ────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\nEvaluating Models...")
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "cv_mean": cv_scores.mean(),
        "test_acc": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "y_pred": y_pred
    }
    print(f"{name:20} | CV: {cv_scores.mean():.4f} | Test: {results[name]['test_acc']:.4f} | F1: {results[name]['f1']:.4f}")

# ── 5. Model Comparison (Figure 3) ──────────────────────────────────────────
names = list(results.keys())
accs = [results[n]["test_acc"] for n in names]
f1s = [results[n]["f1"] for n in names]

x = np.arange(len(names))
width = 0.35
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, accs, width, label="Test Accuracy", color=BLUE, alpha=0.8)
plt.bar(x + width/2, f1s, width, label="Macro F1 Score", color=PINK, alpha=0.8)

plt.xticks(x, names, rotation=15)
plt.title("Figure 3: Model Performance (Accuracy vs F1)", fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig("fig3_comparison.png", dpi=300); plt.show()

# ── 6. Normalised Confusion Matrix (Figure 4) ──────────────────────────────
best_name = max(results, key=lambda x: results[x]["test_acc"])
y_pred_best = results[best_name]["y_pred"]

cm_norm = confusion_matrix(y_test, y_pred_best, normalize='true')
plt.figure(figsize=(12, 10))
# Using "magma" for a beautiful Purple/Pink/Orange heatmap
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='magma', xticklabels=class_names, yticklabels=class_names)
plt.title(f"Figure 4: Normalised Confusion Matrix – {best_name}", fontweight='bold')
plt.ylabel('Actual Species'); plt.xlabel('Predicted Species')
plt.tight_layout(); plt.savefig("fig4_confusion.png", dpi=300); plt.show()

# Common Misclassifications Printout
cm_raw = confusion_matrix(y_test, y_pred_best)
print(f"\nCommon Misclassifications for {best_name}:")
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j:
            rate = cm_raw[i, j] / cm_raw[i].sum()
            if rate > 0.1:
                print(f"{class_names[i]} → {class_names[j]}: {rate:.2f}")

# ── 7. Feature Importance (Figure 5) ────────────────────────────────────────
rf = models["Random Forest"].named_steps["clf"]
importances = rf.feature_importances_
idx = np.argsort(importances)[-20:]

plt.figure(figsize=(10, 8))
plt.barh(range(20), importances[idx], color=PURPLE, alpha=0.7)
plt.yticks(range(20), [f"Freq Feature {i}" for i in idx])
plt.title("Figure 5: Top 20 Acoustic Feature Importance", fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("fig5_features.png", dpi=300); plt.show()

# ── 8. Precision-Recall Curve (Figure 6) ────────────────────────────────────
Y_bin = label_binarize(y_test, classes=range(len(class_names)))
y_score = models[best_name].predict_proba(X_test)

plt.figure(figsize=(10, 7))
pr_palette = sns.color_palette("husl", len(class_names))
for i in range(len(class_names)):
    precision, recall, _ = precision_recall_curve(Y_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, label=class_names[i], color=pr_palette[i], lw=2)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Figure 6: Precision-Recall Curve per Species", fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.savefig("fig6_pr_curve.png", dpi=300); plt.show()

# ── 9. 3D Aesthetic PCA (Figure 7) ──────────────────────────────────────────
pca_3d = PCA(n_components=3)
X_pca3 = pca_3d.fit_transform(X_train_scaled)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
for idx, species in enumerate(class_names):
    mask = y_train == idx
    ax.scatter(X_pca3[mask, 0], X_pca3[mask, 1], X_pca3[mask, 2], 
               label=species, s=20, alpha=0.6, color=palette[idx])

ax.set_title("Figure 7: 3D Acoustic Feature Space", fontweight='bold')
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("fig7_pca_3d.png", dpi=300); plt.show()

print("\nUltimate Analysis Complete. All high-res aesthetic figures saved.")