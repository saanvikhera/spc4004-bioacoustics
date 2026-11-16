# =============================================================================
# Marine Bioacoustics – FINAL REFINED VERSION (Fairness + Detailed Visuals)
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

# ── 1. Data Preparation (Fixing Data Leakage) ────────────────────────────────
print("="*60)
print("MARINE BIOACOUSTICS – FINAL PROFESSIONAL ANALYSIS")
print("="*60)

# Load dataset
df = pd.read_csv("SM_1_WMD_Features_and_Labels.csv")
feature_cols = [str(i) for i in range(128)]
X = df[feature_cols].copy()
y = df["SPECIES"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# FIX: Split BEFORE any processing to prevent leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# FIX: Impute missing values using ONLY training set medians to avoid bias
train_median = X_train.median()
X_train = X_train.fillna(train_median)
X_test = X_test.fillna(train_median)

# ── 2. PCA – Exploratory Visualisation (Fig 1) ──────────────────────────────
scaler_vis = StandardScaler()
X_train_scaled = scaler_vis.fit_transform(X_train)

pca_2d = PCA(n_components=2, random_state=42)
X_pca2 = pca_2d.fit_transform(X_train_scaled)
explained = pca_2d.explained_variance_ratio_

plt.figure(figsize=(10, 7))
palette = plt.cm.get_cmap("tab10", len(class_names))
for idx, species in enumerate(class_names):
    mask = y_train == idx
    plt.scatter(X_pca2[mask, 0], X_pca2[mask, 1], label=species, alpha=0.6, s=15, color=palette(idx))

plt.title(f"Figure 1: PCA Species Separation\nPC1 ({explained[0]*100:.1f}%) | PC2 ({explained[1]*100:.1f}%)", fontsize=12)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Species", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("fig1_pca_visualisation.png", dpi=300)
plt.show()

# ── 3. PCA – Information Retention (Fig 2) ──────────────────────────────────
pca_full = PCA().fit(X_train_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.searchsorted(cum_var, 0.95) + 1

plt.figure(figsize=(9, 5))
plt.plot(range(1, len(cum_var) + 1), cum_var * 100, color="navy", lw=2)
plt.axhline(95, color="red", linestyle="--", label="95% Threshold")
plt.axvline(n_95, color="orange", linestyle="--", label=f"{n_95} Components")
plt.title("Figure 2: Cumulative Variance Explained (PCA)", fontsize=12)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig2_pca_variance.png", dpi=300)
plt.show()

# ── 4. Define Optimized Model Pipelines ──────────────────────────────────────
models = {
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1))
    ]),
    "Decision Tree": Pipeline([
        # FIX: Restricted max_depth to prevent the previous overfitting issue
        ("clf", DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42))
    ]),
    "KNN (k=7)": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)), # FIX: Dynamic PCA components
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("clf", SVC(kernel="rbf", C=5, class_weight="balanced", probability=True, random_state=42))
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        # FIX: SAGA solver for multi-class convergence and stability
        ("clf", LogisticRegression(max_iter=5000, solver='saga', class_weight="balanced", random_state=42))
    ])
}

# ── 5. Cross-Validation & Test Comparison (Fig 3) ───────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\nEvaluating Models...")
for name, pipeline in models.items():
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, n_jobs=-1)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    results[name] = {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "test_acc": accuracy_score(y_test, y_pred),
        "y_pred": y_pred
    }
    print(f"{name:20} | CV: {cv_scores.mean():.4f} | Test Acc: {results[name]['test_acc']:.4f}")

# Plot Figure 3 with labels
names = list(results.keys())
cv_means = [results[n]["cv_mean"] for n in names]
cv_stds = [results[n]["cv_std"] for n in names]
test_accs = [results[n]["test_acc"] for n in names]

x = np.arange(len(names))
width = 0.4
fig, ax = plt.subplots(figsize=(12, 7))
bar1 = ax.bar(x - width/2, cv_means, width, yerr=cv_stds, label="CV Accuracy (±SD)", color="#4E84B4", capsize=5)
bar2 = ax.bar(x + width/2, test_accs, width, label="Final Test Accuracy", color="#FF8C61")

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 8), textcoords="offset points", ha='center', va='bottom', fontsize=9)

autolabel(bar1)
autolabel(bar2)
ax.set_ylabel("Accuracy Score")
ax.set_title("Figure 3: Cross-Validation vs Test Performance")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig("fig3_model_comparison.png", dpi=300)
plt.show()

# ── 6. Winning Model Detail (Fig 4) ──────────────────────────────────────────
best_name = "SVM (RBF)"
y_pred_best = results[best_name]["y_pred"]

print(f"\nWINNER REPORT: {best_name}")
print(classification_report(y_test, y_pred_best, target_names=class_names))

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f"Figure 4: Confusion Matrix – {best_name}")
plt.ylabel('Actual Species')
plt.xlabel('Predicted Species')
plt.tight_layout()
plt.savefig("fig4_confusion_matrix.png", dpi=300)
plt.show()

# ── 7. Feature Importance (Fig 5) ─────────────────────────────────────────────
rf_clf = models["Random Forest"].named_steps["clf"]
importances = rf_clf.feature_importances_
top_idx = np.argsort(importances)[-20:]

plt.figure(figsize=(12, 8))
bars = plt.barh(range(len(top_idx)), importances[top_idx], color="teal")
plt.yticks(range(len(top_idx)), [f"Feature {feature_cols[i]}" for i in top_idx])

for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.0005, bar.get_y() + bar.get_height()/2, f'{w:.4f}', va='center', fontsize=9)

plt.title("Figure 5: Top 20 Acoustic Feature Importances")
plt.tight_layout()
plt.savefig("fig5_feature_importance.png", dpi=300)
plt.show()

# ── 8. Precision-Recall Curve (Fig 6) ─────────────────────────────────────────
Y_bin = label_binarize(y_test, classes=range(len(class_names)))
y_score = models["SVM (RBF)"].predict_proba(X_test) # Using probabilities for accurate curves

plt.figure(figsize=(11, 8))
for i in range(len(class_names)):
    precision, recall, _ = precision_recall_curve(Y_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, lw=2, label=class_names[i])

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Figure 6: Precision-Recall Curve per Species")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig6_precision_recall.png", dpi=300)
plt.show()

# ── 9. 3D Cluster Visualisation (Fig 7) ──────────────────────────────────────
pca_3d = PCA(n_components=3)
X_pca3 = pca_3d.fit_transform(X_train_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for idx, species in enumerate(class_names):
    mask = y_train == idx
    ax.scatter(X_pca3[mask, 0], X_pca3[mask, 1], X_pca3[mask, 2], label=species, s=15, alpha=0.6)

ax.set_title("Figure 7: 3D Visualization of Acoustic Classes")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
plt.tight_layout()
plt.savefig("fig7_pca_3d.png", dpi=300)
plt.show()

print("\nAll 7 high-resolution figures saved. Analysis complete.")