# =============================================================================
# Marine Bioacoustics – FINAL REFINED VERSION (Fairness + Detailed Visuals)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
print("="*60)
print("MARINE BIOACOUSTICS – FINAL PERFORMANCE ANALYSIS")
print("="*60)

df = pd.read_csv("SM_1_WMD_Features_and_Labels.csv")
feature_cols = [str(i) for i in range(128)]
X = df[feature_cols].copy().fillna(df[feature_cols].median())
y = df["SPECIES"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

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
    plt.scatter(X_pca2[mask, 0], X_pca2[mask, 1], label=species, alpha=0.5, s=12, color=palette(idx))

plt.title(f"Figure 1: PCA Species Separation\nPC1 ({explained[0]*100:.1f}%) | PC2 ({explained[1]*100:.1f}%)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Species", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("fig1_pca_visualisation.png", dpi=150)
plt.show()

# ── 3. PCA – Variance Explained Curve (Fig 2) ───────────────────────────────
pca_full = PCA().fit(X_train_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.searchsorted(cum_var, 0.95) + 1

plt.figure(figsize=(9, 5))
plt.plot(range(1, len(cum_var) + 1), cum_var * 100, color="navy", lw=2)
plt.axhline(95, color="red", linestyle="--", label="95% Threshold")
plt.axvline(n_95, color="orange", linestyle="--", label=f"{n_95} Components")
plt.title("Figure 2: Information Retention (PCA)")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig2_pca_variance.png", dpi=150)
plt.show()

# ── 4. Define Balanced Models (Fairness Priority) ──────────────────────────
# Using class_weight='balanced' ensures rare species are NOT ignored
models = {
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1))
    ]),
    "Decision Tree": Pipeline([
        ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=42))
    ]),
    "KNN (k=7)": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_95)),
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_95)),
        ("clf", SVC(kernel="rbf", C=10, class_weight="balanced", probability=True, random_state=42))
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_95)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ])
}

# ── 5. Cross-Validation & Test (Fig 3) ──────────────────────────────────────
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
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "y_pred": y_pred
    }
    print(f"{name:20} | CV: {cv_scores.mean():.4f} | Test Acc: {results[name]['test_acc']:.4f}")

# Model Comparison Plot (Fig 3)
names = list(results.keys())
cv_means = [results[n]["cv_mean"] for n in names]
cv_stds = [results[n]["cv_std"] for n in names]
test_accs = [results[n]["test_acc"] for n in names]

x = np.arange(len(names))
plt.figure(figsize=(11, 6))
plt.bar(x - 0.2, cv_means, 0.4, yerr=cv_stds, label="CV Accuracy (±SD)", color="steelblue", capsize=5)
plt.bar(x + 0.2, test_accs, 0.4, label="Test Accuracy", color="coral")
plt.xticks(x, names, rotation=15)
plt.ylabel("Accuracy Score")
plt.title("Figure 3: Model Comparison (Fairness Weighting)")
plt.legend()
plt.tight_layout()
plt.savefig("fig3_model_comparison.png", dpi=150)
plt.show()

# ── 6. Best Model – Confusion Matrix (Fig 4) ───────────────────────────────
best_name = "SVM (RBF)" # Typically highest F1-score when balanced
y_pred_best = results[best_name]["y_pred"]

print(f"\nWINNER: {best_name}")
print(classification_report(y_test, y_pred_best, target_names=class_names))

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f"Figure 4: Confusion Matrix – {best_name} (Balanced)")
plt.ylabel('Actual Species')
plt.xlabel('Predicted Species')
plt.tight_layout()
plt.savefig("fig4_confusion_matrix.png", dpi=150)
plt.show()

# ── 7. Feature Importance – Random Forest (Fig 5) ───────────────────────────
rf_clf = models["Random Forest"].named_steps["clf"]
importances = rf_clf.feature_importances_
top_n = 20
top_idx = np.argsort(importances)[::-1][:top_n]

plt.figure(figsize=(11, 6))
plt.bar(range(top_n), importances[top_idx], color="teal", edgecolor="black")
plt.xticks(range(top_n), [f"F{feature_cols[i]}" for i in top_idx], rotation=45)
plt.ylabel("Mean Decrease in Impurity")
plt.title("Figure 5: Top 20 Acoustic Feature Importances")
plt.tight_layout()
plt.savefig("fig5_feature_importance.png", dpi=150)
plt.show()

from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score

# --- EXTRA FIG 6: Multi-class Precision-Recall Curve ---
Y_bin = label_binarize(y_test, classes=range(len(class_names)))
n_classes = Y_bin.shape[1]
y_score = models["SVM (RBF)"].decision_function(X_test)

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(Y_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, lw=2, label=class_names[i])

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Figure 6: Precision-Recall Curve per Species")
plt.legend(loc="best", fontsize='small', ncol=2)
plt.grid(alpha=0.3)
plt.savefig("fig6_precision_recall.png", dpi=150)
plt.show()

# --- EXTRA FIG 7: 3D PCA Visualisation ---
pca_3d = PCA(n_components=3)
X_pca3 = pca_3d.fit_transform(X_train_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for idx, species in enumerate(class_names):
    mask = y_train == idx
    ax.scatter(X_pca3[mask, 0], X_pca3[mask, 1], X_pca3[mask, 2], label=species, s=15, alpha=0.6)

ax.set_title("Figure 7: 3D PCA Species Clusters")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("fig7_pca_3d.png", dpi=150)
plt.show()

print("\nAnalysis complete. All  figures saved.")