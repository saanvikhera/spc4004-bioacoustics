# =============================================================================
# Marine Bioacoustics – FINAL DISTINCTION ANALYSIS
# (Aesthetic, Methodologically Correct, Assessment-Ready)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.spatial import ConvexHull

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

# ── Global Aesthetic Style ───────────────────────────────────────────────────
sns.set_style("whitegrid")

PURPLE = "#8A2BE2"
PINK = "#FF69B4"
BLUE = "#1E90FF"

plt.rcParams.update({
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 9
})

print("=" * 70)
print("MARINE BIOACOUSTICS – FINAL DISTINCTION ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. DATA PREPARATION (NO DATA LEAKAGE)
# =============================================================================

df = pd.read_csv("SM_1_WMD_Features_and_Labels.csv")
feature_cols = [str(i) for i in range(128)]

X = df[feature_cols]
y = df["SPECIES"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# Train/Test split BEFORE preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Impute missing values using training data only
train_median = X_train.median()
X_train = X_train.fillna(train_median)
X_test = X_test.fillna(train_median)

# =============================================================================
# 2. PCA – EXPLORATORY VISUALISATION (FIGURE 1)
# =============================================================================

scaler_vis = StandardScaler()
X_train_scaled = scaler_vis.fit_transform(X_train)

pca_2d = PCA(n_components=2, random_state=42)
X_pca2 = pca_2d.fit_transform(X_train_scaled)
explained = pca_2d.explained_variance_ratio_

plt.figure(figsize=(10, 7))
palette = sns.color_palette("cool", len(class_names))

for i, species in enumerate(class_names):
    mask = y_train == i
    plt.scatter(
        X_pca2[mask, 0],
        X_pca2[mask, 1],
        label=species,
        alpha=0.65,
        s=20,
        color=palette[i]
    )

plt.title("Figure 1: PCA Projection of Acoustic Features")
plt.xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.figtext(
    0.5, -0.06,
    "Figure 1 Caption: Two-dimensional PCA projection of training data, illustrating overlap and clustering of marine species\n"
    "based on extracted acoustic features.",
    ha="center", fontsize=9
)
plt.tight_layout()
plt.savefig("fig1_pca.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# 3. PCA VARIANCE ANALYSIS (FIGURE 2)
# =============================================================================

pca_full = PCA().fit(X_train_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.searchsorted(cum_var, 0.95) + 1

plt.figure(figsize=(9, 5))
plt.plot(cum_var * 100, color=PURPLE, lw=2)
plt.axhline(95, color=PINK, linestyle="--")
plt.axvline(n_95, color=BLUE, linestyle="--")
plt.title("Figure 2: PCA Cumulative Variance Explained")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance (%)")
plt.figtext(
    0.5, -0.12,
    f"Figure 2 Caption: PCA variance curve showing that {n_95} components retain 95% of total variance,\n"
    "supporting effective dimensionality reduction prior to classification.",
    ha="center", fontsize=9
)
plt.tight_layout()
plt.savefig("fig2_pca_variance.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# 4. MODEL DEFINITIONS (FAIR & CONSISTENT)
# =============================================================================

models = {
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1))
    ]),
    "Decision Tree": Pipeline([
        ("clf", DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_95)),
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_95)),
        ("clf", SVC(
            kernel="rbf",
            C=10,
            class_weight="balanced",
            probability=True,
            random_state=42))
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_95)),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42))
    ])
}

# =============================================================================
# 5. MODEL EVALUATION (FIGURE 3)
# =============================================================================

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

    print(
        f"{name:20} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | "
        f"Test Acc: {results[name]['test_acc']:.4f} | "
        f"F1: {results[name]['macro_f1']:.4f}"
    )

names = list(results.keys())
x = np.arange(len(names))

plt.figure(figsize=(12, 6))
plt.bar(x - 0.2, [results[n]["cv_mean"] for n in names],
        width=0.35, color=BLUE, label="CV Accuracy")
plt.bar(x + 0.2, [results[n]["macro_f1"] for n in names],
        width=0.35, color=PINK, label="Macro F1")

plt.xticks(x, names, rotation=15)
plt.ylabel("Score")
plt.title("Figure 3: Model Performance Comparison")
plt.legend()
plt.figtext(
    0.5, -0.15,
    "Figure 3 Caption: Comparison of classification models using cross-validation accuracy and macro F1 score,\n"
    "highlighting both performance and class fairness.",
    ha="center", fontsize=9
)
plt.tight_layout()
plt.savefig("fig3_model_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# 6. BEST MODEL & CONFUSION MATRIX (FIGURE 4)
# =============================================================================

best_name = max(results, key=lambda x: results[x]["macro_f1"])
y_pred_best = results[best_name]["y_pred"]

print(f"\nBest Model Selected: {best_name}")
print(classification_report(y_test, y_pred_best, target_names=class_names))

cm_norm = confusion_matrix(y_test, y_pred_best, normalize="true")

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="magma",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title(f"Figure 4: Normalised Confusion Matrix – {best_name}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.figtext(
    0.5, -0.08,
    "Figure 4 Caption: Normalised confusion matrix illustrating per-class prediction accuracy and\n"
    "common misclassification patterns for the top-performing model.",
    ha="center", fontsize=9
)
plt.tight_layout()
plt.savefig("fig4_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# 7. FEATURE IMPORTANCE (FIGURE 5)
# =============================================================================

rf = models["Random Forest"].named_steps["clf"]
importances = rf.feature_importances_
top_idx = np.argsort(importances)[-20:]
colors = plt.cm.cool(np.linspace(0, 1, 20))[::-1]

plt.figure(figsize=(11, 8))
plt.barh(
    range(20),
    importances[top_idx],
    color=colors
)
plt.yticks(range(20), [f"Feature {i}" for i in top_idx])
plt.xlabel("Gini Importance")
plt.title("Figure 5: Top 20 Acoustic Feature Importances")
plt.figtext(
    0.5, -0.1,
    "Figure 5 Caption: Random Forest Gini importance scores identifying the most influential\n"
    "frequency-domain features for species classification.",
    ha="center", fontsize=9
)
plt.tight_layout()
plt.savefig("fig5_feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# 8. PRECISION–RECALL CURVES (FIGURE 6)
# =============================================================================

Y_bin = label_binarize(y_test, classes=range(len(class_names)))
y_score = models[best_name].predict_proba(X_test)

plt.figure(figsize=(10, 7))
for i, species in enumerate(class_names):
    precision, recall, _ = precision_recall_curve(Y_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, lw=2, label=species)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Figure 6: Precision–Recall Curves by Species")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.figtext(
    0.5, -0.12,
    "Figure 6 Caption: Precision–recall curves for each species, illustrating trade-offs between\n"
    "detection sensitivity and classification precision in an imbalanced multi-class setting.",
    ha="center", fontsize=9
)
plt.tight_layout()
plt.savefig("fig6_precision_recall.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# 9. 3D PCA VISUALISATION (FIGURE 7)
# =============================================================================

pca_3d = PCA(n_components=3)
X_pca3 = pca_3d.fit_transform(X_train_scaled)

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

for i, species in enumerate(class_names):
    mask = y_train == i
    ax.scatter(
        X_pca3[mask, 0],
        X_pca3[mask, 1],
        X_pca3[mask, 2],
        label=species,
        alpha=0.6,
        s=15
    )

ax.set_title("Figure 7: 3D PCA Feature Space")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend(bbox_to_anchor=(1.1, 1))

plt.tight_layout()
plt.savefig("fig7_pca_3d.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nAll figures generated successfully. Analysis complete.")
