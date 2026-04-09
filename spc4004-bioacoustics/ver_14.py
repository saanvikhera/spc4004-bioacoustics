# =============================================================================
# MARINE BIOACOUSTICS – FINAL DISTINCTION ANALYSIS
# (Methodologically Correct + Advanced Visualisation Suite)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.spatial import ConvexHull
from itertools import combinations

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
sns.set_style("whitegrid")

# ── Aesthetic Palette ────────────────────────────────────────────────────────
BLUE = "#1E90FF"
PINK = "#FF69B4"
PURPLE = "#8A2BE2"

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

df = pd.read_csv("../SM_1_WMD_Features_and_Labels.csv")
feature_cols = [str(i) for i in range(128)]

X = df[feature_cols]
y = df["SPECIES"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Imputation using training data only
train_median = X_train.median()
X_train = X_train.fillna(train_median)
X_test = X_test.fillna(train_median)

# =============================================================================
# 2. PCA – 2D EXPLORATORY VISUALISATION (FIGURE 1)
# =============================================================================

scaler_vis = StandardScaler()
X_train_scaled = scaler_vis.fit_transform(X_train)

pca_2d = PCA(n_components=2, random_state=42)
X_pca2 = pca_2d.fit_transform(X_train_scaled)
explained = pca_2d.explained_variance_ratio_

palette = sns.color_palette("tab20", len(class_names))

plt.figure(figsize=(10, 7))
for i, species in enumerate(class_names):
    mask = y_train == i
    plt.scatter(
        X_pca2[mask, 0],
        X_pca2[mask, 1],
        label=species,
        alpha=0.7,
        s=25,
        color=palette[i],
        edgecolor="k",
        linewidth=0.2
    )

plt.title("Figure 1: PCA Projection of Acoustic Features")
plt.xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.figtext(
    0.5, -0.07,
    "Figure 1 Caption: Two-dimensional PCA projection of training data illustrating species clustering "
    "and overlap in reduced acoustic feature space.",
    ha="center", fontsize=9
)
plt.tight_layout()
plt.savefig("fig1_pca.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# 2B. PCA WITH CONVEX HULLS (FIGURE 1C)
# =============================================================================

plt.figure(figsize=(11, 8))
for i, species in enumerate(class_names):
    mask = y_train == i
    points = X_pca2[mask]
    plt.scatter(points[:, 0], points[:, 1], s=18, alpha=0.6, color=palette[i], label=species)
    if len(points) >= 3:
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], color=palette[i], lw=1.2)

plt.title("Figure 1C: PCA Projection with Convex Hulls")
plt.xlabel(f"PC1 ({explained[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}%)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
plt.figtext(
    0.5, -0.06,
    "Figure 1C Caption: Convex hulls highlight the spatial spread and boundaries of each species "
    "in PCA space, revealing inter-class overlap and intra-class variance.",
    ha="center", fontsize=9
)
plt.tight_layout()
plt.savefig("fig1c_pca_hulls.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# 2C. CLASS DISTRIBUTION (FIGURE 1D)
# =============================================================================

train_counts = pd.Series(y_train).map(lambda x: class_names[x]).value_counts()
test_counts = pd.Series(y_test).map(lambda x: class_names[x]).value_counts()
pd.DataFrame({"Train": train_counts, "Test": test_counts}).plot(
    kind="bar", figsize=(12, 5), width=0.75
)

plt.title("Figure 1D: Class Distribution – Train vs Test Sets")
plt.xlabel("Species")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.figtext(
    0.5, -0.08,
    "Figure 1D Caption: Stratified splitting preserves relative class proportions between "
    "training and test datasets, ensuring fair evaluation.",
    ha="center", fontsize=9
)
plt.tight_layout()
plt.savefig("fig1d_class_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# 3. PCA VARIANCE EXPLAINED (FIGURE 2)
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
    f"Figure 2 Caption: Approximately {n_95} principal components retain 95% of total variance, "
    "supporting dimensionality reduction prior to classification.",
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
        f"Macro F1: {results[name]['macro_f1']:.4f}"
    )

# Bar chart
names = list(results.keys())
x = np.arange(len(names))

plt.figure(figsize=(12, 6))
plt.bar(x - 0.25, [results[n]["cv_mean"] for n in names], 0.25, yerr=[results[n]["cv_std"] for n in names],
        label="CV Accuracy (±SD)", color=BLUE, capsize=5)
plt.bar(x, [results[n]["test_acc"] for n in names], 0.25,
        label="Test Accuracy", color=PINK)
plt.bar(x + 0.25, [results[n]["macro_f1"] for n in names], 0.25,
        label="Macro F1 Score", color=PURPLE)

plt.xticks(x, names, rotation=15)
plt.ylabel("Score")
plt.title("Figure 3: Model Performance Comparison")
plt.legend()
plt.figtext(
    0.5, -0.15,
    "Figure 3 Caption: Comparison of classification models using cross-validation accuracy, "
    "test accuracy, and macro F1 score, emphasising balanced performance across species.",
    ha="center", fontsize=9
)
plt.tight_layout()
plt.savefig("fig3_model_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# 6. BEST MODEL & CONFUSION MATRIX (FIGURE 4 & 4B)
# =============================================================================

best_name = max(results, key=lambda x: results[x]["macro_f1"])
y_pred_best = results[best_name]["y_pred"]

print(f"\nBest Model Selected: {best_name}")
print(classification_report(y_test, y_pred_best, target_names=class_names))

cm_norm = confusion_matrix(y_test, y_pred_best, normalize="true")

plt.figure(figsize=(12, 10))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="magma",
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"Figure 4: Normalised Confusion Matrix – {best_name}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.figtext(
    0.5, -0.08,
    "Figure 4 Caption: Normalised confusion matrix highlighting per-class prediction accuracy "
    "and systematic misclassification patterns.",
    ha="center", fontsize=9
)
plt.tight_layout()
plt.savefig("fig4_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# Top misclassifications (Figure 4B)
mis_pairs = [(i, j, cm_norm[i, j]) for i in range(len(class_names))
             for j in range(len(class_names)) if i != j]
mis_pairs = sorted(mis_pairs, key=lambda x: x[2], reverse=True)[:10]

plt.figure(figsize=(12, 6))
plt.bar(
    [f"{class_names[i]} → {class_names[j]}" for i, j, _ in mis_pairs],
    [v for _, _, v in mis_pairs],
    color=PINK
)
plt.xticks(rotation=45)
plt.ylabel("Normalised Error Rate")
plt.title("Figure 4B: Top Misclassifications")
plt.figtext(
    0.5, -0.08,
    "Figure 4B Caption: The most frequently confused species pairs, "
    "indicating acoustically similar call structures.",
    ha="center", fontsize=9
)
plt.tight_layout()
plt.savefig("fig4b_top_misclassifications.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# 7. RANDOM FOREST FEATURE IMPORTANCE SUITE (FIGURES 5, 5B, 5C)
# =============================================================================

rf = models["Random Forest"].named_steps["clf"]
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)

# Figure 5
top20 = sorted_idx[-20:]
plt.figure(figsize=(11, 8))
plt.barh(range(20), importances[top20],
         color=plt.cm.cool(np.linspace(0, 1, 20))[::-1])
plt.yticks(range(20), [f"Feature {i}" for i in top20])
plt.xlabel("Gini Importance")
plt.title("Figure 5: Top 20 Acoustic Feature Importances")
plt.figtext(
    0.5, -0.1,
    "Figure 5 Caption: Random Forest Gini importance identifies the most influential "
    "frequency-domain features contributing to classification.",
    ha="center", fontsize=9
)
plt.tight_layout()
plt.savefig("fig5_feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()

# Figure 5C – cumulative importance
plt.figure(figsize=(12, 5))
plt.plot(np.cumsum(importances[sorted_idx]), color=BLUE, lw=2)
plt.title("Figure 5C: Cumulative Feature Importance")
plt.xlabel("Features Sorted by Importance")
plt.ylabel("Cumulative Importance")
plt.figtext(
    0.5, -0.1,
    "Figure 5C Caption: Demonstrates that a relatively small subset of features "
    "accounts for the majority of predictive power.",
    ha="center", fontsize=9
)
plt.tight_layout()
plt.savefig("fig5c_cumulative_importance.png", dpi=300, bbox_inches="tight")
plt.show()

# Figure 5B – correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(X_train.iloc[:, top20].corr(), cmap="coolwarm", fmt=".2f")
plt.title("Figure 5B: Correlation Heatmap of Top 20 Features")
plt.figtext(
    0.5, -0.05,
    "Figure 5B Caption: Correlation structure among the most important "
    "acoustic frequency bins, highlighting redundancy and independence.",
    ha="center", fontsize=9
)
plt.tight_layout()
plt.savefig("fig5b_feature_correlation.png", dpi=300, bbox_inches="tight")
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
    "Figure 6 Caption: Precision–recall curves illustrate the trade-off "
    "between detection sensitivity and classification precision for each species.",
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
        s=15,
        color=palette[i]
    )

ax.set_title("Figure 7: 3D PCA Feature Space")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend(bbox_to_anchor=(1.1, 1))

plt.tight_layout()
plt.savefig("fig7_pca_3d.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n✅ All figures generated successfully. Final distinction analysis complete.")