# =============================================================================
# MARINE BIOACOUSTICS – MASTER PERFORMANCE ANALYSIS (ULTIMATE VERSION)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import combinations
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
sns.set_style("whitegrid")

# Define Aesthetics
BLUE = "#1E90FF"
PINK = "#FF69B4"
PURPLE = "#8A2BE2"

print("="*60)
print("MARINE BIOACOUSTICS – MASTER PERFORMANCE ANALYSIS (ULTIMATE VERSION)")
print("="*60)

# =============================================================================
# 1. DATA PREPARATION
# =============================================================================
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

# Impute missing values
train_median = X_train.median()
X_train = X_train.fillna(train_median)
X_test = X_test.fillna(train_median)

# Print class distribution
print("\nClass Distribution (Training Set):")
print(pd.Series(y_train).map(lambda x: class_names[x]).value_counts())

# =============================================================================
# 2. PCA INSIGHT & 2D VISUALISATION (Figure 1)
# =============================================================================
scaler_vis = StandardScaler()
X_train_scaled = scaler_vis.fit_transform(X_train)

pca_2d = PCA(n_components=2, random_state=42)
X_pca2 = pca_2d.fit_transform(X_train_scaled)
explained = pca_2d.explained_variance_ratio_

print(f"\nPCA Insight: {sum(explained)*100:.1f}% variance captured → " + 
      ("good" if sum(explained) > 0.4 else "poor") + " 2D separability")
print(f"PCA 2D Explained Variance: PC1={explained[0]*100:.2f}%, PC2={explained[1]*100:.2f}%")

plt.figure(figsize=(10, 7))
palette = sns.color_palette("tab20", len(class_names))

for i, species in enumerate(class_names):
    mask = y_train == i
    plt.scatter(X_pca2[mask, 0], X_pca2[mask, 1], label=species, alpha=0.7, s=25, color=palette[i], edgecolor='k', linewidth=0.2)

plt.title(f"Figure 1: PCA Projection of Acoustic Features", fontsize=14, fontweight='bold')
plt.xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.figtext(0.5, -0.05, "Figure 1 Caption: 2D PCA projection showing overlap and clustering of species acoustic signatures.", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig1_pca.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 3. PCA VARIANCE EXPLAINED (Figure 2)
# =============================================================================
pca_full = PCA().fit(X_train_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.searchsorted(cum_var, 0.95) + 1

plt.figure(figsize=(9, 5))
plt.plot(range(1, len(cum_var)+1), cum_var*100, color=BLUE, lw=2)
plt.axhline(95, color='red', linestyle='--')
plt.axvline(n_95, color='orange', linestyle='--')
plt.title("Figure 2: PCA Cumulative Variance Explained")
plt.xlabel("Number of Components")
plt.ylabel("Variance (%)")
plt.figtext(0.5, -0.1, f"Figure 2 Caption: {n_95} components are required to retain 95% of spectral information.", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig2_variance.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 4. MODEL EVALUATION (Figure 3)
# =============================================================================
models = {
    "Random Forest": Pipeline([("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1))]),
    "Decision Tree": Pipeline([("clf", DecisionTreeClassifier(max_depth=15, class_weight="balanced", random_state=42))]),
    "KNN": Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=0.95)), ("clf", KNeighborsClassifier(n_neighbors=7))]),
    "SVM": Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=0.95)), ("clf", SVC(kernel="rbf", C=5, probability=True, class_weight="balanced", random_state=42))]),
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=0.95)), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42, n_jobs=-1))])
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\nEvaluating Models...")
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[name] = {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "test_acc": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "y_pred": y_pred
    }
    print(f"{name:20} | CV: {results[name]['cv_mean']:.4f} | Test: {results[name]['test_acc']:.4f} | F1: {results[name]['f1']:.4f}")

# Figure 3: Model Comparison
names = list(results.keys())
cv_means = [results[n]["cv_mean"] for n in names]
cv_stds = [results[n]["cv_std"] for n in names]
accs = [results[n]["test_acc"] for n in names]
f1s = [results[n]["f1"] for n in names]

x = np.arange(len(names))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, cv_means, width, yerr=cv_stds, label="CV Accuracy (±SD)", capsize=5, color=BLUE)
plt.bar(x, accs, width, label="Test Accuracy", color=PINK)
plt.bar(x + width, f1s, width, label="Macro F1", color=PURPLE)
plt.xticks(x, names, rotation=15)
plt.title("Figure 3: Model Performance Comparison")
plt.legend()
plt.figtext(0.5, -0.15, "Figure 3 Caption: Comparison of CV and Test results across all 5 models.", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig3_models.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 5. BEST MODEL & CONFUSION MATRIX (Figure 4)
# =============================================================================
best_name = max(results, key=lambda x: results[x]["test_acc"])
y_pred_best = results[best_name]["y_pred"]

print(f"\nBest Model: {best_name}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=class_names))

# Figure 4: Normalized Confusion Matrix
cm_norm = confusion_matrix(y_test, y_pred_best, normalize='true')
plt.figure(figsize=(12, 10))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"Figure 4: Normalised Confusion Matrix – {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.figtext(0.5, -0.05, "Figure 4 Caption: Confusion matrix for the top-performing model.", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig4_confusion.png", dpi=300, bbox_inches='tight')
plt.show()

# Top Misclassifications (Figure 4B)
mis_idx = [(i,j,cm_norm[i,j]) for i in range(len(class_names)) for j in range(len(class_names)) if i != j]
mis_idx = sorted(mis_idx, key=lambda x: x[2], reverse=True)[:10]  # top 10
plt.figure(figsize=(12,6))
plt.bar([f"{class_names[i]}→{class_names[j]}" for i,j,_ in mis_idx],
        [v for _,_,v in mis_idx], color=PINK)
plt.xticks(rotation=45)
plt.ylabel("Normalized Misclassification")
plt.title("Figure 4B: Top Misclassifications", fontsize=14, fontweight='bold')
plt.figtext(0.5, -0.05, "Figure 4B Caption: Highlights the species pairs most commonly confused by the best model.", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig4b_top_misclass.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 6. FEATURE IMPORTANCE (Figure 5)
# =============================================================================
rf = models["Random Forest"].named_steps["clf"]
importances = rf.feature_importances_
idx_top20 = np.argsort(importances)[-20:]
values_top20 = importances[idx_top20]
labels_top20 = [f"Freq Bin {i}" for i in idx_top20]

rainbow_colors = plt.cm.cool(np.linspace(0, 1, 20))[::-1]

plt.figure(figsize=(11, 8))
plt.barh(range(20), values_top20, color=rainbow_colors, alpha=0.8)
plt.yticks(range(20), labels_top20, fontsize=10)
plt.xticks(fontsize=10)
plt.title("Figure 5: Top 20 Acoustic Feature Importances\n(Random Forest Gini Importance)", fontsize=14, fontweight='bold')
plt.xlabel("Importance Score (Gini)")
plt.ylabel("Acoustic Frequency Bins")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.figtext(0.5, -0.08, 
            "Figure 5 Caption: Top 20 most influential acoustic frequency bins identified by Random Forest.\n"
            "Colors represent importance level in rainbow order: Light Blue (Highest) -> Pink -> Purple (Lower).",
            ha="center", fontsize=10, style='italic')
plt.tight_layout()
plt.savefig("fig5_features_rainbow.png", dpi=300, bbox_inches='tight')
plt.show()

# Feature Importance Cumulative (Figure 5C)
plt.figure(figsize=(12,6))
sorted_idx = np.argsort(importances)[::-1]
plt.plot(np.cumsum(importances[sorted_idx]), marker='o', color=BLUE)
plt.title("Figure 5C: Cumulative Feature Importance – Random Forest", fontsize=14, fontweight='bold')
plt.xlabel("Features Sorted by Importance")
plt.ylabel("Cumulative Importance")
plt.grid(alpha=0.3)
plt.figtext(0.5, -0.05, "Figure 5C Caption: Illustrates how many features account for majority of predictive power.", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig5c_cum_importance.png", dpi=300, bbox_inches='tight')
plt.show()

# Correlation Heatmap of Top 20 Features (Figure 5B)
plt.figure(figsize=(12,10))
sns.heatmap(X_train.iloc[:, idx_top20].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Figure 5B: Correlation Heatmap of Top 20 Features", fontsize=14, fontweight='bold')
plt.figtext(0.5, -0.05, "Figure 5B Caption: Shows correlations among the most important frequency bins.", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig5b_top20_corr.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 7. PRECISION-RECALL CURVES (Figure 6)
# =============================================================================
Y_bin = label_binarize(y_test, classes=range(len(class_names)))
y_score = models[best_name].predict_proba(X_test)

plt.figure(figsize=(10, 7))
colors = sns.color_palette("husl", len(class_names))
for i in range(len(class_names)):
    precision, recall, _ = precision_recall_curve(Y_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, label=class_names[i], color=colors[i])

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Figure 6: Precision-Recall Curves")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.figtext(0.5, -0.1, "Figure 6 Caption: Evaluating model performance across all species labels.", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig6_pr.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 8. 3D PCA PROJECTION (Figure 7)
# =============================================================================
pca_3d = PCA(n_components=3)
X_pca3 = pca_3d.fit_transform(X_train_scaled)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
for i, species in enumerate(class_names):
    mask = y_train == i
    ax.scatter(X_pca3[mask, 0], X_pca3[mask, 1], X_pca3[mask, 2], label=species, color=palette[i], alpha=0.6)
ax.set_title("Figure 7: 3D PCA Feature Space")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.figtext(0.5, 0.0, "Figure 7 Caption: 3D projection of the acoustic feature manifold.", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig7_pca3d.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 9. ADDITIONAL TOP-TIER PLOTS
# =============================================================================

# PCA 2D with Convex Hulls (Figure 1C)
plt.figure(figsize=(12, 9))
ax = plt.gca()
for i, species in enumerate(class_names):
    mask = y_train == i
    points = X_pca2[mask, :2]
    plt.scatter(points[:,0], points[:,1], s=15, alpha=0.5, color=palette[i], label=species)
    if len(points) >= 3:
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], color=palette[i], alpha=0.8)
plt.title("Figure 1C: PCA with Convex Hulls", fontsize=14, fontweight='bold')
plt.xlabel(f"PC1 ({explained[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}%)")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
plt.figtext(0.5, -0.05, "Figure 1C Caption: Convex hulls indicate the spread and boundary of each species in PCA space.", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig1c_pca_hulls.png", dpi=300, bbox_inches='tight')
plt.show()

# Class Distribution (Train vs Test) Figure 1D
plt.figure(figsize=(10,5))
train_counts = pd.Series(y_train).map(lambda x: class_names[x]).value_counts()
test_counts = pd.Series(y_test).map(lambda x: class_names[x]).value_counts()
df_counts = pd.DataFrame({"Train": train_counts, "Test": test_counts})
df_counts.plot(kind='bar', figsize=(12,6), width=0.7)
plt.title("Figure 1D: Class Distribution – Train vs Test", fontsize=14, fontweight='bold')
plt.xlabel("Species")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.figtext(0.5, -0.05, "Figure 1D Caption: Ensures stratified split preserves species representation.", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig1d_class_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nAll figures and metrics generated successfully. Analysis complete.")