# =============================================================================
# Marine Bioacoustics – DISTINCTION LEVEL FINAL VERSION (WITH FIGURE CAPTIONS)
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
sns.set_style("whitegrid")

BLUE = "#1E90FF"
PINK = "#FF69B4"
PURPLE = "#8A2BE2"

print("="*60)
print("MARINE BIOACOUSTICS – DISTINCTION ANALYSIS")
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

train_median = X_train.median()
X_train = X_train.fillna(train_median)
X_test = X_test.fillna(train_median)

# =============================================================================
# 2. FIGURE 1: PCA VISUALISATION
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
    plt.scatter(X_pca2[mask, 0], X_pca2[mask, 1], label=species, alpha=0.7, s=20)

plt.title(f"Figure 1: PCA Projection of Acoustic Features")
plt.xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.figtext(0.5, -0.05,
            "Figure 1 Caption: Two-dimensional PCA projection showing separation of species based on acoustic features.\n"
            "Each point represents an audio sample, coloured by species class. Clear clustering indicates strong feature separability.",
            ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig1_pca.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 3. FIGURE 2: PCA VARIANCE
# =============================================================================

pca_full = PCA().fit(X_train_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.searchsorted(cum_var, 0.95) + 1

plt.figure(figsize=(9, 5))
plt.plot(range(1, len(cum_var)+1), cum_var*100)
plt.axhline(95, linestyle='--')
plt.axvline(n_95, linestyle='--')
plt.title("Figure 2: PCA Cumulative Variance Explained")
plt.xlabel("Number of Components")
plt.ylabel("Variance (%)")
plt.figtext(0.5, -0.2,
            "Figure 2 Caption: Cumulative variance explained by principal components. The dashed line indicates the number of components required\n"
            "to retain 95% of total variance, demonstrating effective dimensionality reduction.",
            ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig2_variance.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 4. MODELS
# =============================================================================

models = {
    "Random Forest": Pipeline([("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))]),
    "Decision Tree": Pipeline([("clf", DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42))]),
    "KNN": Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=0.95)), ("clf", KNeighborsClassifier(n_neighbors=7))]),
    "SVM": Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=0.95)), ("clf", SVC(kernel="rbf", C=5, probability=True, class_weight="balanced", random_state=42))]),
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=0.95)), ("clf", LogisticRegression(max_iter=5000, solver='saga', class_weight="balanced", random_state=42))])
}

# =============================================================================
# 5. EVALUATION
# =============================================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "test_acc": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "y_pred": y_pred
    }

# =============================================================================
# 6. FIGURE 3: MODEL COMPARISON
# =============================================================================

names = list(results.keys())
cv_means = [results[n]["cv_mean"] for n in names]
cv_stds = [results[n]["cv_std"] for n in names]
accs = [results[n]["test_acc"] for n in names]
f1s = [results[n]["f1"] for n in names]

x = np.arange(len(names))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, cv_means, width, yerr=cv_stds, label="CV Accuracy", capsize=5)
plt.bar(x, accs, width, label="Test Accuracy")
plt.bar(x + width, f1s, width, label="Macro F1")

plt.xticks(x, names, rotation=15)
plt.title("Figure 3: Model Performance Comparison")
plt.legend()
plt.figtext(0.5, -0.2,
            "Figure 3 Caption: Comparison of model performance using cross-validation accuracy, test accuracy, and macro F1 score.\n"
            "Error bars represent standard deviation across folds, indicating model stability.",
            ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig3_models.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 7. BEST MODEL
# =============================================================================

best_name = max(results, key=lambda x: results[x]["test_acc"])
y_pred_best = results[best_name]["y_pred"]

print(classification_report(y_test, y_pred_best, target_names=class_names))

# =============================================================================
# 8. FIGURE 4: CONFUSION MATRIX
# =============================================================================

cm_norm = confusion_matrix(y_test, y_pred_best, normalize='true')

plt.figure(figsize=(12, 10))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='magma',
            xticklabels=class_names, yticklabels=class_names)

plt.title(f"Figure 4: Normalised Confusion Matrix – {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.figtext(0.5, -0.1,
            "Figure 4 Caption: Normalised confusion matrix showing classification performance across species.\n"
            "Values represent the proportion of correctly and incorrectly classified instances per class.",
            ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig4_confusion.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 9. FIGURE 5: FEATURE IMPORTANCE
# =============================================================================

rf = models["Random Forest"].named_steps["clf"]
importances = rf.feature_importances_
idx = np.argsort(importances)[-20:]

plt.figure(figsize=(10, 8))
plt.barh(range(20), importances[idx])
plt.yticks(range(20), [f"Feature {i}" for i in idx])
plt.title("Figure 5: Top 20 Feature Importances")
plt.figtext(0.5, -0.1,
            "Figure 5 Caption: The 20 most important acoustic features identified by the Random Forest model.\n"
            "Higher values indicate greater contribution to classification decisions.",
            ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig5_features.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 10. FIGURE 6: PRECISION-RECALL
# =============================================================================

Y_bin = label_binarize(y_test, classes=range(len(class_names)))
y_score = models[best_name].predict_proba(X_test)

plt.figure(figsize=(10, 7))
colors = sns.color_palette("husl", len(class_names))

for i in range(len(class_names)):
    precision, recall, _ = precision_recall_curve(Y_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, label=class_names[i])

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Figure 6: Precision-Recall Curves")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.figtext(0.5, -0.2,
            "Figure 6 Caption: Precision-recall curves for each species class, illustrating the trade-off between precision and recall.\n"
            "This is particularly useful for evaluating performance on imbalanced datasets.",
            ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig6_pr.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 11. FIGURE 7: 3D PCA
# =============================================================================

pca_3d = PCA(n_components=3)
X_pca3 = pca_3d.fit_transform(X_train_scaled)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

for i, species in enumerate(class_names):
    mask = y_train == i
    ax.scatter(X_pca3[mask, 0], X_pca3[mask, 1], X_pca3[mask, 2], label=species)

ax.set_title("Figure 7: 3D PCA Feature Space")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.figtext(0.5, -0.05,
            "Figure 7 Caption: Three-dimensional PCA visualisation of the acoustic feature space.\n"
            "This highlights clustering patterns and potential overlap between species in reduced dimensions.",
            ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig7_pca3d.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nAll figures generated with captions. Ready for submission.")
