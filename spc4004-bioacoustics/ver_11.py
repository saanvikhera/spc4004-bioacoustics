# =============================================================================
# Marine Bioacoustics – MASTER PERFORMANCE ANALYSIS (FINAL VERSION)
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

# Define Aesthetics
BLUE = "#1E90FF"
PINK = "#FF69B4"
PURPLE = "#8A2BE2"

print("="*60)
print("MARINE BIOACOUSTICS – MASTER PERFORMANCE ANALYSIS")
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

# =============================================================================
# 2. PCA INSIGHT & VISUALISATION (Fig 1)
# =============================================================================
scaler_vis = StandardScaler()
X_train_scaled = scaler_vis.fit_transform(X_train)

pca_2d = PCA(n_components=2, random_state=42)
X_pca2 = pca_2d.fit_transform(X_train_scaled)
explained = pca_2d.explained_variance_ratio_

print(f"\nPCA Insight: {sum(explained)*100:.1f}% variance captured → " + 
      ("good" if sum(explained) > 0.4 else "poor") + " 2D separability")

plt.figure(figsize=(10, 7))
palette = sns.color_palette("cool", len(class_names))

for i, species in enumerate(class_names):
    mask = y_train == i
    plt.scatter(X_pca2[mask, 0], X_pca2[mask, 1], label=species, alpha=0.7, s=20, color=palette[i])

plt.title(f"Figure 1: PCA Projection of Acoustic Features")
plt.xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.figtext(0.5, -0.05, "Figure 1 Caption: 2D PCA projection showing overlap and clustering of species acoustic signatures.", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig1_pca.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 3. PCA VARIANCE (Fig 2)
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

# Figure 3: Comparison with Error Bars
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
# 5. WINNER REPORT (Figure 4)
# =============================================================================
best_name = max(results, key=lambda x: results[x]["test_acc"])
y_pred_best = results[best_name]["y_pred"]

print(f"\nBest Model: {best_name}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=class_names))

# Figure 4: Confusion Matrix (Clean Blue Heatmap)
cm_norm = confusion_matrix(y_test, y_pred_best, normalize='true')
plt.figure(figsize=(12, 10))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', # Changed from Magma to Blues
            xticklabels=class_names, yticklabels=class_names)

plt.title(f"Figure 4: Normalised Confusion Matrix – {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.figtext(0.5, -0.05, "Figure 4 Caption: Confusion matrix for the top-performing model.", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig4_confusion.png", dpi=300, bbox_inches='tight')
plt.show()

# Common Misclassifications Summary
print("\nCommon Misclassifications for " + best_name + ":")
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm_norm[i, j] > 0.1:  # Threshold of 10%
            print(f"{class_names[i]} → {class_names[j]}: {cm_norm[i, j]:.2f}")

# =============================================================================
# 6. FEATURE IMPORTANCE (Fig 5)
# =============================================================================
# =============================================================================
# 9. FIGURE 5: FEATURE IMPORTANCE (Updated Rainbow Aesthetic)
# =============================================================================
print("\nGenerating Feature Importance (Fig 5)...")

# Access the Random Forest classifier from the pipeline
rf = models["Random Forest"].named_steps["clf"]
importances = rf.feature_importances_

# Get the indices of the top 20 features, sorted by importance (lowest to highest)
# We sort lowest to highest so the *most* important is at the *top* of the horizontal chart.
idx_top20 = np.argsort(importances)[-20:]
values_top20 = importances[idx_top20]
labels_top20 = [f"Freq Bin {i}" for i in idx_top20]

# --- THE COLOR MAGIC ---
# We use the 'cool' colormap, which flows beautifully: Light Blue -> Pink -> Purple.
# We reverse the order ([::-1]) so the Light Blue starts at the top (most important).
rainbow_colors = plt.cm.cool(np.linspace(0, 1, 20))[::-1]

plt.figure(figsize=(11, 8)) # Sightly larger for better readability

# Plot horizontal bars
bars = plt.barh(range(20), values_top20, color=rainbow_colors, edgecolor='none', alpha=0.8)

# Formatting the graph
plt.yticks(range(20), labels_top20, fontsize=10)
plt.xticks(fontsize=10)
plt.title("Figure 5: Top 20 Acoustic Feature Importances\n(Random Forest Gini Importance)", fontsize=14, fontweight='bold')
plt.xlabel("Importance Score (Gini)", fontsize=12)
plt.ylabel("Acoustic Frequency Bins", fontsize=12)

# Add a grid only on the X-axis for better alignment
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Add the formal caption at the bottom
plt.figtext(0.5, -0.08, 
            "Figure 5 Caption: The top 20 most influential acoustic frequency bins identified by the Random Forest model.\n"
            "Colors represent importance level in rainbow order: Light Blue (Highest) -> Pink -> Purple (Lower).",
            ha="center", fontsize=10, style='italic', bbox={"facecolor":"white", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.savefig("fig5_features_rainbow.png", dpi=300, bbox_inches='tight')
plt.show()

print("Figure 5 (Rainbow) saved as 'fig5_features_rainbow.png'.")

# =============================================================================
# 7. PRECISION-RECALL (Fig 6)
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
plt.figtext(0.5, -0.1, "Figure 6 Caption: Evaluating model performance across all 12 species labels.", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig6_pr.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 8. 3D PCA (Fig 7)
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
# NEW GRAPH: FULL SPECIES DENSITY (All 12 Whales & Dolphins)
# =============================================================================
print("\nGenerating Full Species Density Plot (All 12 Labels)...")

# 1. Create a clean DataFrame for plotting
X_vis_df = pd.DataFrame(X_pca2[:, :2], columns=["PC1", "PC2"])
X_vis_df["Species"] = le.inverse_transform(y_train)

# 2. Setup the figure
plt.figure(figsize=(12, 9))

# 3. --- The Density Magic ---
# We use a broad palette ('husl') so each of the 12 species has a unique hue.
# alpha=0.3 is crucial so we can see through the "layers" of sound.
kde_plot = sns.kdeplot(
    data=X_vis_df,
    x="PC1", y="PC2",
    hue="Species",
    fill=True,            # Fills the inner "core" of the species call range
    alpha=0.3,             # High transparency to handle 12 overlapping layers
    levels=5,              # Fewer levels makes the plot cleaner and less "busy"
    palette="husl",
    thresh=0.05            # Clips the very edges so the plot isn't a solid block of color
)

# 4. Professional Formatting
plt.title("Figure 1B: Global Kernel Density Estimate (All 12 Marine Species)", fontsize=15, fontweight='bold')
plt.xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")

# Move the legend outside so it doesn't cover the data
plt.legend(title="Species Labels", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

# Formal Distinction-Level Caption
plt.figtext(0.5, -0.05, 
            "Figure 1B Caption: Kernel Density Estimate (KDE) representing the probability density of acoustic features for all 12 species.\n"
            "While Figure 1 shows individual samples, this contour map identifies the core 'spectral niches' occupied by each species,\n"
            "revealing significant overlap in the low-frequency principal components (PC1/PC2).", 
            ha="center", fontsize=9, style='italic')

plt.tight_layout()
plt.savefig("fig1b_full_density.png", dpi=300, bbox_inches='tight')
plt.show()

print("Full Density Plot saved as 'fig1b_full_density.png'.")

print("\nUltimate Analysis Complete. All figures saved.")