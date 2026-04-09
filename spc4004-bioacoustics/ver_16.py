# =============================================================================
# MARINE BIOACOUSTICS – FINAL DISTINCTION ANALYSIS
# (ALL Graphs, ALL Captions, Pop-up + Save, Stable Backend)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import time
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

# ── Pastel Purple–Pink–Blue Palette ──────────────────────────────────────────
PALETTE = sns.color_palette(["#6A5ACD", "#C77CFF", "#7FB3FF", "#B39DDB",
                             "#F4A6C0", "#9FA8DA", "#CE93D8", "#80CBC4",
                             "#FFB7C5", "#B39BC8", "#AED6F1", "#D7BDE2"])

def show_and_save(fig_name):
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(0.8)      # allow GUI to render
    plt.close()

print("="*75)
print("MARINE BIOACOUSTICS – FINAL DISTINCTION ANALYSIS")
print("="*75)

# =============================================================================
# 1. LOAD DATA
# =============================================================================

DATA_PATH = os.path.join("SM_1_WMD_Features_and_Labels.csv")
df = pd.read_csv(DATA_PATH)

feature_cols = [str(i) for i in range(128)]
X = df[feature_cols]
y = df["SPECIES"]

le = LabelEncoder()
y_enc = le.fit_transform(y)
class_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

print(f"\nSamples: {len(df)} | Features: 128 | Classes: {len(class_names)}")
print("\nTraining class distribution:")
print(pd.Series(y_train).map(lambda x: class_names[x]).value_counts())

# =============================================================================
# 2. PCA ANALYSIS + DIAGNOSTICS
# =============================================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_train_scaled)
expl = pca2.explained_variance_ratio_
var_2d = expl.sum() * 100

sep = "good" if var_2d >= 40 else "moderate" if var_2d >= 25 else "poor"

print("\nPCA Insight:")
print(f"PC1 = {expl[0]*100:.2f}% | PC2 = {expl[1]*100:.2f}%")
print(f"Total 2D variance = {var_2d:.1f}% → {sep} separability")

# ── Figure 1: PCA Scatter ────────────────────────────────────────────────────
plt.figure(figsize=(10,7))
for i, sp in enumerate(class_names):
    m = y_train == i
    plt.scatter(X_pca2[m,0], X_pca2[m,1],
                s=20, alpha=0.6,
                color=PALETTE[i], label=sp)

plt.title("Figure 1: PCA Scatter of Training Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(bbox_to_anchor=(1.05,1))
plt.figtext(0.5, -0.08,
            "Figure 1 Caption: Two-dimensional PCA projection reveals extensive overlap "
            "among species, indicating weak linear separability.",
            ha="center", fontsize=9)
show_and_save("fig1_pca_scatter.png")

# ── Figure 1B: PCA KDE ───────────────────────────────────────────────────────
plt.figure(figsize=(11,8))
for i, sp in enumerate(class_names):
    m = y_train == i
    if m.sum() > 10:
        sns.kdeplot(
            x=X_pca2[m,0], y=X_pca2[m,1],
            fill=True, alpha=0.35,
            levels=10, color=PALETTE[i],
            label=sp
        )

plt.title("Figure 1B: PCA Density Estimation (KDE)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(bbox_to_anchor=(1.05,1))
plt.figtext(0.5, -0.07,
            "Figure 1B Caption: Kernel density estimation highlights regions of high species density "
            "and strong inter-class overlap.",
            ha="center", fontsize=9)
show_and_save("fig1b_pca_kde.png")

# ── Figure 1C: PCA Convex Hulls ───────────────────────────────────────────────
plt.figure(figsize=(11,8))
for i, sp in enumerate(class_names):
    m = y_train == i
    pts = X_pca2[m]
    plt.scatter(pts[:,0], pts[:,1], s=15,
                alpha=0.5, color=PALETTE[i], label=sp)
    if len(pts) >= 3:
        hull = ConvexHull(pts)
        for s in hull.simplices:
            plt.plot(pts[s,0], pts[s,1], color=PALETTE[i])

plt.title("Figure 1C: PCA Convex Hull Boundaries")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(bbox_to_anchor=(1.05,1), fontsize="small")
plt.figtext(0.5, -0.07,
            "Figure 1C Caption: Convex hulls illustrate the spatial extent of each species "
            "in reduced feature space.",
            ha="center", fontsize=9)
show_and_save("fig1c_pca_hulls.png")

# =============================================================================
# 3. PCA VARIANCE CURVE
# =============================================================================

pca_full = PCA().fit(X_train_scaled)
cum = np.cumsum(pca_full.explained_variance_ratio_)
n95 = np.searchsorted(cum, 0.95) + 1

print(f"\n{n95} PCA components preserve ~95% variance")

plt.figure(figsize=(9,5))
plt.plot(cum*100, color="#6A5ACD", lw=2)
plt.axhline(95, ls="--", color="#C77CFF")
plt.axvline(n95, ls="--", color="#7FB3FF")
plt.title("Figure 2: PCA Cumulative Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance (%)")
plt.figtext(0.5, -0.12,
            "Figure 2 Caption: The PCA variance curve shows the number of components "
            "required to retain 95% of information.",
            ha="center", fontsize=9)
show_and_save("fig2_pca_variance.png")

# =============================================================================
# 4. MODELS
# =============================================================================

models = {
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=42, n_jobs=-1))
    ]),
    "Decision Tree": Pipeline([
        ("clf", DecisionTreeClassifier(
            class_weight="balanced", random_state=42))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n95)),
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n95)),
        ("clf", SVC(
            kernel="rbf", C=10, probability=True,
            class_weight="balanced", random_state=42))
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n95)),
        ("clf", LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=42))
    ])
}

cv = StratifiedKFold(5, shuffle=True, random_state=42)
results = {}

print("\nEvaluating Models...")
for name, m in models.items():
    scores = cross_val_score(m, X_train, y_train, cv=cv, n_jobs=-1)
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)

    results[name] = {
        "cv": scores.mean(),
        "acc": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "pred": y_pred
    }

    print(f"{name:18} | CV={scores.mean():.4f} | Test={results[name]['acc']:.4f} | Macro F1={results[name]['f1']:.4f}")

best = max(results, key=lambda k: results[k]["f1"])
print(f"\nBest model selected (Macro F1): {best}")

print("\nClassification report:")
print(classification_report(y_test, results[best]["pred"], target_names=class_names))

# =============================================================================
# FIGURE 10: SVM PREDICTIVE POWER (F1-SCORE PER SPECIES)
# =============================================================================

from sklearn.metrics import classification_report

report = classification_report(
    y_test,
    results[best]["pred"],
    target_names=class_names,
    output_dict=True
)

species = []
f1_scores = []

for sp in class_names:
    species.append(sp)
    f1_scores.append(report[sp]["f1-score"])

mean_f1 = np.mean(f1_scores)

plt.figure(figsize=(12, 5))
bars = plt.bar(
    species,
    f1_scores,
    color=PALETTE[:len(species)],
    edgecolor="black"
)

plt.axhline(
    mean_f1,
    linestyle="--",
    color="red",
    linewidth=1.5,
    label=f"Mean F1 ({mean_f1:.2f})"
)

plt.ylim(0, 1.0)
plt.ylabel("F1-Score")
plt.xlabel("Species")
plt.title("Figure 10: SVM Predictive Power (F1-Score per Species)")
plt.legend()

plt.figtext(
    0.5, -0.2,
    "Figure 10 Caption: Per-species F1-scores for the best-performing SVM (RBF) model. "
    "The dashed line indicates the mean macro F1-score, highlighting species that perform "
    "above and below the overall average.",
    ha="center", fontsize=9
)

plt.tight_layout()
plt.savefig("fig10_svm_f1_per_species.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n✅ All figures popped up and saved successfully.")
print("✅ Diagnostic console output ready for report.")
print("✅ Final distinction‑level analysis complete.")