# =============================================================================
# MARINE BIOACOUSTICS – FINAL DISTINCTION ANALYSIS
# ALL 15 FIGURES | CV + TEST | KDE INCLUDED | ALL CAPTIONS | NO TIGHT_LAYOUT
# =============================================================================

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from scipy.spatial import ConvexHull
from math import pi

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ── Pastel Purple / Pink / Blue Palette ─────────────────────────────────────
PALETTE = sns.color_palette([
    "#6A5ACD", "#7FB3FF", "#C77CFF", "#F4A6C0",
    "#AED6F1", "#CE93D8", "#80CBC4", "#FFB7C5",
    "#9FA8DA", "#D7BDE2", "#A3C1AD", "#B39DDB"
])

def save_show(name):
    plt.savefig(name, dpi=300, bbox_inches="tight")
    plt.show()

def label_bars(ax, precision=3):
    for bar in ax.patches:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h,
                f"{h:.{precision}f}",
                ha="center", va="bottom", fontsize=9)

print("="*75)
print("MARINE BIOACOUSTICS – FINAL DISTINCTION ANALYSIS")
print("="*75)

# =============================================================================
# 1. LOAD DATA
# =============================================================================

df = pd.read_csv("SM_1_WMD_Features_and_Labels.csv")
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

print(f"\nSamples: {len(df)} | Features: {X.shape[1]} | Classes: {len(class_names)}")

# =============================================================================
# 2. PCA (2D)
# =============================================================================

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_train_sc)

expl = pca2.explained_variance_ratio_
print(f"PCA 2D variance = {expl.sum()*100:.1f}%")

# ── FIGURE 1: PCA Scatter ───────────────────────────────────────────────────
plt.figure(figsize=(10,7))
for i, sp in enumerate(class_names):
    m = y_train == i
    plt.scatter(X_pca2[m,0], X_pca2[m,1],
                s=18, alpha=0.6,
                color=PALETTE[i], label=sp)

plt.title("Figure 1: PCA Projection of Acoustic Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(bbox_to_anchor=(1.05,1))
plt.text(0.5, -0.15,
         "Figure 1 Caption: PCA projection shows strong overlap between species, "
         "indicating weak linear separability.",
         ha="center", transform=plt.gca().transAxes)
save_show("fig1_pca_scatter.png")

# ── FIGURE 12: PCA KDE ──────────────────────────────────────────────────────
plt.figure(figsize=(11,8))
handles = []

for i, sp in enumerate(class_names):
    m = y_train == i
    if m.sum() > 10:
        sns.kdeplot(x=X_pca2[m,0], y=X_pca2[m,1],
                    fill=True, alpha=0.35,
                    color=PALETTE[i], levels=10)
        handles.append(plt.Line2D([0],[0], lw=6,
                                  color=PALETTE[i], label=sp))

plt.legend(handles=handles, bbox_to_anchor=(1.05,1))
plt.title("Figure 12: PCA Kernel Density Estimation")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.text(0.5, -0.12,
         "Figure 12 Caption: KDE reveals regions of high acoustic density and "
         "extensive inter‑species overlap.",
         ha="center", transform=plt.gca().transAxes)
save_show("fig12_pca_kde.png")

# ── FIGURE 13: PCA Convex Hulls ─────────────────────────────────────────────
plt.figure(figsize=(11,8))
for i, sp in enumerate(class_names):
    pts = X_pca2[y_train == i]
    plt.scatter(pts[:,0], pts[:,1], alpha=0.5, color=PALETTE[i])
    if len(pts) >= 3:
        hull = ConvexHull(pts)
        for s in hull.simplices:
            plt.plot(pts[s,0], pts[s,1], color=PALETTE[i])

plt.title("Figure 13: PCA Convex Hull Boundaries")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.text(0.5, -0.12,
         "Figure 13 Caption: Convex hulls show the spatial extent of species "
         "clusters within PCA space.",
         ha="center", transform=plt.gca().transAxes)
save_show("fig13_pca_hulls.png")

# =============================================================================
# 3. PCA CUMULATIVE VARIANCE (LINE + MARKERS)
# =============================================================================

pca_full = PCA().fit(X_train_sc)
cum = np.cumsum(pca_full.explained_variance_ratio_)
n95 = np.searchsorted(cum, 0.95) + 1

plt.figure(figsize=(10,5))
plt.plot(range(1,len(cum)+1), cum*100, lw=2, color="#6A5ACD")
plt.scatter(range(1,len(cum)+1), cum*100, c=cum*100, cmap="cool", s=20)
plt.axhline(95, ls="--", color="#F4A6C0")
plt.axvline(n95, ls="--", color="#7FB3FF")
plt.title("Figure 3: PCA Cumulative Variance Explained")
plt.xlabel("Number of Components")
plt.ylabel("Variance Explained (%)")
plt.text(0.5, -0.18,
         f"Figure 3 Caption: {n95} components retain 95% of variance, "
         "supporting dimensionality reduction.",
         ha="center", transform=plt.gca().transAxes)
save_show("fig3_pca_variance.png")

# =============================================================================
# 4. MODEL EVALUATION (CV + TEST)
# =============================================================================

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", random_state=42),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n95)),
        ("clf", KNeighborsClassifier(7))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n95)),
        ("clf", SVC(kernel="rbf", C=10, probability=True,
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

print("\nEvaluating models...")
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "test_f1": f1_score(y_test, y_pred, average="macro"),
        "pred": y_pred
    }

    print(
        f"{name:20} | CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | "
        f"Test Macro F1 = {results[name]['test_f1']:.4f}"
    )

best = max(results, key=lambda k: results[k]["test_f1"])
print(f"\nBest Model: {best}")
print(classification_report(y_test, results[best]["pred"], target_names=class_names))

# =============================================================================
# 5. CONFUSION MATRIX
# =============================================================================

plt.figure(figsize=(9,8))
sns.heatmap(confusion_matrix(y_test, results[best]["pred"], normalize="true"),
            cmap="magma", annot=True, fmt=".2f",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Figure 5: Normalised Confusion Matrix")
plt.text(0.5, -0.12,
         "Figure 5 Caption: Normalised confusion matrix shows per‑species accuracy "
         "of the best performing model.",
         ha="center", transform=plt.gca().transAxes)
save_show("fig5_confusion_matrix.png")

# =============================================================================
# 6. PRECISION–RECALL CURVES
# =============================================================================

Y_bin = label_binarize(y_test, classes=range(len(class_names)))
y_score = models[best].predict_proba(X_test)

plt.figure(figsize=(10,8))
for i, sp in enumerate(class_names):
    p, r, _ = precision_recall_curve(Y_bin[:,i], y_score[:,i])
    plt.plot(r, p, lw=2, color=PALETTE[i], label=sp)

plt.legend(bbox_to_anchor=(1.05,1))
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Figure 10: Precision–Recall Curves by Species")
plt.text(0.5, -0.12,
         "Figure 10 Caption: Precision–recall curves illustrate model performance "
         "under class imbalance.",
         ha="center", transform=plt.gca().transAxes)
save_show("fig10_pr_curves.png")

print("\n✅ FINAL CHECK:")
