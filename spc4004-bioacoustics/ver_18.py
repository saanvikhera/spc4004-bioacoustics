# =============================================================================
# MARINE BIOACOUSTICS – FINAL DISTINCTION ANALYSIS (FULL PIPELINE)
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
    precision_recall_curve, balanced_accuracy_score
)
from sklearn.inspection import permutation_importance

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

PALETTE = [
    "#C3B1E1",  # soft lavender
    "#B8D4F5",  # powder blue
    "#F5B8D0",  # blush pink
    "#B5EAD7",  # mint
    "#FFD6BA",  # peach
    "#FFDDE1",  # rose
    "#C5CAE9",  # periwinkle
    "#E1BEE7",  # lilac
    "#B2EBF2",  # baby blue
    "#F8BBD0",  # light pink
    "#D7CCC8",  # warm grey
    "#DCEDC8",  # sage
]

def save_show(name):
    plt.savefig(name, dpi=300, bbox_inches="tight")
    plt.show()

print("="*75)
print("MARINE BIOACOUSTICS – FINAL DISTINCTION ANALYSIS")
print("="*75)

# =============================================================================
# 1. LOAD DATA
# =============================================================================

df = pd.read_csv("SM_1_WMD_Features_and_Labels.csv")
X = df[[str(i) for i in range(128)]]
y = df["SPECIES"]

le = LabelEncoder()
y_enc = le.fit_transform(y)
class_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

# =============================================================================
# 2. CLASS DISTRIBUTION — FIGURE 1
# =============================================================================

train_counts = pd.Series(y_train).map(lambda i: class_names[i]).value_counts()
test_counts  = pd.Series(y_test ).map(lambda i: class_names[i]).value_counts()

pd.DataFrame({"Train": train_counts, "Test": test_counts}).plot(
    kind="bar", figsize=(10,5), color=[PALETTE[1], PALETTE[2]]
)

plt.title("Figure 1: Class Distribution")
save_show("fig1_class_distribution.png")

# =============================================================================
# 3. PCA — FIGURES
# =============================================================================

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

# 2D PCA
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_train_sc)

plt.figure(figsize=(10,7))
for i, sp in enumerate(class_names):
    pts = X_pca2[y_train == i]
    plt.scatter(pts[:,0], pts[:,1], s=18,
                color=PALETTE[i], alpha=0.6, label=sp)

plt.title("Figure 2: PCA Projection")
plt.legend(bbox_to_anchor=(1.05,1))
save_show("fig2_pca.png")

# KDE
plt.figure(figsize=(10,8))
for i, sp in enumerate(class_names):
    pts = X_pca2[y_train == i]
    sns.kdeplot(x=pts[:,0], y=pts[:,1],
                fill=True, alpha=0.3,
                bw_adjust=1, color=PALETTE[i])

plt.title("Figure 3: PCA KDE")
save_show("fig3_pca_kde.png")

# Variance
pca_full = PCA().fit(X_train_sc)
cum = np.cumsum(pca_full.explained_variance_ratio_)
n95 = np.searchsorted(cum, 0.95) + 1

plt.figure()
plt.plot(cum*100)
plt.axhline(95, ls="--")
plt.axvline(n95, ls="--")
plt.title("Figure 4: PCA Variance")
save_show("fig4_pca_variance.png")

# =============================================================================
# 4. MODEL COMPARISON (ALL MODELS) — FIGURE 5
# =============================================================================

models = {
    "Random Forest": RandomForestClassifier(200, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=15, class_weight="balanced", random_state=42),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n95)),
        ("clf", KNeighborsClassifier(7))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n95)),
        ("clf", SVC(kernel="rbf", C=10,
                    probability=True, class_weight="balanced"))
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n95)),
        ("clf", LogisticRegression(max_iter=3000,
                                   class_weight="balanced"))
    ])
}

cv = StratifiedKFold(5, shuffle=True, random_state=42)

results = {}
print("\nEvaluating Models...")

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "cv_mean": cv_scores.mean(),
        "test_acc": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro")
    }

    print(f"{name:20} | CV={cv_scores.mean():.4f} | "
          f"Test={results[name]['test_acc']:.4f} | "
          f"Macro F1={results[name]['macro_f1']:.4f}")

# Plot
plt.figure()
plt.bar(results.keys(),
        [results[m]["macro_f1"] for m in results],
        color=PALETTE[:len(results)])
plt.xticks(rotation=20)
plt.title("Figure 5: Model Comparison (Macro F1)")
save_show("fig5_model_comparison.png")

# =============================================================================
# 5. SELECT BEST MODEL + GRID SEARCH
# =============================================================================

best_name = max(results, key=lambda k: results[k]["macro_f1"])
print(f"\nBest baseline model: {best_name}")

# Only tune if SVM (most likely)
param_grid = {
    "clf__C": [1, 10],
    "clf__gamma": ["scale", "auto"]
}

grid = GridSearchCV(models[best_name], param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# =============================================================================
# 6. CV BOXPLOT — FIGURE 6
# =============================================================================

cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv)

plt.figure()
sns.boxplot(cv_scores)
plt.title("Figure 6: CV Score Distribution")
save_show("fig6_cv_boxplot.png")

# =============================================================================
# 7. FINAL EVALUATION
# =============================================================================

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nFinal Model Performance:")
print(classification_report(y_test, y_pred, target_names=class_names))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))

# =============================================================================
# 8. CONFUSION MATRIX — FIGURE 7
# =============================================================================

sns.heatmap(confusion_matrix(y_test, y_pred, normalize="true"),
            annot=True, fmt=".2f", cmap="magma")
plt.title("Figure 7: Confusion Matrix")
save_show("fig7_confusion_matrix.png")

# =============================================================================
# 9. PRECISION–RECALL — FIGURE 8
# =============================================================================

Y_bin = label_binarize(y_test, classes=range(len(class_names)))
y_score = best_model.predict_proba(X_test)

plt.figure()
for i in range(len(class_names)):
    p, r, _ = precision_recall_curve(Y_bin[:,i], y_score[:,i])
    plt.plot(r, p)

plt.title("Figure 8: Precision–Recall")
save_show("fig8_pr.png")

# =============================================================================
# 10. MISCLASSIFICATIONS — FIGURE 9
# =============================================================================

cm = confusion_matrix(y_test, y_pred, normalize="true")

mis = [(i,j,cm[i,j]) for i in range(len(class_names))
       for j in range(len(class_names)) if i!=j]

top = sorted(mis, key=lambda x:x[2], reverse=True)[:10]

plt.figure()
plt.bar([f"{class_names[i]}→{class_names[j]}" for i,j,_ in top],
        [v for _,_,v in top])
plt.xticks(rotation=60, ha='right')
plt.title("Figure 9: Misclassifications")
save_show("fig9_mis.png")

# =============================================================================
# 11. FEATURE IMPORTANCE — FIGURES 10 & 11
# =============================================================================

rf = RandomForestClassifier(200, class_weight="balanced")
rf.fit(X_train, y_train)

imp = rf.feature_importances_
idx = np.argsort(imp)[-20:]

plt.figure()
plt.barh(range(20), imp[idx])
plt.title("Figure 10: RF Importance")
save_show("fig10_rf.png")

perm = permutation_importance(best_model, X_test, y_test)
idx2 = np.argsort(perm.importances_mean)[-20:]

plt.figure()
plt.barh(range(20), perm.importances_mean[idx2])
plt.title("Figure 11: Permutation Importance")
save_show("fig11_perm.png")

# =============================================================================
# 12. 3D PCA — FIGURE 12
# =============================================================================

pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_train_sc)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for i in range(len(class_names)):
    pts = X_pca3[y_train==i]
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], color=PALETTE[i], alpha=0.6, label=sp)
ax.set_title("Figure 12: 3D PCA")
save_show("fig12_pca3d.png")

print("\n✅ ALL FIGURES GENERATED + MODELS COMPARED + TUNING COMPLETE")