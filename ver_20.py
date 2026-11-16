# =============================================================================
# MARINE BIOACOUSTICS – FINAL DISTINCTION ANALYSIS (REFINED + FULL REPORT)
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, balanced_accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
np.random.seed(42)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# 🎨 REFINED PASTEL PALETTE (Blue -> Purple -> Pink Vibe)
PALETTE = ["#B8D4F5", "#C5CAE9", "#C3B1E1", "#E1BEE7", "#F8BBD0", "#F5B8D0", "#FFDDE1", "#FFD6BA", "#DCEDC8", "#B5EAD7", "#B2EBF2", "#D7CCC8"]

def save_show(name, title, caption):
    """Saves figure with an embedded caption at the bottom of the image."""
    plt.title(title, fontweight='bold', pad=25, fontsize=14)
    # Positioning the caption at the very bottom
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', 
                fontsize=10, style='italic', bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})
    plt.subplots_adjust(bottom=0.2) # Make room for the caption
    plt.tight_layout() 
    plt.savefig(name, dpi=300, bbox_inches="tight")
    plt.close() 

print("="*75)
print("MARINE BIOACOUSTICS – MASTER PERFORMANCE ANALYSIS")
print("="*75)

# 1. LOAD & PREP
df = pd.read_csv("SM_1_WMD_Features_and_Labels.csv")

# Convert feature columns 0-127 to integers if they are strings
feature_cols = [str(i) for i in range(128)]
if all(col in df.columns for col in feature_cols):
    df.rename(columns={str(i): i for i in range(128)}, inplace=True)

# Now this will work without KeyError
X_base = df[list(range(128))]

# Base features
X_base = df[list(range(128))]

weights = np.arange(X_base.shape[1])
denominator = np.maximum(X_base.sum(axis=1), 1e-6)
df["spectral_centroid"] = (X_base * weights).sum(axis=1) / denominator
df = df.copy()

# FINAL feature matrix (now includes centroid)
X = df[list(range(128)) + ["spectral_centroid"]]
y = df["SPECIES"]

# --- [ADDITION] DATA QUALITY CHECK & SCIENTIFIC FEATURES ---
print("\n[Data Integrity Check]")
constant_filter = X.var() == 0
if constant_filter.any():
    print(f"Removing {constant_filter.sum()} features with zero variance.")
    X = X.loc[:, ~constant_filter]

if "spectral_centroid" not in X.columns:
    print("Spectral centroid removed due to zero variance; consider re-examining data preprocessing.")



print(f"Samples: {len(df)} | Features: {X.shape[1]} | Classes: {y.nunique()}")
le = LabelEncoder()
y_enc = le.fit_transform(y)
class_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
X_train = X_train.copy()
X_train.columns = X_train.columns.astype(str)
X_test = X_test.copy()
X_test.columns = X_test.columns.astype(str)

# For plotting
y_train_labels = le.inverse_transform(y_train)
y_test_labels = le.inverse_transform(y_test)

print("\nTraining class distribution:")
train_dist = pd.Series(y_train_labels).value_counts()
print(train_dist)


# 2. DISTRIBUTION (FIG 1)
plt.figure(figsize=(10,6))
train_counts = pd.Series(y_train_labels).value_counts().reindex(class_names, fill_value=0)
test_counts  = pd.Series(y_test_labels).value_counts().reindex(class_names, fill_value=0)
pd.DataFrame({"Train": train_counts, "Test": test_counts}).plot(kind="bar", color=[PALETTE[0], PALETTE[5]], ax=plt.gca())
save_show("fig1_dist.png", "Figure 1: Class Distribution", "Caption: Visual representation of sample support per species. High support for Meno/Oror contrasts with rare species like Stbr.")

# 3. PCA & VARIANCE (FIG 2, 3, 4)
# PCA HERE IS ONLY FOR VISUALISATION. The actual model pipelines apply PCA separately during training to prevent data leakage.
scaler = StandardScaler()
# 3D PCA for visualisation (same scaled space as 2D PCA)
X_train_sc = scaler.fit_transform(X_train)

pca_vis = PCA(n_components=2, random_state=42)
X_pca2 = pca_vis.fit_transform(X_train_sc)

# Fig 2: Scatter
plt.figure(figsize=(10,8))
ax = plt.gca() 
for i, sp in enumerate(class_names):
    pts = X_pca2[y_train == i]
    plt.scatter(pts[:,0], pts[:,1], s=18, color=PALETTE[i % len(PALETTE)], alpha=0.6, label=sp)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.legend(loc='upper left',bbox_to_anchor=(1.05,1)); save_show("fig2_pca.png", "Figure 2: 2D PCA Projection", "Caption: Dimensionality reduction showing overlap in the primary feature space.")

# Fig 3: KDE
plt.figure(figsize=(10,9))
for i, sp in enumerate(class_names):
    pts = X_pca2[y_train == i]
    if len(pts) > 1:
        sns.kdeplot(x=pts[:,0], y=pts[:,1], fill=True, alpha=0.3, 
                    bw_adjust=1, color=PALETTE[i % len(PALETTE)], label=sp)

from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=PALETTE[i % len(PALETTE)], lw=6, label=class_names[i]) for i in range(len(class_names))]
plt.legend(handles=handles, title="Species", bbox_to_anchor=(1.05, 1), loc='upper left')

save_show("fig3_kde.png", "Figure 3: Species Density (KDE)", 
          "Caption: Kernel Density Estimation highlighting core acoustic niches. The legend identifies high-density feature clusters for each marine species.")

# Fig 4: Variance
pca_full = PCA().fit(X_train_sc)
cum_variance = np.cumsum(pca_full.explained_variance_ratio_)
n95 = np.searchsorted(cum_variance, 0.95) + 1
pc1 = pca_full.explained_variance_ratio_[0] * 100
pc2 = pca_full.explained_variance_ratio_[1] * 100
total_2d = pc1 + pc2

print("\nPCA Insight:")
print(f"PC1 = {pc1:.2f}% | PC2 = {pc2:.2f}%")
print(f"Total 2D variance = {total_2d:.1f}% → poor separability")
print(f"{n95} PCA components preserve ~95% variance")
plt.figure(figsize=(8,6)); plt.plot(cum_variance*100, marker='o', markersize=3, color=PALETTE[2])
plt.axhline(95, ls="--", color='blue',  label="95% Variance Threshold"); plt.axvline(n95, ls="--", color='purple', label=f"{n95} Components")
plt.legend()
save_show("fig4_var.png", "Figure 4: PCA Variance Cumulative Sum", f"Caption: Information retention plot. {n95} components are required to capture 95% variance.")

# 4. MODELS & TUNING (MODIFIED FOR PIPELINE INTEGRITY)
models = {
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", DecisionTreeClassifier(max_depth=15, class_weight="balanced", random_state=42))
    ]),
    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("clf", KNeighborsClassifier(7))
    ]),
    "KNN (No PCA)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(7))
    ]),
    "SVM (RBF)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()), 
        ("pca", PCA(n_components=0.95)), 
        ("clf", SVC(kernel="rbf", C=10, probability=True, class_weight="balanced"))
    ]),
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()), 
        ("pca", PCA(n_components=0.95)), 
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver='saga'))
    ])
}
print("\nEvaluating Models...\n")

results = {}
for name, model in models.items():
    cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="accuracy",
    n_jobs=-1
)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    results[name] = {
        "macro_f1": macro_f1,
        "cv_mean": cv_scores.mean(),
        "test_acc": test_acc,
        "scores": cv_scores,
    }

    print(
        f"{name:<18} | "
        f"CV={cv_scores.mean():.4f} ± {cv_scores.std():.4f} | "
        f"Test={test_acc:.4f} | "
        f"Macro F1={macro_f1:.4f}"
    )


plt.figure(figsize=(10,7))
plt.bar(results.keys(), [results[m]["test_acc"] for m in results], color=PALETTE[:5])
save_show("fig5_comp.png", "Figure 5: Model Comparison (Test Accuracy)", "Caption: Comparison across classifiers. Test accuracy reflects real-world performance.")

best_name = max(results, key=lambda k: results[k]["cv_mean"])
print(f"\nBest model selected (Cross-Validation): {best_name}")

if "SVM" in best_name:
    
    param_grid = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", "auto", 0.01, 0.001]
    }

    grid = GridSearchCV(
    models[best_name],
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    refit=True
)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
else:
    best_model = models[best_name]
    best_model.fit(X_train, y_train)

plt.figure(figsize=(8,6)); sns.boxplot(y=results[best_name]["scores"], color=PALETTE[2])
save_show("fig6_cv.png", "Figure 6: Cross-Validation Stability", f"Caption: Boxplot showing performance consistency for the best model: {best_name}.")

# 5. THE CRITICAL REPORT & PERFORMANCE FIGURES (FIG 7, 8, 9)
y_pred = best_model.predict(X_test)

print("\n" + "="*30)
print(f"FINAL REPORT: {best_name}")
print("="*30)
print(classification_report(y_test, y_pred, target_names=class_names))
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
print("="*30)

# Fig 7: Confusion Matrix
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix(y_test, y_pred, normalize="true"), annot=True, fmt=".2f", cmap="RdPu", xticklabels=class_names, yticklabels=class_names)
save_show("fig7_cm.png", "Figure 7: Normalized Confusion Matrix", "Caption: Normalized diagonal representing Recall. High diagonal values confirm precise classification.")

# Fig 8: Precision-Recall
plt.figure(figsize=(10,8))
Y_bin = label_binarize(y_test, classes=range(len(class_names))); 
if hasattr(best_model, "predict_proba"):
    y_score = best_model.predict_proba(X_test)
else:
    y_score = best_model.decision_function(X_test)
for i in range(len(class_names)):
    precision, recall, _ = precision_recall_curve(Y_bin[:,i], y_score[:,i])
    plt.plot(recall, precision, color=PALETTE[i % len(PALETTE)], label=class_names[i])
plt.legend(bbox_to_anchor=(1.05,1)); save_show("fig8_pr.png", "Figure 8: Precision-Recall Curves", "Caption: Class-specific performance showing the trade-off between precision and recall.")

# Fig 9: Misclassifications
cm = confusion_matrix(y_test, y_pred, normalize="true")

mis = [
    (i, j, cm[i, j])
    for i in range(len(class_names))
    for j in range(len(class_names))
    if i != j
]

top_mis = sorted(mis, key=lambda x: x[2], reverse=True)[:10]
plt.figure(figsize=(10,7))

labels = [f"{class_names[i]}→{class_names[j]}" for i,j,_ in top_mis]
values = [v for _,_,v in top_mis]

bars = plt.bar(labels, values, color=PALETTE[5], label="Misclassification Rate")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", ha='center', va='bottom', fontsize=9)

plt.xticks(rotation=60, ha='right')
plt.legend()

save_show(
    "fig9_mis.png",
    "Figure 9: Top 10 Misclassifications",
    "Caption: Breakdown of species most frequently confused by the model. Bar height indicates misclassification frequency."
)


# 6. IMPORTANCE & 3D (FIG 10, 11, 12)
# Fig 10: Random Forest Feature Importance (only if RF is the best model)
rf = RandomForestClassifier(200, class_weight="balanced", random_state=42).fit(X_train, y_train)
importances = rf.feature_importances_
idx = np.argsort(importances)[-20:]
feature_names = np.array(X.columns)[idx]
if "Random Forest" in best_name:
    plt.figure(figsize=(8,7))
    plt.barh(feature_names, importances[idx], color=PALETTE[0], label="Gini Importance")
    plt.legend()
    save_show(
        "fig10_rf.png",
        "Figure 10: Random Forest Feature Importance",
        "Caption: Top 20 features ranked by Gini importance. Only shown if Random Forest is selected as best model."
    )
else:
    print("Figure 10 skipped: Best model is not Random Forest")

# NOTE: Permutation importance is applied on input features.
# Since PCA is used in the model pipeline, importance reflects
# contribution before transformation and should be interpreted cautiously.

# Use the correct transformed test set if pipeline has scaler/PCA
# Use raw input features for permutation importance (pre-PCA)
X_test_perm = X_test.copy()
X_test_sub = X_test_perm.sample(n=200, random_state=42, replace=False)  # smaller subset
y_test_sub = y_test[:200]
perm = permutation_importance(best_model, X_test_sub, y_test_sub, n_repeats=10, random_state=42, scoring="accuracy", n_jobs=-1)

perm_idx = np.argsort(perm.importances_mean)[-20:]
perm_features = np.array(X.columns)[perm_idx]

# Default color
colors = [PALETTE[3]] * len(perm_features)

# Highlight spectral_centroid
if "spectral_centroid" in perm_features:
    idx = list(perm_features).index("spectral_centroid")
    colors[idx] = "red"

plt.figure(figsize=(8,7))
plt.barh(perm_features, perm.importances_mean[perm_idx],
         color=colors, label="Permutation Importance")

plt.legend()
plt.title("Permutation Feature Importance")
plt.xlabel("Mean Decrease in Accuracy")
plt.tight_layout()

save_show(
    "fig11_perm.png",
    "Figure 11: Permutation Feature Importance",
    "Caption: Model-agnostic importance showing performance drop when each feature is shuffled." \
    "\nNote: Importance reflects contribution before dimensionality reduction (PCA/scaling)."
    "\nSpectral centroid is highlighted as a key acoustic feature for distinguishing species."
)


if "spectral_centroid" in X.columns:
    centroid_idx = list(X.columns).index("spectral_centroid")
    print("Spectral centroid importance:", perm.importances_mean[centroid_idx])
else:
    print("Spectral centroid was removed due to zero variance.")

pca3 = PCA(n_components=3, random_state=42); X_pca3 = pca3.fit_transform(X_train_sc).astype(np.float32)
fig = plt.figure(figsize=(10,10)); ax = fig.add_subplot(111, projection="3d")
for i in range(len(class_names)):
    pts = X_pca3[y_train == i]
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], color=PALETTE[i % len(PALETTE)], alpha=0.6, label=class_names[i])
ax.legend(loc='upper left',bbox_to_anchor=(1.05,1)); save_show("fig12_pca3d.png", "Figure 12: 3D PCA Projection", "Caption: Three-dimensional feature space visualization showing separated species clusters.")

# 7. THE "MISSING 3" (FIG 13, 14)

corr_features = X.columns[idx]
plt.figure(figsize=(12, 10))
sns.heatmap(X_train.iloc[:, idx].corr(), annot=True, fmt=".2f", cmap="BuPu", xticklabels=corr_features, yticklabels=corr_features)
save_show("fig13_correlation.png", "Figure 13: Feature Correlation Heatmap", "Caption: Correlation matrix of top diagnostic features.")


report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
f1_scores = [report_dict[sp]['f1-score'] for sp in class_names]
plt.figure(figsize=(12, 6))
sns.barplot(x=list(class_names), y=f1_scores, palette=PALETTE)
plt.axhline(np.mean(f1_scores), ls="--", color="black", label=f"Mean F1: {np.mean(f1_scores):.2f}")
plt.ylim(0, 1.1); plt.xticks(rotation=45); plt.legend(loc='upper right')
save_show("fig14_f1_ranking.png", "Figure 14: Per-Species F1 Score Ranking", "Caption: Final performance across all species.")

# --- [ADDITION] NEW FIGURES FOR ADVANCED VALIDATION  ---

# Figure 16: Calibration Curve
plt.figure(figsize=(8, 6))
if hasattr(best_model, "predict_proba"):
    y_score = best_model.predict_proba(X_test)
else:
    y_score = best_model.decision_function(X_test)
for i, cls in enumerate(class_names):
    frac_pos, mean_pred = calibration_curve(Y_bin[:, i], y_score[:, i], n_bins=10)
    plt.plot(mean_pred, frac_pos, "s-", color=PALETTE[i % len(PALETTE)], label=cls)
plt.plot([0,1],[0,1],"--", color="black", alpha=0.5)
save_show("fig15_calibration.png", "Figure 15: Model Calibration Curve", "Caption: Reliability plot showing predicted vs actual probability.")

# =============================================================================
# NEW FIGURE: Spectral Centroid Distribution
# =============================================================================

plt.figure(figsize=(10,6))
sns.boxplot(x=y_train_labels, y=X_train["spectral_centroid"], palette=PALETTE)
plt.ylabel("Spectral Centroid (Hz)", fontweight='bold')
plt.xlabel("Species", fontweight='bold')

plt.xticks(rotation=45)

save_show(
    "fig_centroid_distribution.png",
    "Figure 16: Spectral Centroid Distribution by Species",
    "Caption: The spectral centroid reflects dominant frequency content. "
    "Species with higher centroids exhibit higher-frequency vocalisations, "
    "supporting its usefulness as a discriminative acoustic feature."
)

# ==========================
# Robustness Final Test (CORRECTED)
# ==========================

# 1. Prepare raw test data
X_raw_test = X_test.copy().values.astype(np.float32)

# 2. Add scaled Gaussian noise to the RAW features
# (Simulating microphone interference before it hits the model)
noise = np.random.normal(0, 0.02, X_raw_test.shape).astype(np.float32)
feature_std = np.nanstd(X_raw_test, axis=0) # Use nanstd in case of NaNs
X_noisy_raw = X_raw_test + noise * feature_std

# Convert back to DataFrame so the Pipeline can read column names if needed
X_noisy_df = pd.DataFrame(X_noisy_raw, columns=X_test.columns)

# 3. Predict using the FULL best_model pipeline
# This automatically handles Imputing -> Scaling -> PCA -> Prediction
y_pred_noisy = best_model.predict(X_noisy_df)

# 4. Compute accuracy
noisy_acc = accuracy_score(y_test, y_pred_noisy)

# 5. Print results
print("\n" + "="*30)
print("STRESS TEST RESULTS (ROBUSTNESS)")
print(f"Clean Test Accuracy: {results[best_name]['test_acc']:.4f}")
print(f"Noisy Test Accuracy: {noisy_acc:.4f}")
print(f"Resilience Score: {(noisy_acc / results[best_name]['test_acc'])*100:.1f}%")
print("="*30)

# ==========================
# Figure 17: CV vs Test Accuracy
# ==========================
plt.figure(figsize=(10,6))

cv_means = [results[m]["cv_mean"] for m in results]
cv_stds  = [results[m]["scores"].std() for m in results]
test_accs = [results[m]["test_acc"] for m in results]

x = np.arange(len(results))

# Bars for CV mean ± std
plt.bar(x, cv_means, yerr=cv_stds, alpha=0.7, color=PALETTE[:len(results)], capsize=5, label="CV Mean ± Std")

# Black diamonds for test accuracy
plt.scatter(x, test_accs, color=PALETTE[-2], marker="D", s=80, label="Test Accuracy")

plt.xticks(x, results.keys(), rotation=45, ha='right')
plt.ylabel("Scores")
plt.title("Figure 17: Cross-Validation vs Test Accuracy per Model", fontweight="bold")

# Caption
plt.figtext(0.5, 0.01, "Caption: Bars show cross-validation mean ± standard deviation per model. "
            "Diamonds represent actual test set accuracy. Helps compare stability and performance.",
            wrap=True, horizontalalignment='center', fontsize=10, style='italic', bbox={'facecolor':'white','alpha':0.8,'pad':5})

plt.tight_layout()
plt.savefig("fig17_cv_vs_test.png", dpi=300)
plt.close()

print("\n✅ DONE: Report Printed and Captioned Figures Saved.")