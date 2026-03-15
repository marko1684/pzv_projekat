# Data Science Exam Cheat Sheet

---

## General Workflow (almost every problem)

```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
```

### Standard Pipeline

```python
# 1. Load
data = pd.read_csv('file.csv')

# 2. Explore
data.head()
data.describe()
data.info()
data.isna().sum()

# 3. Handle missing values (see section below)

# 4. Split X and Y
Y = data['target_col']
X = data.drop('target_col', axis=1)

# 5. Encode target if needed
Y = Y.replace({'B': 0, 'M': 1})

# 6. Train/test split (BEFORE any preprocessing)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, stratify=Y, random_state=42)

# 7. Preprocessing (fit on train, transform both)
# 8. Train model
# 9. Evaluate
```

---

## 02 — Missing Values

```python
df.isna().any().any()    # any missing?
df.isna().sum()          # count per column
```

| Technique | Code |
|---|---|
| Drop rows | `df.dropna()` |
| Fill constant | `df['col'].fillna(0)` |
| Forward fill | `df['col'].ffill()` |
| Backward fill | `df['col'].bfill()` |
| Mean fill | `df['col'].fillna(df['col'].mean())` |
| Median fill | `df['col'].fillna(df['col'].median())` |
| Grouped mean | `df['col'].fillna(df.groupby('g')['col'].transform('mean'))` |

### Iterative Imputation

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

imputer = IterativeImputer(RandomForestRegressor(), max_iter=100, tol=0.01, random_state=0)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

---

## 02 — Similarity / Distance Measures

- **Nominal**: similarity = 1 if equal, 0 otherwise
- **Ordinal**: `d(p,q) = |rank(p) - rank(q)| / (n-1)`
- **Minkowski**: `(sum |x_k - y_k|^r)^(1/r)` — r=1 Manhattan, r=2 Euclidean
- **Binary**: SMC, Jaccard, Cosine, Hamming

---

## 03 — Normalization

```python
# MinMax (manual)
def minmax(X):
    return (X - X.min()) / (X.max() - X.min())

# Clipping
X.clip(lower=min_val, upper=max_val)

# Log scaling
np.log(X)

# Z-score / Standardization
def z_score(X):
    return (X - X.mean()) / X.std()
```

### sklearn scalers

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()  # or StandardScaler()
scaler.fit(X_train)                # fit on TRAIN only
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # transform both
```

| When to use | Scaler |
|---|---|
| Approx uniform, few outliers | MinMaxScaler |
| Has outliers, need mean=0 std=1 | StandardScaler |
| Power-law distribution | Log scaling |
| Extreme outliers | Clipping + other |

---

## 04 — Decision Trees

**Preprocessing needed?** NO normalization needed. Robust to outliers.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

dtc = DecisionTreeClassifier()  # defaults
dtc.fit(X_train, Y_train)
y_pred = dtc.predict(X_test)
```

### Visualize tree

```python
plt.figure(figsize=(7, 7))
plot_tree(dtc, class_names=['B', 'M'], feature_names=feature_names, filled=True)
plt.show()
```

### Feature importance

```python
pd.Series(dtc.feature_importances_, index=feature_names).plot.barh()
```

### Hyperparameter tuning — GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 4, 6, 8]
}
gs = GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=5)
gs.fit(X_train, Y_train)

gs.best_params_
gs.best_score_
gs.best_estimator_.predict(X_test)
```

### Random Forest (ensemble of trees)

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)
```

---

## 05 — K-Nearest Neighbors (KNN)

**Preprocessing needed?** YES — normalization required (distance-based). Sensitive to outliers.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()  # default n_neighbors=5
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
```

### Hyperparameter tuning

```python
params = {
    'n_neighbors': range(10, 50, 5),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]   # 1=Manhattan, 2=Euclidean
}
gs = GridSearchCV(KNeighborsClassifier(), params, cv=6)
gs.fit(X_train, Y_train)
```

### Bagging ensemble

```python
from sklearn.ensemble import BaggingClassifier

bag = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=20)
bag.fit(X_train, Y_train)
```

**Key facts:**
- n_neighbors=1 → overfitting (train acc = 1)
- n_neighbors=len(train) → predicts majority class only

---

## 06 — Naive Bayes

**Preprocessing needed?** Categorical features → encode to integers with OrdinalEncoder. No normalization needed.

### CategoricalNB (for categorical features)

```python
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB

oe = OrdinalEncoder()
oe.fit(X_train)
X_train = oe.transform(X_train)
X_test = oe.transform(X_test)

model = CategoricalNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### MultinomialNB (for text / count data)

```python
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

dv = DictVectorizer()
dv.fit(X_train)   # X_train is list of dicts {word: count}
X_train_vec = pd.DataFrame(dv.transform(X_train).toarray(), columns=dv.feature_names_)
X_test_vec = pd.DataFrame(dv.transform(X_test).toarray(), columns=dv.feature_names_)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
```

---

## 07 — PCA (Principal Component Analysis)

**Preprocessing needed?** YES — StandardScaler required (mean=0 needed).

PCA is unsupervised — no target needed for PCA itself.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scale first
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# PCA
pca = PCA(n_components=2)       # int = num components, float = variance ratio
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
```

### Useful attributes

```python
pca.explained_variance_ratio_              # variance explained per component
np.cumsum(pca.explained_variance_ratio_)   # cumulative variance
pca.components_                            # coefficients (loadings)
pca.inverse_transform(X_pca)               # reconstruct original features
```

### 2D visualization

```python
scatter = plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c=y_train)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(handles=scatter.legend_elements()[0], labels=class_names.tolist())
```

**Limitation:** PCA finds linear subspaces only.

---

## 08 — SVM (Support Vector Machine)

**Preprocessing needed?** YES — StandardScaler.

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='rbf')  # or 'linear', 'poly'
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Hyperparameter tuning (multiple param grids)

```python
params = [
    {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10]},
    {'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10], 'gamma': [0.01, 0.1, 1, 10]},
]
gs = GridSearchCV(SVC(), params, scoring='accuracy', cv=5)
gs.fit(X_train, y_train)
gs.best_params_
```

**Key facts:**
- `kernel='linear'` — linear boundary
- `kernel='rbf'` — non-linear boundary (Gaussian)
- `C` — regularization (higher = harder margin)
- `gamma` — RBF kernel width
- Support vectors: `model.support_vectors_`

---

## 09 — Imbalanced Classes

**Don't use accuracy** with imbalanced data. Use precision, recall, F1, PR curve.

**Resampling happens AFTER train/test split (on train only).**

```python
# pip install imblearn
from imblearn.metrics import classification_report_imbalanced
```

### Oversampling

```python
from imblearn.over_sampling import RandomOverSampler, SMOTE

# Random (duplicates minority)
ros = RandomOverSampler(random_state=0)
X_res, y_res = ros.fit_resample(X_train, Y_train)

# SMOTE (interpolates new minority instances)
smote = SMOTE(k_neighbors=10, random_state=42)
X_res, y_res = smote.fit_resample(X_train, Y_train)
```

### Undersampling

```python
from imblearn.under_sampling import RandomUnderSampler, NearMiss, CondensedNearestNeighbour

rus = RandomUnderSampler()
X_res, y_res = rus.fit_resample(X_train, Y_train)

nm = NearMiss(version=1, n_neighbors=20)  # version 1, 2, or 3
X_res, y_res = nm.fit_resample(X_train, Y_train)
```

### Combined

```python
from imblearn.combine import SMOTEENN
smoteenn = SMOTEENN(random_state=42)
X_res, y_res = smoteenn.fit_resample(X_train, Y_train)
```

### Balanced ensemble

```python
from imblearn.ensemble import BalancedRandomForestClassifier
model = BalancedRandomForestClassifier(max_depth=6, random_state=42)
model.fit(X_train, Y_train)
```

### PR Curve (better than ROC for imbalanced)

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(Y_test, model.predict(X_test))
ap = average_precision_score(Y_test, model.predict(X_test))
plt.plot(recall, precision, label=f'Model (AP: {ap:.2f})')
```

---

## 10 — K-Means Clustering

**Preprocessing needed?** YES — normalization (MinMaxScaler). Sensitive to outliers.

**Unsupervised** — no train/test split needed.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, n_init='auto')
kmeans.fit(X)
```

### Useful attributes

```python
kmeans.labels_            # cluster assignment per instance
kmeans.cluster_centers_   # centroid coordinates
kmeans.inertia_           # SSE
```

### Choosing K — Elbow + Silhouette

```python
from sklearn.metrics import silhouette_score, silhouette_samples

inertias, sils = [], []
for k in range(2, 10):
    km = KMeans(n_clusters=k, n_init='auto')
    km.fit(X)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X, km.labels_))

# Plot elbow
plt.plot(range(2,10), inertias, marker='o')
# Plot silhouette
plt.plot(range(2,10), sils, marker='o')
```

- **Elbow method**: pick K at the "bend"
- **Silhouette**: pick K with highest score (closer to 1 = better)

### Bisecting K-Means

```python
from sklearn.cluster import BisectingKMeans
bkm = BisectingKMeans(n_clusters=3, bisecting_strategy='largest_cluster')
bkm.fit(X)
```

### Fuzzy C-Means (soft clustering)

```python
# pip install fuzzy-c-means
from fcmeans import FCM

fcm = FCM(n_clusters=3, m=3)
fcm.fit(X.to_numpy())
labels = fcm.predict(X.to_numpy())       # hard clustering
memberships = fcm.soft_predict(X.to_numpy())  # membership degrees
fcm.centers                                    # centroids
```

---

## 11 — Agglomerative Clustering & DBSCAN

### Agglomerative (Hierarchical)

**Preprocessing needed?** YES — MinMaxScaler.

```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=3, linkage='average')
# linkage: 'single', 'complete', 'average'
model.fit(X)
model.labels_
```

### Dendrogram

```python
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage

Z = scipy_linkage(X, method='average')
dendrogram(Z, labels=names, leaf_rotation=90)
plt.show()
```

### DBSCAN (density-based)

No need to specify number of clusters. Finds clusters of arbitrary shape. Label `-1` = noise.

```python
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.1, min_samples=2)
model.fit(X)
model.labels_   # -1 = noise point
```

- `eps` — neighborhood radius
- `min_samples` — min points to form a dense region

---

## 12 — Association Rules (Apriori)

```python
# pip install apriori_python
from apriori_python import apriori

# dataset = list of lists (transactions)
# e.g. [['bread','milk'], ['bread','eggs'], ...]
freq_items, rules = apriori(dataset, minSup=0.02, minConf=0.80)
pd.DataFrame(rules, columns=['Antecedent', 'Consequent', 'Confidence'])
```

---

## Model Comparison — ROC Curve

```python
from sklearn.metrics import roc_curve, roc_auc_score

for model, name in zip(models, names):
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(Y_test, y_pred)
    auc = roc_auc_score(Y_test, y_pred)
    plt.plot(fpr, tpr, label=f'{name} (AUC: {auc:.2f})')

plt.plot([0, 1], [0, 1], label='Random (AUC: 0.5)', color='red')
plt.legend()
plt.show()
```

**For imbalanced classes → use PR curve instead of ROC.**

---

## Saving / Loading Models

```python
import pickle
# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
# Load
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

```python
import joblib  # better for large numpy arrays (sklearn models)
joblib.dump(model, 'model.pkl')
model = joblib.load('model.pkl')
```

---

## Confusion Matrix Quick Reference

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actually Positive** | TP | FN (Type II error) |
| **Actually Negative** | FP (Type I error) | TN |

- **Precision** = TP / (TP + FP) — "of all predicted positive, how many are correct?"
- **Recall** = TP / (TP + FN) — "of all actual positive, how many did we catch?"
- **F1** = 2 * (Precision * Recall) / (Precision + Recall)

---

## Algorithm Requirements Summary

| Algorithm | Normalization | Handles Categorical | Sensitive to Outliers | Supervised |
|---|---|---|---|---|
| Decision Tree | No | Yes (with encoding) | No | Yes |
| Random Forest | No | Yes (with encoding) | No | Yes |
| KNN | **Yes** (MinMax) | No | **Yes** | Yes |
| Naive Bayes | No (OrdinalEncoder) | Yes | No | Yes |
| SVM | **Yes** (StandardScaler) | No | Depends on kernel | Yes |
| PCA | **Yes** (StandardScaler) | No | Yes | No |
| K-Means | **Yes** (MinMax) | No | **Yes** | No |
| Agglomerative | **Yes** (MinMax) | No | Depends on linkage | No |
| DBSCAN | **Yes** (MinMax) | No | Less sensitive | No |
