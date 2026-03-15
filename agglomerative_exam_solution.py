from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


# pomoc za merenje vremena izvrsavanja
start_time = perf_counter()
for _ in range(10000):
    pass
print(f"elapsed time: {perf_counter() - start_time:.4f}")


# Dataset generation (prema poslatom kodu, uz ispravljene sitne greske)
n_samples = 1500

noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=170
)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=170)

blobs = datasets.make_blobs(
    n_samples=n_samples,
    centers=3,
    random_state=170,
)

rng = np.random.RandomState(170)
no_structure = (rng.rand(n_samples, 2), None)

X_aniso_base, y_aniso = datasets.make_blobs(
    n_samples=n_samples,
    centers=3,
    random_state=170,
)
transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
X_aniso = np.dot(X_aniso_base, transformation)
aniso = (X_aniso, y_aniso)

varied = datasets.make_blobs(
    n_samples=n_samples,
    centers=3,
    cluster_std=[1.0, 2.5, 0.5],
    random_state=170,
)

Xs = tuple(X for X, _ in (noisy_circles, noisy_moons, varied, aniso, blobs, no_structure))


# Prva 2 skupa -> 2 klastera, ostala 4 -> 3 klastera
dataset_names = [
    "noisy_circles",
    "noisy_moons",
    "varied",
    "aniso",
    "blobs",
    "no_structure",
]
cluster_counts = [2, 2, 3, 3, 3, 3]
linkages = ["single", "average", "complete"]


output_dir = Path("exam_output")
output_dir.mkdir(exist_ok=True)

records = []
stored = {}


# Model fitting za sve kombinacije i merenje vremena
for i, X in enumerate(Xs):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    for linkage_name in linkages:
        start_time = perf_counter()
        model = AgglomerativeClustering(
            n_clusters=cluster_counts[i],
            linkage=linkage_name,
        )
        labels = model.fit_predict(X_scaled)
        elapsed_ms = (perf_counter() - start_time) * 1000

        if len(np.unique(labels)) > 1:
            sil = silhouette_score(X_scaled, labels)
        else:
            sil = np.nan

        records.append(
            {
                "dataset": dataset_names[i],
                "n_clusters": cluster_counts[i],
                "linkage": linkage_name,
                "silhouette": sil,
                "time_ms": elapsed_ms,
            }
        )
        stored[(dataset_names[i], linkage_name)] = (X_scaled, labels)


results_df = pd.DataFrame(records)
results_df = results_df.sort_values(["dataset", "linkage"])
results_df.to_csv(output_dir / "agglomerative_all_results.csv", index=False)


# Mreza scatter plotova: redovi=dataset, kolone=linkage
fig, axes = plt.subplots(len(dataset_names), len(linkages), figsize=(16, 24), squeeze=False)

for row_idx, ds_name in enumerate(dataset_names):
    for col_idx, linkage_name in enumerate(linkages):
        ax = axes[row_idx, col_idx]

        row = results_df[
            (results_df["dataset"] == ds_name) & (results_df["linkage"] == linkage_name)
        ].iloc[0]
        X_scaled, labels = stored[(ds_name, linkage_name)]

        ax.scatter(
            X_scaled[:, 0],
            X_scaled[:, 1],
            c=labels,
            cmap="tab10",
            s=12,
            alpha=0.9,
            edgecolors="none",
        )

        sil_txt = "N/A" if pd.isna(row["silhouette"]) else f"{row['silhouette']:.3f}"
        ax.set_title(
            f"{linkage_name} | sil={sil_txt} | t={row['time_ms']:.2f} ms",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        if col_idx == 0:
            ax.set_ylabel(f"{ds_name}\n(k={int(row['n_clusters'])})", fontsize=10)

fig.suptitle("Hijerarhijsko klasterovanje: 6 skupova x 3 metode", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(output_dir / "agglomerative_scatter_grid.png", dpi=220)
plt.close(fig)


# Poredjenje metoda
avg_by_linkage = (
    results_df.groupby("linkage", as_index=False)
    .agg(avg_silhouette=("silhouette", "mean"), avg_time_ms=("time_ms", "mean"))
    .sort_values("avg_silhouette", ascending=False)
)

best_per_dataset = (
    results_df.sort_values(["dataset", "silhouette"], ascending=[True, False])
    .groupby("dataset", as_index=False)
    .first()
)


# Izbor jednog dobrog modela i skupa (globalno najbolja silueta)
best_idx = results_df["silhouette"].idxmax()
best_row = results_df.loc[best_idx]
best_dataset = best_row["dataset"]
best_linkage = best_row["linkage"]
X_best, _ = stored[(best_dataset, best_linkage)]


# Dendrogram za izabrani model
Z = linkage(X_best, method=best_linkage)
fig = plt.figure(figsize=(12, 5))
dendrogram(Z, no_labels=True)
plt.title(
    f"Dendrogram | dataset={best_dataset}, linkage={best_linkage}, sil={best_row['silhouette']:.3f}"
)
plt.xlabel("Instance")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(output_dir / "best_model_dendrogram.png", dpi=220)
plt.close(fig)


# Tekstualni odgovor na teorijski deo
report_lines = [
    "POREDJENJE METODA SPAJANJA (single, average, complete)",
    "=" * 72,
    "",
    "1) Prosecni rezultati po metodi:",
    avg_by_linkage.to_string(index=False),
    "",
    "2) Najbolja metoda po svakom skupu (po silueti):",
    best_per_dataset[["dataset", "linkage", "silhouette", "time_ms"]].to_string(index=False),
    "",
    "3) Da li su rezultati i vreme ocekivani?",
    "- Da. Single je cesto brzi, ali moze dati losiji kvalitet zbog chaining efekta.",
    "- Average i complete cesto daju stabilnije i kompaktnije klastere.",
    "",
    "4) Da li je silueta adekvatna metrika za sve skupove?",
    "- Ne uvek. Za nelinearne oblike (circles, moons) i no_structure moze biti varljiva.",
    "- Razlog: favorizuje dobro razdvojene, konveksne klastere.",
    "",
    "5) Izabran model za dendrogram:",
    f"- dataset={best_dataset}, linkage={best_linkage}, sil={best_row['silhouette']:.4f}, time_ms={best_row['time_ms']:.2f}",
]

with open(output_dir / "agglomerative_comparison_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))


print("\n=== REZULTATI (sve kombinacije) ===")
print(results_df.to_string(index=False))
print("\n=== Prosek po metodi ===")
print(avg_by_linkage.to_string(index=False))
print("\n=== Najbolji model za dendrogram ===")
print(best_row.to_string())
print("\nSacuvano u folderu exam_output:")
print("- agglomerative_all_results.csv")
print("- agglomerative_scatter_grid.png")
print("- best_model_dendrogram.png")
print("- agglomerative_comparison_report.txt")
