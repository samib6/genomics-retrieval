import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------

params = np.array([35, 150, 650, 3000])  # in millions

arch_esm_map5 = [0.2730, 0.3053, 0.3139, 0.3130]
euk_esm_map5  = [0.3091, 0.3514, 0.3591, 0.3570]  # DGEB reported ESM2-3B Euk

arch_blast = 0.2987
euk_blast  = 0.3403

arch_best_kmer = 0.1826   # k-mer k=4
euk_best_kmer  = 0.2484   # k-mer k=4

# -----------------------------
# Plot function
# -----------------------------

def plot_scaling(task_name, esm_scores, blast_score, kmer_score):
    plt.figure(figsize=(8, 5))

    plt.plot(
        params,
        esm_scores,
        marker="o",
        linewidth=2,
        label="ESM2"
    )

    plt.axhline(
        blast_score,
        linestyle="--",
        linewidth=2,
        label="BLASTP"
    )

    plt.axhline(
        kmer_score,
        linestyle=":",
        linewidth=2,
        label="Best k-mer (k=4)"
    )

    plt.xscale("log")

    plt.xticks(
        params,
        ["35M", "150M", "650M", "3B"]
    )

    plt.xlabel("Number of Parameters")
    plt.ylabel("MAP@5")
    plt.title(f"ESM2 Scaling on {task_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Generate plots
# -----------------------------

plot_scaling(
    "Arch Retrieval",
    arch_esm_map5,
    arch_blast,
    arch_best_kmer
)

'''plot_scaling(
    "Euk Retrieval",
    euk_esm_map5,
    euk_blast,
    euk_best_kmer
)'''