import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest, wasserstein_distance, entropy, probplot
from scipy.stats import gaussian_kde
from guassian_pertubations import build_pure_quantum_gaussian_perturber
import os

VECTOR = [0.2]
SIGMA = [0.1]
SHOTS = 20000
MAX_QUBITS = 63
M_VALUES = []
for m in [4, 8, 16, 32, 64]:
    required_qubits = len(VECTOR) * (m + 1)
    if required_qubits <= MAX_QUBITS:
        M_VALUES.append(m)
print("Using m values:", M_VALUES)
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
def extract_samples(counts, decoder):
    samples = []
    for bitstring, freq in counts.items():
        xprime = decoder(bitstring)[0]
        samples.extend([xprime] * freq)
    return np.array(samples)
def compute_kl_divergence(samples, mu, sigma, bins=100):
    hist, edges = np.histogram(samples, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    gaussian_pdf = norm.pdf(centers, mu, sigma)
    hist += 1e-12
    gaussian_pdf += 1e-12
    return entropy(hist, gaussian_pdf)
results = []
for m in M_VALUES:
    print(f"Running m = {m}")
    qc, counts, decoder = build_pure_quantum_gaussian_perturber(
        V=VECTOR,
        sigma=SIGMA,
        m=m,
        shots=SHOTS
    )
    samples = extract_samples(counts, decoder)
    empirical_mean = np.mean(samples)
    empirical_variance = np.var(samples)
    target_mean = VECTOR[0]
    target_sigma = SIGMA[0]
    standardized = (samples - target_mean) / target_sigma
    ks_stat, ks_pvalue = kstest(standardized, 'norm')
    ideal_gaussian = np.random.normal(
        target_mean,
        target_sigma,
        size=len(samples)
    )
    wasserstein = wasserstein_distance(samples, ideal_gaussian)
    kl_div = compute_kl_divergence(
        samples,
        target_mean,
        target_sigma
    )
    results.append({
        "m": m,
        "mean": empirical_mean,
        "variance": empirical_variance,
        "ks_stat": ks_stat,
        "wasserstein": wasserstein,
        "kl_divergence": kl_div
    })
    plt.figure(figsize=(6, 4))
    plt.hist(
        samples,
        bins=min(m + 1, 40),
        density=True,
        alpha=0.7,
        label="Generated Samples"
    )
    x = np.linspace(
        min(samples),
        max(samples),
        1000
    )
    plt.plot(
        x,
        norm.pdf(x, target_mean, target_sigma),
        linewidth=2,
        label="Target Gaussian"
    )
    plt.title(f"Histogram vs Gaussian (m={m})")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/histogram_m_{m}.png")
    plt.close()
    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    probplot(
        standardized,
        dist="norm",
        plot=ax
    )
    for line in ax.get_lines():
        line.set_color('gray')
    plt.title(f"QQ Plot (m={m})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/qqplot_m_{m}.png")
    plt.close()

df = pd.DataFrame(results)
df.to_csv(f"{OUTPUT_DIR}/validation_metrics.csv", index=False)
print(df)
plt.figure(figsize=(6, 4))
plt.plot(df["m"], df["ks_stat"], marker='o')
plt.xlabel("Ancilla Count (m)")
plt.ylabel("KS Statistic")
plt.title("Gaussian Convergence vs Ancilla Count")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ks_convergence.png")
plt.close()
plt.figure(figsize=(6, 4))
plt.plot(df["m"], df["variance"], marker='o')
plt.axhline(SIGMA[0]**2, linestyle='--')
plt.xlabel("Ancilla Count (m)")
plt.ylabel("Empirical Variance")
plt.title("Variance Convergence")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/variance_convergence.png")
plt.close()
print("All experiments completed.")