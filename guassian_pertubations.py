import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
import numpy as np
from collections import Counter


def build_pure_quantum_gaussian_perturber(V, sigma, m=16, a=None, b=None, shots=1024):
    d = len(V)
    if isinstance(sigma, (int, float)):
        sigma = [float(sigma)] * d
    if a is None:
        a = [1.0] * d
    if b is None:
        b = [0.0] * d

    q_feat = QuantumRegister(d, name="f")
    q_anc = QuantumRegister(d * m, name="a")
    c_all = ClassicalRegister(d + d * m, name="c")
    qc = QuantumCircuit(q_feat, q_anc, c_all, name="qGauss")

    thetas = [a[i] * V[i] + b[i] for i in range(d)]
    for i, theta in enumerate(thetas):
        qc.ry(theta, q_feat[i])

    for i in range(d):
        for k in range(m):
            qc.h(q_anc[i * m + k])

    alpha = [2.0 * a[i] * sigma[i] / np.sqrt(m) for i in range(d)]
    for i in range(d):
        for k in range(m):
            qc.cry(alpha[i], q_anc[i * m + k], q_feat[i])
        qc.ry(-(m / 2.0) * alpha[i], q_feat[i])

    qc.measure(q_feat[:], c_all[:d])
    qc.measure(q_anc[:], c_all[d:])

    backend = AerSimulator(method="matrix_product_state")
    tqc = transpile(qc, backend)
    job = backend.run(tqc, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)

    def bitstr_to_Vprime(
        bitstr, thetas=thetas, alpha=alpha, a=a, b=b, d=d, m=m
    ):
        bits = bitstr[::-1]
        anc_bits = bits[d : d + d * m]

        Vprime = []
        for i in range(d):
            block = anc_bits[i * m : (i + 1) * m]
            W = sum(1 for ch in block if ch == "1")
            theta_i = thetas[i] + alpha[i] * W - (m / 2.0) * alpha[i]
            xprime_i = (theta_i - b[i]) / a[i]
            Vprime.append(xprime_i)
        return Vprime

    return qc, counts, bitstr_to_Vprime


# --------------------------
# Example: try V in 2D or 3D
# --------------------------
V = [0.2]   # can be 1D, 2D, or 3D
sigma = 0.1
qc, counts, to_vec = build_pure_quantum_gaussian_perturber(V, sigma, m=20, shots=20000)

samples = []
for bitstr, freq in counts.items():
    for _ in range(freq):
        samples.append(to_vec(bitstr))
samples = np.array(samples)

# --------------------------
# Radial distribution (just for reference)
# --------------------------
dists = np.linalg.norm(samples - np.array(V), axis=1)
step = 0.1
radial_bins = (dists // step).astype(int)
counts = Counter(radial_bins)

print("\nCounts per radial shell (step = 0.1):")
for k in sorted(counts.keys()):
    r_min = k * step
    r_max = (k + 1) * step
    print(f"Radius {r_min:.1f}–{r_max:.1f}: {counts[k]} samples")

# --------------------------
# Scatter plot in 2D
# --------------------------
d = samples.shape[1]
plt.style.use('grayscale')
if d == 1:
    plt.hist(samples[:, 0], bins=40, density=True, alpha=0.7)

    plt.axvline(V[0], linestyle="--", label="Base V")
    plt.xlabel("Feature 1")
    plt.ylabel("Density")
    plt.title("1D Quantum Gaussian Perturbations")
    plt.legend()
    plt.show()

elif d == 2:
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, label="Samples")
    plt.scatter([V[0]], [V[1]], marker="x", s=100, label="Base V")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("2D Quantum Gaussian Perturbations")
    plt.legend()
    plt.show()

elif d == 3:
    from mpl_toolkits.mplot3d import Axes3D  # optional with modern Matplotlib
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
               alpha=0.30, s=10, label="Samples")
    ax.scatter([V[0]], [V[1]], [V[2]],
                marker="x", s=100, label="Base V")

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    ax.set_title("3D Quantum Gaussian Perturbations")
    ax.legend()
    # optional: make aspect ratio look nicer
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    plt.tight_layout()
    plt.show()

else:
    print(f"⚠️ Plotting only supported for 1D, 2D, or 3D. Got {d}D.")
