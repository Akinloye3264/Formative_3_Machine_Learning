import numpy as np
import matplotlib.pyplot as plt
DATA_PATH = "GaltonFamilies.csv"

# Read the header row separately so we can find column positions by name
header = np.genfromtxt(DATA_PATH, delimiter= ',', dtype=str, max_rows = 1)
col = {name: i for i, name in enumerate(header)}

# Read the rest of the file as raw strings and pconvert only the numeric columns we actually need
rows = np. genfromtxt(DATA_PATH, delimiter= ',', dtype= str, skip_header = 1)

family = rows[:, col["family"]]
father =  rows [:, col["father"]].astype(float)
child_heights = rows[:, col["childHeight"]].astype(float)

"""One father height per family (avoid re-using the same father's height
once per child -- that would bias the "adult" cluster). np.unique with
return_index gives the position of each family's FIRST appearance.
"""

_, first_idx = np.unique(family, return_index = True)
father_heights = father[first_idx]

X = np.concatenate([child_heights, father_heights])

""" Ground-truth origin of each point (0 = child, 1 = father/adult). EM never
sees this, it is kept ONLY so we can later score how well an unsupervised
method recovers the true groups (needed for the "why not just split at the
mean" comparison).
"""
labels_true = np.concatenate([np.zeros(len(child_heights)),
                               np.ones(len(father_heights))])

rng = np.random.default_rng(42)
perm = rng.permutation(len(X))
X = X[perm]
labels_true = labels_true[perm]
n = len(X)

print(f"Pooled, unlabeled data points: {n}"
      f"(children: {len(child_heights)}, fathers: {len(father_heights)})")
print(f"Global mean: {X.mean():.3f}, Global std: {X.std():3f}\n")


# EM(Expectation Maximization)

def gaussian_pdf(x, mu, sigma2):
    sigma2 = max(sigma2, 1e-6)
    return (1.0 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((x - mu) ** 2) / (2 * sigma2))

def log_likelihood(X, mu1, mu2, s1, s2, pi1, pi2):
    mix = pi1 * gaussian_pdf(X, mu1, s1) + pi2 * gaussian_pdf(X, mu2, s2)
    return np.sum(np.log(np.clip(mix, 1e-300, None)))

def e_step(X, mu1, mu2, s1, s2, pi1, pi2):
    """Compute responsibilities: gamma_i = P(component | x_i), via Bayes' rule."""
    p1 = pi1 * gaussian_pdf(X, mu1, s1)
    p2 = pi2 * gaussian_pdf(X, mu2, s2)
    total = np.clip(p1 + p2, 1e-300, None)
    return p1 / total, p2 / total
 
def m_step(X, gamma1, gamma2):
    N1, N2 = gamma1.sum(), gamma2.sum()
    mu1 = np.sum(gamma1 * X) / N1
    mu2 = np.sum(gamma2 * X) / N2
    s1 = np.sum(gamma1 * (X - mu1) ** 2) / N1
    s2 = np.sum(gamma2 * (X - mu2) ** 2) / N2
    pi1, pi2 = N1 / len(X), N2 / len(X)
    return mu1, mu2, s1, s2, pi1, pi2

#  Initialization (reproducible, seed=42)

mu1, mu2 = np.percentile(X, 25), np.percentile(X, 75)
s1 = s2 = np.var(X)
pi1 = pi2 = 0.5
 
history = [(0, mu1, mu2, s1, s2, pi1, pi2,
            log_likelihood(X, mu1, mu2, s1, s2, pi1, pi2))]
 
MAX_ITERS, TOL = 1000, 1e-6
for it in range(1, MAX_ITERS + 1):
    gamma1, gamma2 = e_step(X, mu1, mu2, s1, s2, pi1, pi2)
    mu1, mu2, s1, s2, pi1, pi2 = m_step(X, gamma1, gamma2)
    ll_new = log_likelihood(X, mu1, mu2, s1, s2, pi1, pi2)
    history.append((it, mu1, mu2, s1, s2, pi1, pi2, ll_new))
    if abs(ll_new - history[-2][-1]) < TOL:
        break

# TRACKING TABLE -- Initialization + first 2 iterations

print(f"{'Iter':<10}{'mu1(Child)':<12}{'mu2(Pro)':<12}{'sigma1^2':<12}{'sigma2^2':<12}{'pi1':<9}{'pi2':<9}{'LogLik':<12}")
for row in history[:3]:
    it, m1, m2, v1, v2, p1, p2, l = row
    label = "0 (Init)" if it == 0 else str(it)
    print(f"{label:<10}{m1:<12.4f}{m2:<12.4f}{v1:<12.4f}{v2:<12.4f}{p1:<9.4f}{p2:<9.4f}{l:<12.2f}")
 
final = history[-1]
_, mu1f, mu2f, s1f, s2f, pi1f, pi2f, llf = final
print(f"\nConverged at iteration {final[0]}:")
print(f"  Children -> mu={mu1f:.3f}, sigma={np.sqrt(s1f):.3f}, pi={pi1f:.3f}")
print(f"  Pros     -> mu={mu2f:.3f}, sigma={np.sqrt(s2f):.3f}, pi={pi2f:.3f}")

"""NAIVE BASELINE vs EM "just split at the global mean?"
 We use the true child/father origin (never shown to EM) purely to score
 accuracy here, this is the evidence for the "should you just split at
 the mean".
 """

global_mean = X.mean()
naive_pile_child = X[X < global_mean]
naive_pile_adult = X[X >= global_mean]
naive_mu_child = naive_pile_child.mean()
naive_mu_adult = naive_pile_adult.mean()

# Whichever EM/naive cluster ends up with the lower mean is the "child" cluster.
child_is_component1 = mu1f < mu2f
gamma1_final, gamma2_final = e_step(X, mu1f, mu2f, s1f, s2f, pi1f, pi2f)
em_pred_child = (gamma1_final > gamma2_final) if child_is_component1 else (gamma2_final > gamma1_final)
naive_pred_child = X < global_mean

em_accuracy = np.mean(em_pred_child == (labels_true == 0))
naive_accuracy = np.mean(naive_pred_child == (labels_true == 0))

print(f"\nNaive split at global mean ({global_mean:.3f}):")
print(f"  'Child' pile  mean={naive_mu_child:.3f}, n={len(naive_pile_child)}")
print(f"  'Adult' pile  mean={naive_mu_adult:.3f}, n={len(naive_pile_adult)}")
print(f"  Classification accuracy vs true origin: {naive_accuracy:.3%}")
print(f"EM model classification accuracy vs true origin: {em_accuracy:.3%}")

def classify_height(h, mu1, mu2, s1, s2, pi1, pi2):
    p_child = pi1 * gaussian_pdf(h, mu1, s1)
    p_pro = pi2 * gaussian_pdf(h, mu2, s2)
    total = p_child + p_pro
    return p_child / total, p_pro / total

def plot_em_results(X, history, mu1f, mu2f, s1f, s2f, pi1f, pi2f):
    fig, (ax_fit, ax_ll) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: pooled histogram plus initial vs converged Gaussian fits.
    _, edges, _ = ax_fit.hist(X, bins=32, color="0.6", alpha=0.35,
                               label="Pooled histogram (unlabeled)")
    bin_width = edges[1] - edges[0]
    xs = np.linspace(edges[0] - 1, edges[-1] + 1, 300)

    init = history[0]
    _, i_mu1, i_mu2, i_s1, i_s2, i_pi1, i_pi2, _ = init

    def curve(mu, s2, pi):
        return pi * gaussian_pdf(xs, mu, s2) * len(X) * bin_width

    ax_fit.plot(xs, curve(i_mu1, i_s1, i_pi1), "C0--", lw=1.5, alpha=0.6,
                label="Init guess (Children)")
    ax_fit.plot(xs, curve(i_mu2, i_s2, i_pi2), "C1--", lw=1.5, alpha=0.6,
                label="Init guess (Pro/Adult)")
    ax_fit.plot(xs, curve(mu1f, s1f, pi1f), "C0-", lw=2.2, label="Converged (Children)")
    ax_fit.plot(xs, curve(mu2f, s2f, pi2f), "C1-", lw=2.2, label="Converged (Pro/Adult)")

    ax_fit.set_xlabel("Height (inches)")
    ax_fit.set_ylabel("Count")
    ax_fit.set_title("Mixture of two Gaussians: initial guess vs converged fit")
    ax_fit.legend(fontsize=8)

    """ 
    Right: log-likelihood convergence (log-scaled x so the early jump
    and the long flat tail are both visible)
    """
    its = [row[0] for row in history]
    lls = [row[7] for row in history]
    ax_ll.plot(its, lls, "C0-", lw=2)
    ax_ll.scatter([its[0], its[min(1, len(its)-1)], its[min(2, len(its)-1)], its[-1]],
                  [lls[0], lls[min(1, len(lls)-1)], lls[min(2, len(lls)-1)], lls[-1]],
                  color="C0", zorder=5)
    ax_ll.set_xscale("symlog")
    ax_ll.set_xlabel("Iteration (log scale)")
    ax_ll.set_ylabel("Log-likelihood")
    ax_ll.set_title(f"Convergence (stopped at iteration {its[-1]})")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_em_results(X, history, mu1f, mu2f, s1f, s2f, pi1f, pi2f)

    test_height = float(input("\nEnter a test height (inches): "))
    pc, pp = classify_height(test_height, mu1f, mu2f, s1f, s2f, pi1f, pi2f)
    print(f"P(Child)     = {pc:.4f}")
    print(f"P(Pro/Adult) = {pp:.4f}")
    print("Classified as:", "CHILD" if pc > pp else "PRO/ADULT")