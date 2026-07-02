import numpy as np
DATA_PATH = "GaltonFamilies.csv"

# Read the header row separately so we can find column positions by name
header = np.genfromtxt(DATA_PATH, delimiter= ',', dtype=str, max_rows = 1)
col = {name: i for i, name in enumerate(header)}

# Read the rest of the file as raw strings (mixed dtypes -> keep as str,
# convert only the numeric columns we actually need)
rows = np. genfromtxt(DATA_PATH, delimiter= ',', dtype= str, skip_header = 1)

family = rows[:, col["family"]]
father =  rows [:, col["father"]].astype(float)
child_heights = rows[:, col["childHeight"]].astype(float)

# One father height per family (avoid re-using the same father's height
# once per child -- that would bias the "adult" cluster). np.unique with
# return_index gives the position of each family's FIRST appearance.

_, first_idx = np.unique(family, return_index = True)
father_heights = father[first_idx]

X = np.concatenate([child_heights, father_heights])
rng = np.random.default_rng(42)
rng.shuffle(X)
n = len(X)

print(f"Pooled, unlabeled data points: {n}" 
      f"(children: {len(child_heights)}, fathers: {len(father_heights)})")
print(f"Global mean: {X.mean():.3f}, Global std: {X.std():3f}\n")


# EM(Expectation Maximization)

def gaussian_pdf(x,mu, sigma2):
    sigma2 = max(sigma2, 1e-6)
    return(1.0 /np.sqrt(2 * np.pi * sigma2)) * np.exp(-(x - mu) ** 2) / (2 * sigma2)

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
 
MAX_ITERS, TOL = 100, 1e-6
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

def classify_height(h, mu1, mu2, s1, s2, pi1, pi2):
    p_child = pi1 * gaussian_pdf(h, mu1, s1)
    p_pro = pi2 * gaussian_pdf(h, mu2, s2)
    total = p_child + p_pro
    return p_child / total, p_pro / total
 
if __name__ == "__main__":
    test_height = float(input("\nEnter a test height (inches): "))
    pc, pp = classify_height(test_height, mu1f, mu2f, s1f, s2f, pi1f, pi2f)
    print(f"P(Child)     = {pc:.4f}")
    print(f"P(Pro/Adult) = {pp:.4f}")
    print("Classified as:", "CHILD" if pc > pp else "PRO/ADULT")