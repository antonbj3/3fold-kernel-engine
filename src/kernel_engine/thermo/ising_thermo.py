"""
ising_thermo.py  -- canonical simulable Ising / p-bit sampler with full
stochastic-thermodynamic bookkeeping.  Core library for the two probes:

  probe_A_cert.py : identifiability (sigma_min(Fisher)), FDT/T_eff over-determination,
                    Landauer erasure energy-per-decision.
  probe_B_tur.py  : thermodynamic uncertainty relation (precision vs dissipation).

Conventions (k_B = 1, so beta = 1/T, energies in units of T where noted):
  spins  s_i in {-1,+1}
  E(s)   = - sum_{i<j} J_ij s_i s_j - sum_i h_i s_i
  P(s)   = exp(-beta E(s)) / Z          (Boltzmann; the sampler's TARGET)
  Glauber heat-bath update of spin i:
     local field  H_i = sum_j J_ij s_j + h_i      (excludes self)
     set s_i = +1 with prob  sigmoid(2 beta H_i) = 1/(1+exp(-2 beta H_i))
  This IS the p-bit equation  m_i = sign( tanh(beta H_i) - r ),  r~U(-1,1).
  Heat-bath Glauber satisfies detailed balance wrt P(s) -> stationary = Boltzmann.

Everything is EXACT by enumeration for small N (2^N states) so the simulator is
anchored to an analytic ground truth, not to itself.
"""
import numpy as np

LN2 = np.log(2.0)


# ----------------------------------------------------------------------------
# enumeration + exact Boltzmann
# ----------------------------------------------------------------------------
def all_states(N):
    """(2^N, N) array of +-1 spin configurations."""
    idx = np.arange(2 ** N)
    bits = ((idx[:, None] >> np.arange(N)[None, :]) & 1).astype(np.float64)
    return 2.0 * bits - 1.0


def energy(states, J, h):
    """E(s) = -sum_{i<j} J_ij s_i s_j - sum_i h_i s_i  ; states (M,N)."""
    # pairwise term: 0.5 s J s (J symmetric, zero diagonal) equals sum_{i<j}
    quad = 0.5 * np.einsum('mi,ij,mj->m', states, J, states)
    lin = states @ h
    return -quad - lin


def boltzmann(states, J, h, beta):
    E = energy(states, J, h)
    w = np.exp(-beta * (E - E.min()))
    Z = w.sum()
    P = w / Z
    logZ = np.log(Z) - beta * E.min()
    return P, logZ, E


# ----------------------------------------------------------------------------
# sufficient statistics + Fisher information (identifiability)
# ----------------------------------------------------------------------------
def suff_stats(states):
    """phi(s) = [ s_i s_j for i<j , s_i for i ].  Returns (M, D), plus label list."""
    M, N = states.shape
    cols = []
    labels = []
    for i in range(N):
        for j in range(i + 1, N):
            cols.append(states[:, i] * states[:, j])
            labels.append(('J', i, j))
    for i in range(N):
        cols.append(states[:, i])
        labels.append(('h', i))
    return np.stack(cols, axis=1), labels


def fisher(states, phi, P, beta):
    """
    Boltzmann law is exponential family P ~ exp(sum_a theta_a phi_a), theta_a = beta*p_a
    with physical params p = (J_ij, h_i).  Fisher wrt natural theta = Cov(phi);
    Fisher wrt physical params p = beta^2 * Cov(phi).
    Returns (Fisher_phys, Cov_phi).
    """
    mean = phi.T @ P                      # (D,)
    cov = (phi * P[:, None]).T @ phi - np.outer(mean, mean)   # E[phi phi] - E[phi]E[phi]
    cov = 0.5 * (cov + cov.T)             # symmetrize
    return beta ** 2 * cov, cov


def sigma_min(M):
    return float(np.linalg.eigvalsh(M)[0])


# ----------------------------------------------------------------------------
# Glauber (heat-bath / p-bit) dynamics
# ----------------------------------------------------------------------------
def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))   # = 1/(1+exp(-x)), stable


def local_field(s, J, h):
    """H_i = sum_j J_ij s_j + h_i (J has zero diagonal so self excluded)."""
    return J @ s + h


def glauber_sweep(s, J, h, beta, rng, order=None):
    """One sweep = N single-spin heat-bath updates (random-site if order given)."""
    N = s.shape[0]
    if order is None:
        order = rng.permutation(N)
    for i in order:
        Hi = float(J[i] @ s + h[i])
        p_up = sigmoid(2.0 * beta * Hi)
        s[i] = 1.0 if rng.random() < p_up else -1.0
    return s


def glauber_chain(J, h, beta, n_sweeps, rng, s0=None, burn=0):
    N = J.shape[0]
    s = (rng.integers(0, 2, N).astype(np.float64) * 2 - 1) if s0 is None else s0.copy()
    for _ in range(burn):
        glauber_sweep(s, J, h, beta, rng)
    traj = np.empty((n_sweeps, N))
    for t in range(n_sweeps):
        glauber_sweep(s, J, h, beta, rng)
        traj[t] = s
    return traj


# ----------------------------------------------------------------------------
# detailed-balance residual (independent structural anchor)
# ----------------------------------------------------------------------------
def detailed_balance_residual(states, J, h, beta):
    """
    For every single-spin-flip pair (s, s'), check pi_s W(s->s') = pi_s' W(s'->s)
    under the heat-bath kernel (uniform site choice 1/N).  Returns max relative
    residual over all such pairs.  W(s->s') = (1/N) * p(flip that spin to new val).
    """
    M, N = states.shape
    P, _, _ = boltzmann(states, J, h, beta)
    # map state -> index
    key = ((states > 0).astype(np.int64) * (1 << np.arange(N))).sum(1)
    order = np.argsort(key)
    lookup = {int(key[k]): k for k in range(M)}
    worst = 0.0
    for a in range(M):
        s = states[a]
        for i in range(N):
            Hi = float(J[i] @ s + h[i])
            # flipping spin i: new value = -s_i
            p_up = sigmoid(2.0 * beta * Hi)
            # prob that heat-bath sets spin i to +1 or -1
            new_val = -s[i]
            p_flip = p_up if new_val > 0 else (1.0 - p_up)
            Wab = p_flip / N
            # reverse
            skey = int(key[a]) ^ (1 << i)
            b = lookup[skey]
            old_val = s[i]
            p_flip_rev = p_up if old_val > 0 else (1.0 - p_up)  # same Hi (self excluded)
            Wba = p_flip_rev / N
            lhs = P[a] * Wab
            rhs = P[b] * Wba
            denom = 0.5 * (lhs + rhs) + 1e-300
            worst = max(worst, abs(lhs - rhs) / denom)
    return worst


# ----------------------------------------------------------------------------
# small canonical instances
# ----------------------------------------------------------------------------
def ferro_ring(N, Jcoup=1.0, h0=0.0, seed=0):
    """1D periodic ferromagnetic Ising ring."""
    J = np.zeros((N, N))
    for i in range(N):
        J[i, (i + 1) % N] = Jcoup
        J[(i + 1) % N, i] = Jcoup
    h = np.full(N, h0)
    return J, h


def random_coupled(N, scale=1.0, hscale=0.3, seed=0):
    """Dense random symmetric couplings + random fields (generic, non-degenerate)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, scale, (N, N))
    J = np.triu(A, 1)
    J = J + J.T
    h = rng.normal(0, hscale, N)
    return J, h


if __name__ == "__main__":
    # smoke test: detailed balance + Boltzmann sampling on a 3-spin ring
    rng = np.random.default_rng(1)
    J, h = random_coupled(3, seed=2)
    st = all_states(3)
    for beta in [0.3, 1.0, 3.0]:
        db = detailed_balance_residual(st, J, h, beta)
        print(f"beta={beta:4.1f}  detailed-balance max-resid = {db:.2e}")
