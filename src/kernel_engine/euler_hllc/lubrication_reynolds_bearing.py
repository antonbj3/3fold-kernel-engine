#!/usr/bin/env python3
"""Hydrodynamic lubrication (Reynolds slider bearing): lubricated contact validated against the analytic
solution and a known optimum.

A tilted slider builds hydrodynamic pressure in the lubricant film that carries the load without solid
contact. 1D Reynolds equation: d/dx(h^3 dp/dx) = 6 mu U dh/dx with p = 0 at both edges. Differentiable.

Gates: (0) the finite-difference Reynolds solver reproduces the analytic pressure profile p(x) for a
linear wedge (mesh-converging); (1) load capacity W = integral p dx matches the analytic value;
(2) dW/d{h1, h2, mu, U} from autograd through the solver matches central finite differences;
(3) gradient optimisation of the wedge ratio K = h1/h2 for maximum load capacity recovers the known
optimum K* ~= 2.19 for a fixed-incline slider.

  python3 lubrication_reynolds_bearing.py
"""
import sys

import numpy as np

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

MU = 0.01            # dynamisk viskositet [Pa·s]
U = 1.0              # glidhastighet [m/s]
B = 0.05            # bearing length [m]
H1, H2 = 2e-4, 1e-4  # film-tjocklek inlopp/utlopp (konvergerande kil) [m]
N = 2001
K_OPT_KNOWN = 2.19   # known optimal h1/h2 for maximum load capacity (fixed-incline slider)


def analytic_p(x, h1, h2, mu, U):
    """Closed-form pressure for a linear wedge h(x)=h1+m*x: double integration of Reynolds, p(0)=p(B)=0."""
    m = (h2 - h1) / B; h = h1 + m * x
    C = -12 * mu * U * h1 * h2 / (h1 + h2)
    return (-6 * mu * U / m) * (1 / h - 1 / h1) + (-C / (2 * m)) * (1 / h ** 2 - 1 / h1 ** 2)


def fd_solve_np(x, h, mu, U):
    """Finite-difference solution of d/dx(h^3 p')=6 mu U h', p=0 at the edges (conservative form, h^3 at midpoints)."""
    n = len(x); dx = x[1] - x[0]
    hm = 0.5 * (h[:-1] + h[1:]); h3 = hm ** 3
    A = np.zeros((n, n)); b = np.zeros(n)
    for i in range(1, n - 1):
        A[i, i - 1] = h3[i - 1] / dx ** 2; A[i, i + 1] = h3[i] / dx ** 2
        A[i, i] = -(h3[i - 1] + h3[i]) / dx ** 2
        b[i] = 6 * mu * U * (h[i + 1] - h[i - 1]) / (2 * dx)
    A[0, 0] = A[-1, -1] = 1.0
    return np.linalg.solve(A, b)


def main():
    print(f"Hydrodynamic lubrication (Reynolds slider bearing): mu={MU} Pa s, U={U}m/s, B={B}m, h1/h2={H1/H2:.1f}, N={N}")
    x = np.linspace(0, B, N)
    h = H1 + (H2 - H1) / B * x

    # (0)/(1) FD vs analytic (pressure profile and load capacity), mesh convergence
    p_fd = fd_solve_np(x, h, MU, U); p_an = analytic_p(x, H1, H2, MU, U)
    p_err = np.max(np.abs(p_fd - p_an)) / np.max(np.abs(p_an))
    W_fd = trapz(p_fd, x); W_an = trapz(p_an, x); W_err = abs(W_fd - W_an) / abs(W_an)
    # mesh convergence: coarse vs fine
    xc = np.linspace(0, B, 501); hc = H1 + (H2 - H1) / B * xc
    p_err_coarse = np.max(np.abs(fd_solve_np(xc, hc, MU, U) - analytic_p(xc, H1, H2, MU, U))) / np.max(np.abs(analytic_p(xc, H1, H2, MU, U)))
    print(f"  (0) FD vs analytic p(x): rel err {p_err:.2e} (coarse mesh {p_err_coarse:.2e} -> converging); p_max {p_fd.max():.3e} Pa")
    print(f"  (1) load capacity W=int p dx: FD {W_fd:.4e} vs analytic {W_an:.4e} (rel err {W_err:.2e}) N/m")
    forward_ok = p_err < 1e-3 and W_err < 1e-3 and p_err_coarse > p_err

    # (2) differentiable: dW/d{h1,h2,mu,U} via autograd through the solver vs central FD
    import torch
    torch.set_default_dtype(torch.float64)
    xt = torch.tensor(x); dx = float(x[1] - x[0])
    def W_torch(h1, h2, mu, U_):
        h = h1 + (h2 - h1) / B * xt
        hm = 0.5 * (h[:-1] + h[1:]); h3 = hm ** 3
        n = N
        main_d = torch.zeros(n); lo = torch.zeros(n - 1); up = torch.zeros(n - 1)
        rhs = torch.zeros(n)
        main_d[1:-1] = -(h3[:-1] + h3[1:]) / dx ** 2
        up[1:] = h3[1:] / dx ** 2          # A[i,i+1] for i=1..n-2
        lo[:-1] = h3[:-1] / dx ** 2        # A[i,i-1] for i=1..n-2
        main_d[0] = 1.0; main_d[-1] = 1.0; up[0] = 0.0; lo[-1] = 0.0
        rhs[1:-1] = 6 * mu * U_ * (h[2:] - h[:-2]) / (2 * dx)
        A = torch.diag(main_d) + torch.diag(up, 1) + torch.diag(lo, -1)
        p = torch.linalg.solve(A, rhs)
        return torch.trapezoid(p, xt)
    h1v = torch.tensor(H1, requires_grad=True); h2v = torch.tensor(H2, requires_grad=True)
    muv = torch.tensor(MU, requires_grad=True); Uv = torch.tensor(U, requires_grad=True)
    W_torch(h1v, h2v, muv, Uv).backward()
    def Wn(a, b_, c, d): return float(W_torch(torch.tensor(a), torch.tensor(b_), torch.tensor(c), torch.tensor(d)))
    grads_an = {}
    for nm, (val, gv) in {"h1": (H1, h1v.grad), "h2": (H2, h2v.grad), "mu": (MU, muv.grad), "U": (U, Uv.grad)}.items():
        hh = val * 1e-6; base = [H1, H2, MU, U]; idx = ["h1", "h2", "mu", "U"].index(nm)
        bp = base.copy(); bp[idx] += hh; bm = base.copy(); bm[idx] -= hh
        fd = (Wn(*bp) - Wn(*bm)) / (2 * hh)
        grads_an[nm] = abs(float(gv) - fd) / (abs(fd) + 1e-30)
    grad_err = max(grads_an.values())
    print(f"  (2) differentiable dW/d{{h1,h2,mu,U}} autograd vs FD: max rel err {grad_err:.2e} ({grads_an})")
    diff_ok = grad_err < 1e-5

    # (3) gradient optimisation of the wedge ratio K=h1/h2 for max load capacity = known optimum ~2.19
    logK = torch.tensor(np.log(1.5), requires_grad=True)   # start from K=1.5
    opt = torch.optim.Adam([logK], lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        K = torch.exp(logK)
        Wk = W_torch(K * torch.tensor(H2), torch.tensor(H2), torch.tensor(MU), torch.tensor(U))
        (-Wk).backward(); opt.step()
    K_star = float(torch.exp(logK).detach())
    K_err = abs(K_star - K_OPT_KNOWN) / K_OPT_KNOWN
    print(f"  (3) gradient-optimised wedge ratio K*={K_star:.3f} (known optimum {K_OPT_KNOWN}) rel err {K_err:.2e}")
    design_ok = K_err < 0.02

    ok = forward_ok and diff_ok and design_ok
    print(f"\nVERDICT: hydrodynamic lubrication (Reynolds slider bearing) = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"(0) the FD Reynolds solver reproduces the analytic pressure profile p(x) for a linear wedge "
             f"({p_err:.0e}, mesh-converging) and (1) the load capacity W = int p dx matches the analytic value "
             f"({W_err:.0e}). (2) differentiable: dW/d{{h1,h2,mu,U}} via autograd through the solver matches central FD "
             f"({grad_err:.0e}). (3) gradient optimisation of the wedge ratio recovers the known optimum "
             f"K*={K_star:.2f} ~= {K_OPT_KNOWN} (textbook fixed-incline slider maximum load capacity). " if ok else
             f"Not validated (forward {forward_ok}: p {p_err:.1e}/W {W_err:.1e}, diff {diff_ok}, design {design_ok}: "
             f"K*={K_star:.3f}). ")
          + "SCOPE: 1D steady Reynolds (incompressible, iso-viscous, laminar, no cavitation, thermal or elasto-"
          "hydrodynamic effects); linear wedge geometry; rigid surfaces; the known optimum K ~= 2.19 is the maximum of "
          "the dimensionless load capacity for a fixed incline. The analytic comparison, mesh convergence, autograd vs "
          "FD and the known design optimum are the falsifiable parts.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
