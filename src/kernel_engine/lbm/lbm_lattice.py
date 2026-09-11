"""Canonical lattice-Boltzmann primitives shared by the solvers: stencils, streaming, equilibrium.

  D2Q5: scalar transport (heat, species, moisture) - TCX/TCY/TW, stream5, geq5
  D2Q9: flow (Navier-Stokes)                       - CX/CY/W,   stream9, feq9

Stateless NumPy primitives; import instead of re-declaring the constants.
"""
import numpy as np

# ── D2Q5 (diffusing scalar) ──
TCX = np.array([0, 1, 0, -1, 0]); TCY = np.array([0, 0, 1, 0, -1]); TW = np.array([1/3, 1/6, 1/6, 1/6, 1/6])
# ── D2Q9 (flow) ──
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1]); CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])


def _stream(pop, cx, cy, Q, out):
    """advect Q populations by integer lattice velocities (cx,cy) — single combined roll, skipping zero shifts, written
    into a preallocated buffer. ~7× faster than stack([roll(roll(...))]) and BIT-IDENTICAL (np.roll is exact). Pass `out`
    (a buffer of pop's shape) in hot loops to avoid the per-call allocation."""
    if out is None:
        out = np.empty_like(pop)
    for q in range(Q):
        dx, dy = int(cx[q]), int(cy[q])
        if dx == 0 and dy == 0:
            out[q] = pop[q]
        elif dy == 0:
            out[q] = np.roll(pop[q], dx, 0)
        elif dx == 0:
            out[q] = np.roll(pop[q], dy, 1)
        else:
            out[q] = np.roll(pop[q], (dx, dy), (0, 1))
    return out


def stream5(g, out=None):
    """advect the 5 D2Q5 populations by their lattice velocities (exact integer rolls)."""
    return _stream(g, TCX, TCY, 5, out)


def stream9(f, out=None):
    """advect the 9 D2Q9 populations by their lattice velocities."""
    return _stream(f, CX, CY, 9, out)


def geq5(s, ux, uy, out=None):
    """D2Q5 advection-diffusion equilibrium for scalar s carried by velocity (ux,uy). Per-population evaluation keeps the
    temporaries small (one plane at a time) — faster + lower peak memory than the (5,nx,ny) broadcast."""
    if out is None:
        out = np.empty((5, *s.shape))
    for q in range(5):
        out[q] = TW[q] * s * (1.0 + 3.0 * (TCX[q] * ux + TCY[q] * uy))
    return out


def moments9(f):
    """fused D2Q9 macroscopic moments (ρ, u_x, u_y) in ONE pass over the 9 populations — ~2× faster than the 3-pass
    rho=f.sum(0) / (CX[:,None,None]*f).sum(0) form (which builds large (9,nx,ny) CX·f temporaries) and BIT-IDENTICAL."""
    rho = f[0].copy(); mx = np.zeros_like(rho); my = np.zeros_like(rho)
    for q in range(1, 9):
        rho += f[q]
        if CX[q]:
            mx += CX[q] * f[q]
        if CY[q]:
            my += CY[q] * f[q]
    return rho, mx / rho, my / rho


def feq9(rho, ux, uy, out=None):
    """D2Q9 Navier-Stokes equilibrium. Per-population evaluation (small temporaries) — ~3× faster than the (9,nx,ny)
    broadcast, which is memory-bandwidth-bound on the large cu/cu² temporaries."""
    u2 = ux * ux + uy * uy
    if out is None:
        out = np.empty((9, *rho.shape))
    for q in range(9):
        cu = CX[q] * ux + CY[q] * uy
        out[q] = W[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
    return out
