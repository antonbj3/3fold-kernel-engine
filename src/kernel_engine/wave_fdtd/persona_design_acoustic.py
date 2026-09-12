#!/usr/bin/env python3
"""ACOUSTIC RESONANCE ENGINE (★acoustic ladder foundation) — air-column / cavity eigenmodes, known-GT validated.

Acoustic ladder: tuning fork -> tuned enclosure -> WIND INSTRUMENT (flute/whistle/ocarina) -> Aztec death whistle, with
the simulated sound validated against a REAL recording = a real-world anchor for acoustics (not sim-to-sim). The engine is
the eigenmodes of the acoustic wave equation (grad^2 p + k^2 p = 0) = the resonances = the tones.

INSTRUMENT VALIDATION against ANALYTIC truth BEFORE any design conclusion (same discipline as the structural FEM):
  - open-open pipe (flute):            f_n = n.c/2L
  - open-closed pipe (clarinet/whistle): f_n = (2n-1).c/4L     (even harmonics absent -> "hollow" tone)
  - 2D rectangular cavity:             f_mn = (c/2).sqrt((m/Lx)^2+(n/Ly)^2)
Real air: c = 343 m/s -> frequencies in Hz (audible).

Run:  python3 persona_design_acoustic.py
"""
import numpy as np
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import eigsh

C_AIR = 343.0   # m/s


# ───────────────────────── 1D Laplacians (Neumann = rigid/closed, Dirichlet = open/pressure-node) ─────────────────────────
def _neumann_1d(N):
    main = -2.0*np.ones(N); off = np.ones(N-1)
    L = diags([off, main, off], [-1, 0, 1]).tolil()
    L[0, 0] = -1.0; L[N-1, N-1] = -1.0          # zero-gradient (rigid) at both ends
    return L.tocsr()


def pipe_frequencies(L_phys, bc, n_modes=4, N=400, c=C_AIR):
    """Resonances of a 1D pipe. bc ∈ {'OO' open-open, 'OC' open-closed, 'CC' closed-closed}."""
    h = L_phys/(N-1)
    Ln = _neumann_1d(N)
    if bc == 'CC':                               # both rigid (Neumann)
        A = -(Ln.toarray())/h**2
    elif bc == 'OO':                             # both open (Dirichlet) → interior nodes 1..N-2
        main = -2.0*np.ones(N-2); off = np.ones(N-3)
        A = -(diags([off, main, off], [-1, 0, 1]).toarray())/h**2
    elif bc == 'OC':                             # open(node0 Dirichlet) + closed(node N-1 Neumann)
        A = -(Ln.toarray()[1:, 1:])/h**2
    vals = np.linalg.eigvalsh(A)
    vals = np.sort(vals[vals > 1e-6])
    return c*np.sqrt(vals[:n_modes])/(2*np.pi)


def cavity_2d_frequencies(Lx, Ly, n_modes=6, nx=60, ny=60, c=C_AIR):
    """Rigid-wall 2D rectangular cavity acoustic modes (Neumann Laplacian, Kronecker sum)."""
    hx, hy = Lx/(nx-1), Ly/(ny-1)
    Dx = -_neumann_1d(nx)/hx**2
    Dy = -_neumann_1d(ny)/hy**2
    A = (kron(Dx, identity(ny)) + kron(identity(nx), Dy)).tocsc()
    vals = eigsh(A, k=n_modes+1, sigma=-1e-3, which='LM')[0]
    vals = np.sort(np.real(vals)); vals = vals[vals > 1e-3]
    return c*np.sqrt(vals[:n_modes])/(2*np.pi)


# ───────────────────────── validations vs analytic ─────────────────────────
def validate():
    print("="*92)
    print("ACOUSTIC RESONANCE ENGINE — eigensolver validated vs ANALYTIC pipe/cavity formulas (known-GT)")
    print("="*92)
    L = 0.5
    # open-open (flute-like): f_n = n c / 2L
    f_oo = pipe_frequencies(L, 'OO', 4)
    a_oo = np.array([n*C_AIR/(2*L) for n in (1, 2, 3, 4)])
    e_oo = np.max(np.abs(f_oo-a_oo)/a_oo)
    print(f"\n(P-OO) open-open pipe L={L}m: f={np.round(f_oo,1)}  analytic={np.round(a_oo,1)}  "
          f"max.err {e_oo:.2%}: {'PASS' if e_oo < 0.02 else 'FAIL'}")
    # open-closed (clarinet/whistle): f_n = (2n-1) c / 4L — ODD harmonics only ("hollow" timbre)
    f_oc = pipe_frequencies(L, 'OC', 4)
    a_oc = np.array([(2*n-1)*C_AIR/(4*L) for n in (1, 2, 3, 4)])
    e_oc = np.max(np.abs(f_oc-a_oc)/a_oc)
    print(f"(P-OC) open-closed pipe L={L}m: f={np.round(f_oc,1)}  analytic={np.round(a_oc,1)}  "
          f"max.err {e_oc:.2%}: {'PASS' if e_oc < 0.02 else 'FAIL'}  (odd harmonics → 'hollow' timbre)")
    # 2D cavity: f_mn = (c/2)√((m/Lx)²+(n/Ly)²)
    Lx, Ly = 0.4, 0.3
    f_cav = cavity_2d_frequencies(Lx, Ly, 5)
    analytic = sorted([(C_AIR/2)*np.hypot(m/Lx, n/Ly) for m in range(4) for n in range(4)])[1:6]
    e_cav = np.max(np.abs(np.sort(f_cav)-np.array(analytic))/np.array(analytic))
    print(f"(CAV)  2D cavity {Lx}×{Ly}m: f={np.round(np.sort(f_cav),1)}  analytic={np.round(analytic,1)}  "
          f"max.err {e_cav:.2%}: {'PASS' if e_cav < 0.05 else 'FAIL'}")
    ok = e_oo < 0.02 and e_oc < 0.02 and e_cav < 0.05
    print(f"\n  ACOUSTIC INSTRUMENT VALIDATED: {'all PASS — trustworthy as the wind-instrument/whistle oracle' if ok else 'FAIL'}")
    return ok


# ───────────────────────── wind-instrument design: tune length to a target pitch (known inverse) ─────────────────────────
def design_pipe_for_pitch(target_hz, bc='OC', c=C_AIR):
    """Inverse: find the pipe length giving a target fundamental pitch. Analytic + verified by the solver."""
    if bc == 'OC':
        L_analytic = c/(4*target_hz)             # f1=(c/4L) → L=c/4f
    else:
        L_analytic = c/(2*target_hz)
    f_solver = pipe_frequencies(L_analytic, bc, 1)[0]
    return L_analytic, f_solver


def main():
    if not validate():
        print("instrument not validated — stopping before any design conclusion.")
        return
    print("\n" + "="*92)
    print("WIND-INSTRUMENT DESIGN — tune pipe length to a target musical pitch (known inverse)")
    print("="*92)
    notes = [("A4", 440.0), ("C5", 523.25), ("E5", 659.25), ("A5", 880.0)]
    print(f"  {'note':>5} {'target Hz':>10} {'pipe L (open-closed)':>22} {'solver f1':>11} {'err':>7}")
    for name, hz in notes:
        L, f = design_pipe_for_pitch(hz, 'OC')
        print(f"  {name:>5} {hz:>10.1f} {L*1000:>19.1f}mm {f:>10.1f}Hz {abs(f-hz)/hz:>6.2%}")
    print(f"\n  ➤ the acoustic engine inverts pitch→geometry (here a length; next: bore profile / hole "
          f"placement / death-whistle dual chamber). NEXT reality-anchor: compare the synthesized spectrum to "
          f"a REAL recording (YouTube) — the honest sim2real for sound (a recording IS reality, not a sim).")
    print("="*92)


if __name__ == "__main__":
    main()
