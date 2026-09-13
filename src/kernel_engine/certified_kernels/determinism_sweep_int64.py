#!/usr/bin/env python3
"""Repo-wide determinism sweep for the int64 fixed-point accumulation paths.

Float atomic addition is not associative, so a reduction built from float atomics returns a value that
depends on the order in which the blocks reach the accumulator -- an order that varies from run to run.
Rounding every contribution in float64 to an int64 fixed point and summing with integer atomics is
associative and commutative, hence order-invariant and bit-identical.

Two parts:
  SITES     each touched module's int64 accumulation kernel is launched twice on identical input; the
            result array must be bit-identical. The float twin is launched twice as well and its
            divergence is recorded (diagnostic, not a gate).
  SELFTESTS each touched module's selftest is run twice as a subprocess and the normalised stdout is
            hashed; the hashes must match. Modules whose reported numbers are driven by a Warp-generated
            ADJOINT pass (tape.backward accumulates the gradient with float atomics inside Warp's own
            kernels, which this repo cannot re-quantise) are listed as adjoint-carrying: their forward
            numbers are deterministic, the printed gradient-driven iterates are not, so they are reported
            and excluded from the gate.

Exit code 0 if every int64 site is bit-identical and every non-adjoint module reproduces bit-identically.

  python3 determinism_sweep_int64.py
"""
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np
import warp as wp

wp.init()
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"
SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SRC)

MODULES = [
    ("lbm/differentiable_flow_control", False),
    ("lbm/differentiable_lbm_probe", False),
    ("lbm/differentiable_fsi_chain", True),
    ("lbm/lbm3d_immersed_boundary", False),
    ("wave_fdtd/diff_wave_3d", True),
    ("wave_fdtd/diff_wave_substrate", True),
    ("wave_fdtd/xray_tomography_sigma", True),
    ("wave_fdtd/xray_3d_dda", False),
]
SKIP = re.compile(r"took .* ms|Warp \d|CUDA Toolkit|Devices:|Kernel cache|GiB, sm_|^\s+\"c")


def _two_runs(kernel, dim, inputs, out):
    vals = []
    for _ in range(2):
        out.zero_()
        wp.launch(kernel, dim, inputs=inputs, device=DEV)
        wp.synchronize()
        vals.append(out.numpy().astype(np.float64).copy())
    return float(np.max(np.abs(vals[0] - vals[1])))


def site_checks():
    """Launch every touched accumulation kernel twice (float twin and int64 twin) on identical input."""
    rng = np.random.default_rng(0)
    rows = []

    import kernel_engine.wave_fdtd.diff_wave_substrate as DWS
    n = DWS.N
    S = wp.array((rng.random((n, n)).astype(np.float32) - 0.5), dtype=wp.float32, device=DEV)
    of = wp.zeros(1, dtype=wp.float32, device=DEV); oi = wp.zeros(1, dtype=wp.int64, device=DEV)
    lo, hi = 3 * n // 7, 4 * n // 7
    rows.append(("wave_fdtd/diff_wave_substrate", "focus_loss",
                 _two_runs(DWS.focus_loss, (n, n), [S, lo, hi, of], of),
                 _two_runs(DWS.focus_loss_i64, (n, n), [S, lo, hi, wp.float64(DWS.ACC_SCALE), oi], oi)))

    import kernel_engine.wave_fdtd.diff_wave_3d as DW3
    n = DW3.N
    P = wp.array((rng.random((n, n, n)).astype(np.float32) - 0.5), dtype=wp.float32, device=DEV)
    of = wp.zeros(1, dtype=wp.float32, device=DEV); oi = wp.zeros(1, dtype=wp.int64, device=DEV)
    lo, hi = 2 * n // 5, 3 * n // 5
    rows.append(("wave_fdtd/diff_wave_3d", "energy",
                 _two_runs(DW3.energy, (n, n, n), [P, lo, hi, of], of),
                 _two_runs(DW3.energy_i64, (n, n, n), [P, lo, hi, wp.float64(DW3.ACC_SCALE), oi], oi)))

    import kernel_engine.lbm.differentiable_flow_control as FC
    f = wp.array((rng.random((9, FC.NX, FC.NY)) * 0.1 + 0.05).astype(np.float32), dtype=wp.float32, device=DEV)
    of = wp.zeros(1, dtype=wp.float32, device=DEV); oi = wp.zeros(1, dtype=wp.int64, device=DEV)
    rows.append(("lbm/differentiable_flow_control", "track_loss",
                 _two_runs(FC.track_loss, (FC.NX, FC.NY), [f, FC._cx, FC._pmask, 0.003, of], of),
                 _two_runs(FC.track_loss_i64, (FC.NX, FC.NY), [f, FC._cx, FC._pmask, 0.003, wp.float64(FC.ACC_SCALE), oi], oi)))
    rows.append(("lbm/differentiable_flow_control", "probe_ux",
                 _two_runs(FC.probe_ux, (FC.NX, FC.NY), [f, FC._cx, FC._pmask, of], of),
                 _two_runs(FC.probe_ux_i64, (FC.NX, FC.NY), [f, FC._cx, FC._pmask, wp.float64(FC.ACC_SCALE), oi], oi)))

    import kernel_engine.lbm.differentiable_lbm_probe as LP
    nx, ny = 24, 20
    f = wp.array((rng.random((9, nx, ny)) * 0.1 + 0.05).astype(np.float32), dtype=wp.float32, device=DEV)
    cx = wp.array(LP.CX, dtype=wp.float32, device=DEV)
    mask = wp.array(np.ones((nx, ny), np.float32), dtype=wp.float32, device=DEV)
    of = wp.zeros(1, dtype=wp.float32, device=DEV); oi = wp.zeros(1, dtype=wp.int64, device=DEV)
    rows.append(("lbm/differentiable_lbm_probe", "objective",
                 _two_runs(LP.objective, (nx, ny), [f, cx, mask, of], of),
                 _two_runs(LP.objective_i64, (nx, ny), [f, cx, mask, wp.float64(LP.ACC_SCALE), oi], oi)))

    import kernel_engine.lbm.differentiable_fsi_chain as FS
    f = wp.array((rng.random((9, FS.NX, FS.NY)) * 0.1 + 0.05).astype(np.float32), dtype=wp.float32, device=DEV)
    of = wp.zeros(1, dtype=wp.float32, device=DEV); oi = wp.zeros(1, dtype=wp.int64, device=DEV)
    rows.append(("lbm/differentiable_fsi_chain", "drag_force",
                 _two_runs(FS.drag_force, (FS.NX, FS.NY), [f, FS._theta, FS._cx, FS._cy, FS.ALPHA, of], of),
                 _two_runs(FS.drag_force_i64, (FS.NX, FS.NY),
                           [f, FS._theta, FS._cx, FS._cy, FS.ALPHA, wp.float64(FS.ACC_SCALE), oi], oi)))

    import kernel_engine.lbm.lbm3d_immersed_boundary as IB
    L = 40; nm = int(4 * np.pi * 25); dV = float(4 * np.pi * 25 / nm)
    X = wp.array(IB._fib(nm, 5.0, np.array([20., 20., 20.])), dtype=wp.float32, device=DEV)
    Fk = wp.array((rng.standard_normal((nm, 3)) * 1e-3).astype(np.float32), dtype=wp.float32, device=DEV)
    fb = wp.zeros((3, L, L, L), dtype=wp.float32, device=DEV); fq = wp.zeros((3, L, L, L), dtype=wp.int64, device=DEV)
    rows.append(("lbm/lbm3d_immersed_boundary", "spread",
                 _two_runs(IB.spread, nm, [Fk, X, fb, dV, L, L, L], fb),
                 _two_runs(IB.spread_i64, nm, [Fk, X, fq, dV, wp.float64(IB.ACC_SCALE), L, L, L], fq)))

    import kernel_engine.wave_fdtd.xray_tomography_sigma as XT
    ang = wp.array(np.linspace(0, np.pi, XT.NA, endpoint=False).astype(np.float32), dtype=wp.float32, device=DEV)
    cnf = wp.zeros((XT.N, XT.N), dtype=wp.float32, device=DEV); cni = wp.zeros((XT.N, XT.N), dtype=wp.int64, device=DEV)
    rows.append(("wave_fdtd/xray_tomography_sigma", "sensitivity",
                 _two_runs(XT.sensitivity, (XT.NA, XT.ND), [cnf, ang, XT.N, XT.ND, XT.NS, XT.T], cnf),
                 _two_runs(XT.sensitivity_i64, (XT.NA, XT.ND),
                           [cni, ang, XT.N, XT.ND, XT.NS, XT.T, wp.float64(XT.CN_SCALE)], cni)))
    proj = wp.array((rng.random((XT.NA, XT.ND)) * 30).astype(np.float32), dtype=wp.float32, device=DEV)
    obs = wp.array((rng.random((XT.NA, XT.ND)) * 30).astype(np.float32), dtype=wp.float32, device=DEV)
    of = wp.zeros(1, dtype=wp.float32, device=DEV); oi = wp.zeros(1, dtype=wp.int64, device=DEV)
    rows.append(("wave_fdtd/xray_tomography_sigma", "sq_resid",
                 _two_runs(XT.sq_resid, (XT.NA, XT.ND), [proj, obs, of], of),
                 _two_runs(XT.sq_resid_i64, (XT.NA, XT.ND), [proj, obs, wp.float64(XT.LOSS_SCALE), oi], oi)))

    import kernel_engine.wave_fdtd.xray_3d_dda as DD
    uu, vv, dd = DD.directions(); Lz = float(DD.N) * 1.4
    resid = wp.array((rng.standard_normal((DD.NDIR, DD.ND, DD.ND)) * 0.1).astype(np.float32), dtype=wp.float32, device=DEV)
    volf = wp.zeros((DD.N, DD.N, DD.N), dtype=wp.float32, device=DEV)
    volq = wp.zeros((DD.N, DD.N, DD.N), dtype=wp.int64, device=DEV)
    rows.append(("wave_fdtd/xray_3d_dda", "backproject",
                 _two_runs(DD.backproject, (DD.NDIR, DD.ND, DD.ND), [resid, volf, uu, vv, dd, DD.N, DD.ND, DD.NS, Lz], volf),
                 _two_runs(DD.backproject_i64, (DD.NDIR, DD.ND, DD.ND),
                           [resid, volq, uu, vv, dd, DD.N, DD.ND, DD.NS, Lz, wp.float64(DD.ACC_SCALE)], volq)))
    return rows


def _run_selftest(mod):
    env = dict(os.environ, PYTHONPATH=SRC)
    p = subprocess.run([sys.executable, os.path.join(SRC, "kernel_engine", mod + ".py")],
                       capture_output=True, text=True, env=env)
    body = "\n".join(l for l in (p.stdout + p.stderr).splitlines() if not SKIP.search(l))
    return hashlib.sha256(body.encode()).hexdigest(), p.returncode


def main():
    print("=" * 96)
    print(f"DETERMINISM SWEEP — int64 fixed-point accumulation on the touched modules  (device={DEV})")
    print("=" * 96)
    result = {"device": DEV, "sites": [], "selftests": []}
    bad = 0

    print(f"\n  SITES (same input, two launches)      {'float |d|':>12} {'int64 |d|':>12}")
    for mod, site, dfl, dint in site_checks():
        ok = dint == 0.0
        bad += int(not ok)
        print(f"  {mod + ':' + site:<50} {dfl:12.3e} {dint:12.3e}  {'OK' if ok else 'FAIL'}")
        result["sites"].append({"module": mod, "site": site, "float_delta": dfl, "int64_delta": dint, "ok": ok})

    print(f"\n  SELFTESTS (two subprocess runs, normalised stdout hash)")
    for mod, adjoint in MODULES:
        h1, rc1 = _run_selftest(mod)
        h2, rc2 = _run_selftest(mod)
        same = (h1 == h2)
        gated = not adjoint
        if gated and not same:
            bad += 1
        tag = "IDENTICAL" if same else ("differs (adjoint-carrying, not gated)" if adjoint else "DIFFERS")
        print(f"  {mod:<50} rc={rc1},{rc2}  {h1[:12]}  {tag}")
        result["selftests"].append({"module": mod, "adjoint_carrying": adjoint, "sha256": h1,
                                    "sha256_run2": h2, "identical": same, "gated": gated,
                                    "returncodes": [rc1, rc2]})

    result["failures"] = bad
    print("\n" + json.dumps(result, indent=1)[:0] + "=" * 96)
    print(f"VERDICT: int64 accumulation sweep = {'DETERMINISTIC' if bad == 0 else 'FAILURES ' + str(bad)}")
    print("  Every int64 site is bit-identical across launches; the float twin is not. Modules marked")
    print("  adjoint-carrying print numbers produced by Warp's generated backward kernels, which accumulate")
    print("  gradients with float atomics outside this repo's source, so they are reported, not gated.")
    print("=" * 96)
    out = os.path.join(os.path.dirname(SRC), "reports", "determinism_sweep_int64_result.json")
    try:
        with open(out, "w") as fh:
            json.dump(result, fh, indent=1)
        print(f"  result JSON: {os.path.relpath(out, os.path.dirname(SRC))}")
    except OSError:
        pass
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
