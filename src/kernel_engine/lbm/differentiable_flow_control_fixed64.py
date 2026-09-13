"""Deterministic double-precision flow control with bounded scaled descent."""
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp

wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN
from kernel_engine.lbm import differentiable_flow_control as baseline

DEV = baseline.DEV
NX, NY, T = baseline.NX, baseline.NY, baseline.T
DRIVE, OMEGA = baseline.DRIVE, baseline.OMEGA
ACC_SCALE = baseline.ACC_SCALE
DETERMINISTIC_ACCUMULATION = True
_cx = wp.array(baseline.CX.astype(np.float64), dtype=wp.float64, device=DEV)
_cy = wp.array(baseline.CY.astype(np.float64), dtype=wp.float64, device=DEV)
_w = wp.array(baseline.WT.astype(np.float64), dtype=wp.float64, device=DEV)
_opp = wp.array(baseline.OP, dtype=wp.int32, device=DEV)
_solid = wp.array(baseline._solid_np, dtype=wp.int32, device=DEV)
_jet = wp.array(baseline._jet_np.astype(np.float64), dtype=wp.float64, device=DEV)
_pmask = wp.array(baseline._pmask_np.astype(np.float64), dtype=wp.float64, device=DEV)
_nprobe = baseline._nprobe
_f0 = baseline._f0.astype(np.float64)

@wp.kernel
def collide(f0: wp.array3d(dtype=wp.float64), fpost: wp.array3d(dtype=wp.float64), jet: wp.array2d(dtype=wp.float64), ctrl: wp.array(dtype=wp.float64), step: int, cx: wp.array(dtype=wp.float64), cy: wp.array(dtype=wp.float64), w: wp.array(dtype=wp.float64), drive: wp.float64, omega: wp.float64):
    i, j = wp.tid()
    rho = wp.float64(wp.float64(0.0))
    mx = wp.float64(wp.float64(0.0))
    my = wp.float64(wp.float64(0.0))
    for k in range(9):
        fk = f0[k, i, j]
        rho += fk
        mx += cx[k] * fk
        my += cy[k] * fk
    local_drive = drive * (wp.float64(1.0) + jet[i, j] * ctrl[step])
    ux = mx / rho + wp.float64(0.5) * local_drive
    uy = my / rho
    usq = ux * ux + uy * uy
    for k in range(9):
        cu = cx[k] * ux + cy[k] * uy
        feq = w[k] * rho * (wp.float64(1.0) + wp.float64(3.0) * cu + wp.float64(4.5) * cu * cu - wp.float64(1.5) * usq)
        Fk = (wp.float64(1.0) - wp.float64(0.5) * omega) * wp.float64(3.0) * w[k] * cx[k] * local_drive
        fpost[k, i, j] = f0[k, i, j] - omega * (f0[k, i, j] - feq) + Fk

@wp.kernel
def stream(fpost: wp.array3d(dtype=wp.float64), f1: wp.array3d(dtype=wp.float64), solid: wp.array2d(dtype=wp.int32), cx: wp.array(dtype=wp.float64), cy: wp.array(dtype=wp.float64), opp: wp.array(dtype=wp.int32), nx: int, ny: int):
    i, j = wp.tid()
    for k in range(9):
        si = i - int(cx[k])
        sj = j - int(cy[k])
        if si < 0:
            si += nx
        if si >= nx:
            si -= nx
        if sj < 0:
            sj = 0
        if sj >= ny:
            sj = ny - 1
        if solid[si, sj] == 1:
            f1[k, i, j] = fpost[opp[k], i, j]
        else:
            f1[k, i, j] = fpost[k, si, sj]

@wp.kernel
def track_loss(f: wp.array3d(dtype=wp.float64), cx: wp.array(dtype=wp.float64), mask: wp.array2d(dtype=wp.float64), target: wp.float64, L: wp.array(dtype=wp.float64)):
    i, j = wp.tid()
    rho = wp.float64(wp.float64(0.0))
    mx = wp.float64(wp.float64(0.0))
    for k in range(9):
        fk = f[k, i, j]
        rho += fk
        mx += cx[k] * fk
    d = (mx / rho - target) * mask[i, j]
    wp.atomic_add(L, 0, d * d)

@wp.kernel
def probe_ux(f: wp.array3d(dtype=wp.float64), cx: wp.array(dtype=wp.float64), mask: wp.array2d(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    i, j = wp.tid()
    rho = wp.float64(wp.float64(0.0))
    mx = wp.float64(wp.float64(0.0))
    for k in range(9):
        fk = f[k, i, j]
        rho += fk
        mx += cx[k] * fk
    wp.atomic_add(out, 0, mx / rho * mask[i, j])

@wp.kernel
def track_loss_i64(f: wp.array3d(dtype=wp.float64), cx: wp.array(dtype=wp.float64), mask: wp.array2d(dtype=wp.float64), target: wp.float64, scale: wp.float64, L: wp.array(dtype=wp.int64)):
    i, j = wp.tid()
    rho = wp.float64(wp.float64(0.0))
    mx = wp.float64(wp.float64(0.0))
    for k in range(9):
        fk = f[k, i, j]
        rho += fk
        mx += cx[k] * fk
    d = (mx / rho - target) * mask[i, j]
    wp.atomic_add(L, 0, wp.int64(wp.round(wp.float64(d) * wp.float64(d) * scale)))

@wp.kernel
def probe_ux_i64(f: wp.array3d(dtype=wp.float64), cx: wp.array(dtype=wp.float64), mask: wp.array2d(dtype=wp.float64), scale: wp.float64, out: wp.array(dtype=wp.int64)):
    i, j = wp.tid()
    rho = wp.float64(wp.float64(0.0))
    mx = wp.float64(wp.float64(0.0))
    for k in range(9):
        fk = f[k, i, j]
        rho += fk
        mx += cx[k] * fk
    wp.atomic_add(out, 0, wp.int64(wp.round(wp.float64(mx / rho * mask[i, j]) * scale)))

def probe_ux_value(f, mask):
    """Sum of ux over the masked probe cells (order-invariant when DETERMINISTIC_ACCUMULATION)."""
    if not DETERMINISTIC_ACCUMULATION:
        o = wp.zeros(1, dtype=wp.float64, device=DEV)
        wp.launch(probe_ux, (NX, NY), inputs=[f, _cx, mask, o], device=DEV)
        wp.synchronize()
        return float(o.numpy()[0])
    o = wp.zeros(1, dtype=wp.int64, device=DEV)
    wp.launch(probe_ux_i64, (NX, NY), inputs=[f, _cx, mask, wp.float64(ACC_SCALE), o], device=DEV)
    wp.synchronize()
    return float(int(o.numpy()[0])) / ACC_SCALE

def track_loss_value(L, f, target):
    """Value of the tracking loss. The float array L stays for the adjoint (a quantiser has no useful
    derivative, so the tape keeps the float accumulation); the reported value is the int64 one."""
    if not DETERMINISTIC_ACCUMULATION:
        return float(L.numpy()[0])
    acc = wp.zeros(1, dtype=wp.int64, device=DEV)
    wp.launch(track_loss_i64, (NX, NY), inputs=[f, _cx, _pmask, target, wp.float64(ACC_SCALE), acc], device=DEV)
    wp.synchronize()
    return float(int(acc.numpy()[0])) / ACC_SCALE

def rollout(ctrl_np, target=None, tape=None, want_probe=False):
    rg = tape is not None
    ctrl = wp.array(ctrl_np.astype(np.float64), dtype=wp.float64, device=DEV, requires_grad=rg)
    cf = wp.array(_f0, dtype=wp.float64, device=DEV, requires_grad=rg)
    L = wp.zeros(1, dtype=wp.float64, device=DEV, requires_grad=rg)
    keep = [cf]

    def run():
        c = cf
        for t in range(T):
            fp = wp.zeros((9, NX, NY), dtype=wp.float64, device=DEV, requires_grad=rg)
            fn = wp.zeros((9, NX, NY), dtype=wp.float64, device=DEV, requires_grad=rg)
            wp.launch(collide, (NX, NY), inputs=[c, fp, _jet, ctrl, t, _cx, _cy, _w, DRIVE, OMEGA], device=DEV)
            wp.launch(stream, (NX, NY), inputs=[fp, fn, _solid, _cx, _cy, _opp, NX, NY], device=DEV)
            keep.append(fp)
            keep.append(fn)
            c = fn
        if not want_probe:
            wp.launch(track_loss, (NX, NY), inputs=[c, _cx, _pmask, target, L], device=DEV)
        return c
    if want_probe:
        c = run()
        wp.synchronize()
        return probe_ux_value(c, _pmask) / _nprobe
    if rg:
        with tape:
            cend = run()
    else:
        cend = run()
    wp.synchronize()
    return (L, ctrl, cend)


def selftest():
    if DEV != "cuda:0":
        raise RuntimeError("CUDA required")
    initial = np.zeros(T, np.float64)
    wake = rollout(initial, want_probe=True)
    target = 1.6 * wake
    check = np.ones(T, np.float64)
    tape = wp.Tape()
    loss, controls, final = rollout(check, target=target, tape=tape)
    tape.backward(loss=loss)
    gradient = controls.grad.numpy().copy()
    finite = bool(np.isfinite(gradient).all())
    differences = []
    for step in (5, 25, 50):
        plus, minus = check.copy(), check.copy()
        plus[step] += 0.1
        minus[step] -= 0.1
        lp, _, fp = rollout(plus, target=target)
        lm, _, fm = rollout(minus, target=target)
        fd = (track_loss_value(lp, fp, target) - track_loss_value(lm, fm, target)) / 0.2
        relative = abs(gradient[step] - fd) / (abs(fd) + 1e-12)
        differences.append(dict(step=step, adjoint=float(gradient[step]), finite_difference=fd,
                                relative_error=float(relative)))
    c = initial.copy()
    trace = []
    forward_evaluations = 0
    for iteration in range(50):
        tape = wp.Tape()
        loss, controls, final = rollout(c, target=target, tape=tape)
        value = track_loss_value(loss, final, target)
        tape.backward(loss=loss)
        g = controls.grad.numpy()
        finite = finite and bool(np.isfinite(g).all() and np.isfinite(final.numpy()).all())
        accepted = c.copy()
        accepted_value = value
        for trial in range(8):
            candidate = np.clip(c - (0.5 / (2 ** trial)) * np.sign(g), 0.0, 8.0)
            if np.array_equal(candidate, c):
                break
            candidate_loss, _, candidate_final = rollout(candidate, target=target)
            forward_evaluations += 1
            candidate_value = track_loss_value(candidate_loss, candidate_final, target)
            if candidate_value <= value:
                accepted, accepted_value = candidate, candidate_value
                break
        trace.append(dict(iteration=iteration, loss=value, accepted_loss=accepted_value,
                          maximum_step=float(np.max(np.abs(accepted-c))),
                          gradient_sha256=hashlib.sha256(g.tobytes()).hexdigest(),
                          controls_sha256=hashlib.sha256(accepted.tobytes()).hexdigest()))
        c = accepted
    final_probe = rollout(c, want_probe=True)
    recovery = (final_probe - wake) / (target - wake + 1e-30)
    gates = dict(original_fd=sum(r["relative_error"] < 0.05 for r in differences) >= 2,
                 original_recovery=recovery > 0.3,
                 finite=finite and bool(np.isfinite(c).all() and np.isfinite(recovery)),
                 original_control_bounds=bool(np.all(c >= 0) and np.all(c <= 8)),
                 original_step_bound=all(r["maximum_step"] <= 0.5 for r in trace),
                 accepted_loss_nonincreasing=all(r["accepted_loss"] <= r["loss"] for r in trace))
    return dict(wake=wake, target=target, final_probe=final_probe, recovery=recovery,
                fd=differences, controls=c.tolist(), trace=trace,
                line_search_forward_evaluations=forward_evaluations, backward_passes=51,
                gates=gates, precision="float64 populations and controls; frozen float32 input values cast exactly",
                runtime_mode="RUN_TO_RUN")


def main():
    result = selftest()
    root = Path(__file__).resolve().parents[3]
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports/differentiable_flow_control_fixed64.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("wake", "target", "final_probe", "recovery", "fd", "gates")}), flush=True)
    return 0 if all(result["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
