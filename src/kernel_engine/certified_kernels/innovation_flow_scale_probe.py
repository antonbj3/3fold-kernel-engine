"""Measure the frozen control optimizer's gradient and actuator scales."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def worker():
    import warp as wp
    wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN
    from kernel_engine.lbm import differentiable_flow_control as f
    if f.DEV != "cuda:0":
        raise RuntimeError("CUDA required")
    probes = {str(c): f.rollout(np.full(f.T, c, np.float32), want_probe=True) for c in (0, 1, 8)}
    target = 1.6 * probes["0"]
    gradients = {}
    for c in (0, 1):
        tape = wp.Tape()
        loss, ctrl, final = f.rollout(np.full(f.T, c, np.float32), target=target, tape=tape)
        value = f.track_loss_value(loss, final, target)
        tape.backward(loss=loss)
        g = ctrl.grad.numpy()
        gradients[str(c)] = dict(loss=value, gradient=g.tolist(),
                                 gradient_sha256=hashlib.sha256(g.tobytes()).hexdigest(),
                                 max_abs_gradient=float(np.max(np.abs(g))),
                                 original_max_step=float(np.max(np.abs(6000 * g))))
    fd = []
    c0 = np.ones(f.T, np.float32)
    for ti in (5, 25, 50):
        plus = c0.copy()
        minus = c0.copy()
        plus[ti] += 0.1
        minus[ti] -= 0.1
        lp, _, fp = f.rollout(plus, target=target)
        lm, _, fm = f.rollout(minus, target=target)
        difference = (f.track_loss_value(lp, fp, target) - f.track_loss_value(lm, fm, target)) / 0.2
        adjoint = gradients["1"]["gradient"][ti]
        relative = abs(adjoint - difference) / (abs(difference) + 1e-12)
        fd.append(dict(step=ti, adjoint=adjoint, finite_difference=difference, relative_error=relative))
    result = dict(probes=probes, target=target, gradients=gradients, fd=fd,
                  recovery_at_eight=(probes["8"] - probes["0"]) / (target - probes["0"] + 1e-30))
    print("RESULT " + json.dumps(result), flush=True)


def main():
    if sys.argv[1:] == ["--worker"]:
        worker()
        return 0
    if sys.argv[1:]:
        raise ValueError("Expected no arguments or --worker")
    from kernel_engine.kernel_gen.innovation_lbm_stream_baseline import idle
    legs = []
    for _ in range(2):
        idle()
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                               cwd=ROOT, capture_output=True, text=True)
        if child.returncode:
            print(child.stdout)
            print(child.stderr, file=sys.stderr)
            raise RuntimeError("Worker failed")
        idle()
        lines = [line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT ")]
        if len(lines) != 1:
            raise RuntimeError("Expected one worker result")
        legs.append(json.loads(lines[0]))
        print(json.dumps({k: v for k, v in legs[-1].items() if k != "gradients"}), flush=True)
    gates = dict(exact_repeats=legs[0] == legs[1],
                 finite=all(np.isfinite(g["gradient"]).all() and np.isfinite(g["loss"])
                            for r in legs for g in r["gradients"].values()),
                 positive_deficit=all(r["target"] > r["probes"]["0"] for r in legs),
                 nonzero_gradient=all(r["gradients"]["0"]["max_abs_gradient"] > 0 for r in legs),
                 original_fd=all(sum(v["relative_error"] < 0.05 for v in r["fd"]) >= 2 for r in legs))
    (ROOT / "reports/innovation_flow_scale_probe.json").write_text(json.dumps(dict(legs=legs, gates=gates), indent=2) + "\n")
    print(json.dumps(gates), flush=True)
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
