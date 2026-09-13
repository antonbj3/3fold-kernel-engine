"""Full-array evidence for eight fixed-order selftests with explicit variants."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from kernel_engine.certified_kernels.innovation_runtime_adjoint import config_from_reference

FLOW = "lbm/differentiable_flow_control"
VARIANT = "lbm/differentiable_flow_control_fixed64"
TAPE_SLOTS = {FLOW, "lbm/differentiable_lbm_probe", "lbm/differentiable_fsi_chain",
              "wave_fdtd/diff_wave_3d", "wave_fdtd/diff_wave_substrate", "wave_fdtd/xray_tomography_sigma"}


def worker(slot):
    import warp as wp
    wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN
    if slot == "wave_fdtd/xray_tomography_sigma":
        from kernel_engine.certified_kernels.innovation_tomography_record_bound import record_bound
        wp.config.deterministic_max_records = record_bound()
    selected = VARIANT if slot == FLOW else slot
    records = []
    backward_calls = 0
    finite = True
    original_numpy = wp.array.numpy
    original_backward = wp.Tape.backward

    def record(value, role):
        nonlocal finite
        a = np.ascontiguousarray(value)
        finite = finite and bool(np.isfinite(a).all())
        records.append(dict(ordinal=len(records), role=role, shape=list(a.shape),
                            dtype=str(a.dtype), bytes=a.nbytes,
                            sha256=hashlib.sha256(a.tobytes()).hexdigest()))

    def readback(array, *args, **kwargs):
        value = original_numpy(array, *args, **kwargs)
        record(value, "readback")
        return value

    def backward(tape, *args, **kwargs):
        nonlocal backward_calls
        result = original_backward(tape, *args, **kwargs)
        backward_calls += 1
        for primal, gradient in tape.gradients.items():
            if not isinstance(primal, wp.array) or not isinstance(gradient, wp.array):
                raise TypeError("Unsupported non-array tape gradient")
            record(original_numpy(primal), "tape_primal")
            record(original_numpy(gradient), "tape_gradient")
        return result

    out, err = io.StringIO(), io.StringIO()
    code = 0
    exception_classes = []
    wp.array.numpy = readback
    wp.Tape.backward = backward
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                runpy.run_path(str(ROOT / "src/kernel_engine" / (selected + ".py")), run_name="__main__")
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else int(exc.code is not None)
            except Exception as exc:
                code = 1
                exception_classes.append(type(exc).__name__)
                traceback.print_exc()
    finally:
        wp.array.numpy = original_numpy
        wp.Tape.backward = original_backward
    _, skip = config_from_reference()
    normalized = "\n".join(line for line in (out.getvalue() + err.getvalue()).splitlines() if not skip.search(line))
    if exception_classes:
        print(err.getvalue(), file=sys.stderr)
    result = dict(slot=slot, selected_module=selected, returncode=code, exception_classes=exception_classes,
                  normalized_text_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
                  records=records, backward_calls=backward_calls, finite=finite,
                  observed_bytes=sum(r["bytes"] for r in records),
                  gradient_arrays=sum(r["role"] == "tape_gradient" for r in records))
    print("CAPTURE_RESULT " + json.dumps(result), flush=True)


def child(args):
    from kernel_engine.kernel_gen.innovation_lbm_stream_baseline import idle
    idle()
    p = subprocess.run([sys.executable, str(Path(__file__).resolve()), *args],
                       cwd=ROOT, capture_output=True, text=True, timeout=1800)
    idle()
    lines = [line[15:] for line in p.stdout.splitlines() if line.startswith("CAPTURE_RESULT ")]
    if p.returncode or len(lines) != 1:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        raise RuntimeError("Capture worker failed")
    return json.loads(lines[0])


def main():
    modules, _ = config_from_reference()
    slots = [m for m, _ in modules]
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        if sys.argv[2] not in slots:
            raise ValueError("Unknown suite slot")
        worker(sys.argv[2])
        return 0
    if sys.argv[1:] == ["--sites"]:
        import warp as wp
        # Preserve the frozen float diagnostic controls. Only the int64 deltas
        # are gated here; fixed-order adjoints are checked by the eight workers.
        wp.config.deterministic = wp.DeterministicMode.NOT_GUARANTEED
        from kernel_engine.certified_kernels.determinism_sweep_int64 import site_checks
        rows = [dict(module=m, site=s, float_delta=f, int64_delta=i) for m, s, f, i in site_checks()]
        print("CAPTURE_RESULT " + json.dumps(rows), flush=True)
        return 0
    if sys.argv[1:]:
        raise ValueError("Expected no arguments, --sites, or --worker SLOT")
    sites = child(["--sites"])
    rows = []
    report_path = ROOT / "reports/fixed_order_selftests_v1.json"
    for slot in slots:
        legs = [child(["--worker", slot]) for _ in range(2)]
        gates = dict(both_selftests_pass=all(r["returncode"] == 0 for r in legs),
                     no_runtime_exceptions=all(not r["exception_classes"] for r in legs),
                     full_array_records_identical=legs[0]["records"] == legs[1]["records"],
                     full_text_identical=legs[0]["normalized_text_sha256"] == legs[1]["normalized_text_sha256"],
                     finite=all(r["finite"] for r in legs),
                     arrays_observed=all(len(r["records"]) > 0 for r in legs),
                     gradients_observed=slot not in TAPE_SLOTS or all(r["backward_calls"] > 0 and r["gradient_arrays"] > 0 for r in legs))
        rows.append(dict(slot=slot, selected_module=legs[0]["selected_module"], legs=legs, gates=gates))
        report_path.write_text(json.dumps(dict(sites=sites, selftests=rows, complete=False), indent=2) + "\n")
        print(json.dumps(dict(slot=slot, gates=gates, arrays_per_leg=[len(r["records"]) for r in legs],
                              observed_bytes_per_leg=[r["observed_bytes"] for r in legs],
                              backward_calls=[r["backward_calls"] for r in legs])), flush=True)
    gates = dict(all_eight_slots=len(rows) == 8,
                 all_selected_selftests_pass=all(r["gates"]["both_selftests_pass"] for r in rows),
                 all_array_and_text_records_identical=all(r["gates"]["full_array_records_identical"] and r["gates"]["full_text_identical"] for r in rows),
                 all_observations_valid=all(all(v for k, v in r["gates"].items() if k not in ("both_selftests_pass", "full_array_records_identical", "full_text_identical")) for r in rows),
                 all_int64_sites_exact=len(sites) == 10 and all(r["int64_delta"] == 0 for r in sites))
    report_path.write_text(json.dumps(dict(sites=sites, selftests=rows, complete=True, gates=gates,
                                           runtime_mode="RUN_TO_RUN", tomography_records=440,
                                           flow_variant=VARIANT, performance_measured=False), indent=2) + "\n")
    print(json.dumps(gates), flush=True)
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
