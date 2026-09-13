"""Conservative numerical guard beside the frozen structural chunk rule."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from chunkable_recurrence_rule import (CHUNKABLE, detect_recurrence,
                                      gated_delta_chunked, gated_delta_sequential)

SOURCE = 'state = gated_delta_rule(q, k, v, beta)'


def transition_guard(source, k=None, g=None, beta=None):
    structural = detect_recurrence(source)[0]
    if structural != CHUNKABLE:
        return dict(accepted=False, reason='unsupported_structure')
    if k is None or g is None or beta is None:
        return dict(accepted=False, reason='missing_numeric_transitions')
    k, g, beta = (np.asarray(a, dtype=np.float64) for a in (k, g, beta))
    if k.ndim != 2 or k.size == 0 or g.shape != k.shape or beta.shape != (len(k),):
        return dict(accepted=False, reason='invalid_shape')
    if not all(np.isfinite(a).all() for a in (k, g, beta)):
        return dict(accepted=False, reason='nonfinite_transition')
    norms = []
    for kt, gt, bt in zip(k, g, beta):
        operator = np.diag(gt) @ (np.eye(len(kt)) - bt * np.outer(kt, kt))
        norms.append(float(np.linalg.norm(operator, ord=2)))
    maximum = max(norms)
    return dict(accepted=maximum <= 1.0,
                reason='nonexpansive' if maximum <= 1.0 else 'expansive_transition',
                max_operator_norm=maximum)


def checked_evaluate(source, q, k, v, g, beta, chunk, initial):
    """Refuse expansive/unknown transitions, then apply the original error gates.

    This audit computes both evaluators and therefore makes no speed claim.
    Acceptance applies only to the supplied arrays, state and chunk size.
    """
    guard = transition_guard(source, k, g, beta)
    if not guard['accepted']:
        return None, guard
    if not isinstance(chunk, int) or chunk < 1:
        return None, dict(accepted=False, reason='invalid_chunk')
    sequential = gated_delta_sequential(q, k, v, g, beta, initial)
    chunked = gated_delta_chunked(q, k, v, g, beta, chunk, initial)
    eps = float(np.finfo(np.float32).eps)
    limits = [eps * float(np.max(np.abs(sequential[0]))) * np.sqrt(chunk * k.shape[1]),
              eps * float(np.max(np.abs(sequential[1]))) * np.sqrt(len(k) * k.shape[1])]
    errors = [float(np.max(np.abs(a - b))) for a, b in zip(sequential, chunked)]
    accepted = all(np.isfinite(b).all() for b in chunked) and all(e <= t for e, t in zip(errors, limits))
    return (chunked if accepted else None), dict(accepted=bool(accepted),
        reason='measured_bounds_pass' if accepted else 'numerical_error', errors=errors, limits=limits)


def experiment():
    rows = []
    arrays = {}
    for gate in (.98, 1.2):
        for chunk in (16, 32, 64, 128):
            q = np.ones((128, 1), np.float32)
            k = q.copy(); v = q.copy(); g = np.full_like(k, gate)
            beta = np.full(128, .15, np.float32)
            initial = np.array([[.25]], np.float32)
            sequential = gated_delta_sequential(q, k, v, g, beta, initial)
            chunked = gated_delta_chunked(q, k, v, g, beta, chunk, initial)
            _, decision = checked_evaluate(SOURCE, q, k, v, g, beta, chunk, initial)
            error = float(np.max(np.abs(sequential[0] - chunked[0])))
            bound = float(np.finfo(np.float32).eps * np.max(np.abs(sequential[0])) * np.sqrt(chunk))
            key = f'{gate}_{chunk}'
            for name, a in zip(('sequential_output', 'sequential_state', 'chunked_output', 'chunked_state'), (*sequential, *chunked)):
                arrays[key + '_' + name] = a
            # Evaluate the represented float32 inputs in float64 for the scalar oracle.
            a = float(g[0, 0]) * (1.0 - float(beta[0]))
            oracle = float(initial[0, 0]); history = []
            for _ in range(128):
                oracle = a * oracle + float(beta[0]); history.append(oracle)
            arrays[key + '_oracle64'] = np.array(history)
            rows.append(dict(gate=gate, chunk=chunk, eigenvalue=a,
                             max_output_error=error, original_bound=bound,
                             error_over_bound=error / bound, decision=decision,
                             sequential_endpoint=float(sequential[0][-1, 0]),
                             chunked_endpoint=float(chunked[0][-1, 0]),
                             oracle_endpoint=oracle))
    unstable = [r for r in rows if r['gate'] == 1.2]
    stable = {r['chunk']: r for r in rows if r['gate'] == .98}
    gates = dict(coverage=len(rows) == 8,
                 error_blowup=unstable[-1]['max_output_error'] > 1000 * unstable[0]['max_output_error']
                              and unstable[-1]['error_over_bound'] > 1000,
                 expansive_eigenvalue=all(r['eigenvalue'] > 1 for r in unstable),
                 syntax_accepts_numeric_refuses=detect_recurrence(SOURCE)[0] == CHUNKABLE
                     and all(not r['decision']['accepted'] and r['decision']['reason'] == 'expansive_transition' for r in unstable),
                 stable_control=stable[32]['decision']['accepted'],
                 stable_roundoff_failure_retained=not stable[16]['decision']['accepted']
                     and stable[16]['decision']['reason'] == 'numerical_error')
    hashes = {k: hashlib.sha256(a.tobytes()).hexdigest() for k, a in arrays.items()}
    return dict(rows=rows, gates=gates, hashes=hashes), arrays


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        report, arrays = experiment()
        np.savez_compressed(sys.argv[2], **arrays)
        print(json.dumps(report, sort_keys=True))
        return 0
    reports = []; arrays = []
    with tempfile.TemporaryDirectory() as tmp:
        for leg in range(2):
            output = Path(tmp) / f'{leg}.npz'
            p = subprocess.run([sys.executable, __file__, '--worker', str(output)], check=True, capture_output=True, text=True)
            reports.append(json.loads(p.stdout))
            with np.load(output) as data:
                arrays.append({k: data[k].copy() for k in data.files})
    exact = reports[0] == reports[1] and all(arrays[0][k].tobytes() == arrays[1][k].tobytes() for k in arrays[0])
    gates = dict(reports[0]['gates'], exact_full_repeats=exact)
    output = Path(__file__).resolve().parent / 'artifacts'
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / 'chunkable_stability_v1.npz', **{f'{i}_{k}': a for i, leg in enumerate(arrays) for k, a in leg.items()})
    (output / 'chunkable_stability_v1.json').write_text(json.dumps(dict(legs=reports, gates=gates), indent=2) + '\n')
    print(json.dumps(gates))
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
