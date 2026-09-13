"""Conservative numerical screen beside the frozen structural chunking rule.

Non-expansion does not certify the conditioning of cumulative-gate division or
triangular solves. Passing this screen still requires the original output gates.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from chunkable_recurrence_rule import (CHUNKABLE, detect_recurrence,
    gated_delta_sequential, gated_delta_chunked)

SOURCE = "S = gated_delta_rule(q, k, v, beta)"


def classify(src, transitions=None):
    structural = detect_recurrence(src)[0]
    result = dict(structural_class=structural, recommend_chunk=False,
                  reason="missing numerical transition evidence")
    if structural != CHUNKABLE:
        result['reason'] = "not structurally chunkable"
        return result
    if transitions is None:
        return result
    a = np.asarray(transitions, dtype=np.float64)
    if a.ndim != 3 or not a.shape[0] or not a.shape[1] or a.shape[1] != a.shape[2] or not np.isfinite(a).all():
        result['reason'] = "invalid transition evidence"
        return result
    norm = float(np.linalg.svd(a, compute_uv=False).max())
    radius = float(np.abs(np.linalg.eigvals(a)).max())
    result.update(max_operator_norm=norm, max_spectral_radius=radius,
                  recommend_chunk=norm <= 1.0,
                  reason="non-expansion screen passed; output gates required" if norm <= 1.0 else "expansive transition")
    return result


def observe():
    rows = []
    for gain in (.99, 1.01, 1.1, 1.5):
        for length in (32, 64, 128):
            rng = np.random.default_rng(19)
            width, chunk = 4, min(length, 64)
            q = rng.normal(size=(length, width)).astype(np.float32)
            k = rng.normal(size=(length, width)).astype(np.float32)
            k /= np.linalg.norm(k, axis=1)[:, None]
            v = rng.normal(size=(length, width)).astype(np.float32)
            g = np.full_like(k, gain)
            beta = np.full(length, .5, dtype=np.float32)
            transitions = g.astype(np.float64)[:, :, None] * (np.eye(width)[None] -
                beta[:, None, None] * k.astype(np.float64)[:, :, None] * k.astype(np.float64)[:, None, :])
            seq, seq_state = gated_delta_sequential(q, k, v, g, beta)
            chk, chk_state = gated_delta_chunked(q, k, v, g, beta, chunk)
            errors = [float(np.max(np.abs(a-b))) for a,b in ((seq,chk),(seq_state,chk_state))]
            budgets = [float(np.finfo(np.float32).eps * np.max(np.abs(a)) * np.sqrt(n*width))
                       for a,n in ((seq,chunk),(seq_state,length))]
            hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in
                      (q,k,v,g,beta,transitions,seq,chk,seq_state,chk_state)]
            rows.append(dict(gain=gain,length=length,chunk=chunk,errors=errors,budgets=budgets,
                output_budget_ratio=errors[0]/budgets[0], policy=classify(SOURCE,transitions),
                hashes=hashes,finite=bool(all(np.isfinite(a).all() for a in (seq,chk,seq_state,chk_state)))))
    return rows


def report():
    first, second = observe(), observe()
    stable = [r for r in first if r['gain']==.99]
    bad = next(r for r in first if r['gain']==1.1 and r['length']==64)
    nonnormal = classify(SOURCE, [[[.9, 10.], [0., .9]]])
    gates = dict(exact_repeat=first==second, all_finite=all(r['finite'] for r in first),
        measured_blowup=bad['output_budget_ratio']>1,
        expansive_refusal=bad['policy']['max_spectral_radius']>1 and not bad['policy']['recommend_chunk'],
        stable_control=all(r['policy']['recommend_chunk'] and all(e<=b for e,b in zip(r['errors'],r['budgets'])) for r in stable),
        unknown_refusal=not classify(SOURCE)['recommend_chunk'],
        nonfinite_refusal=not classify(SOURCE, [[[float('nan')]]])['recommend_chunk'],
        nonnormal_refusal=nonnormal['max_spectral_radius']<1 and not nonnormal['recommend_chunk'],
        nonlinear_refusal=not classify('h = tanh(W @ h)', [[[.9]]])['recommend_chunk'])
    return dict(schema=1,scope="synthetic CPU numerical boundary; no general stability certificate",
                cases=first,gates=gates,status="VERIFIED-FRESH" if all(gates.values()) else "OWN-GATE-FAIL")


if __name__ == '__main__':
    result = report()
    target = Path('reports/chunkable_stability_v1.json')
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    raise SystemExit(0 if all(result['gates'].values()) else 1)
