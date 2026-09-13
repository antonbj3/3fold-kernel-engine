"""Durable evidence emission for the FSI cells.

Each headline cell saves a machine-inspectable JSON artifact under ``artifacts/``
(``artifacts/<cell_name>.json``) carrying the COMPUTED numbers + sigma + each
gate's PASS/FAIL, the external-anchor PROVENANCE each render->match is checked
against, and the three independent CROSS-CHECKS (computed-vs-known-reference,
NULL/falsifier, over-determination) as explicit PASS/FAIL booleans with the
numbers behind them.

This is PURE ADDITIVE serialization: the cell still prints exactly what it
printed before; ``emit`` only writes what the cell already computed, never
recomputes physics, never alters gates or stdout. Dependency-light (json + os
only); no timestamps (Date.now is non-deterministic -- omitted by design).
"""
import json
import os

SCHEMA_VERSION = "1.0"
_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


def _clean(o):
    """Recursively coerce numpy scalars/arrays and non-finite floats into
    JSON-safe Python types WITHOUT importing numpy (duck-typed)."""
    if o is None or isinstance(o, (bool, int, str)):
        return o
    if isinstance(o, float):
        if o != o:                       # NaN
            return "NaN"
        if o == float("inf"):
            return "Infinity"
        if o == float("-inf"):
            return "-Infinity"
        return o
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "tolist"):             # numpy array (incl. 0-d)
        return _clean(o.tolist())
    if hasattr(o, "item"):               # numpy / wrapped scalar
        return _clean(o.item())
    return str(o)


def emit(cell_name, claim, numbers, sigma, gates, provenance, cross_checks):
    """Write ``artifacts/<cell_name>.json`` (pretty JSON). Returns the path.

    Parameters mirror the directive: NUMBERS + sigma + per-gate PASS/FAIL, the
    data/anchor PROVENANCE, and the AUTOMATED CROSS-CHECKS (the three kinds) as
    explicit booleans with computed-vs-reference numbers.
    """
    doc = {
        "schema_version": SCHEMA_VERSION,
        "cell": cell_name,
        "claim": claim,
        "numbers": _clean(numbers),
        "sigma": _clean(sigma),
        "gates": _clean(gates),
        "provenance": _clean(provenance),
        "cross_checks": _clean(cross_checks),
    }
    os.makedirs(_DIR, exist_ok=True)
    path = os.path.join(_DIR, cell_name + ".json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")
    return path
