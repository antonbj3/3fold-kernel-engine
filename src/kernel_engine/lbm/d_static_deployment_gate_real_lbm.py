#!/usr/bin/env python3
"""d_static_deployment_gate_real_lbm.py -- THE DEPLOYMENT-GATE edge (CPU/static-analysis, honest-forced).

Real usage: the cert must GATE what ships (a kernel deploys IFF certified), not run as a side analysis. This
builds that gate: given a real solver kernel's SOURCE TEXT (no execution, no GPU), it STATICALLY extracts the
cert-facets and emits SHIP / DON'T-SHIP / ABSTAIN per kernel. Core analyzer is pure stdlib (ast + re) -- ZERO
warp/GPU dependency -- so the gate itself runs on any kernel source, CPU-only, before a single CUDA context exists.

FACETS (mechanical, AST-derived, each cross-checked by an INDEPENDENT regex census -- path A vs path B must
agree; over-determination, not eyeballing):

  R1 min-traffic:  per-kernel STATE-array touch count (reads+writes), loop-unrolled by literal range() trip
                   counts, x dtype-bytes -> bytes/voxel. CONSTANT/lookup tables (never written anywhere in the
                   file -- e.g. lattice directions cx/cy/w/opp) are excluded from the byte count, matching the
                   standard LBM roofline convention (small constant tables live in cache/registers, not DRAM)
                   and this project's own established number: 72 B/voxel = 9 f_i read + 9 write x4B (measured
                   in d_certvector_on_real_lbm.py). Compared against that DAG-min where one has been derived;
                   report-only where it has not (no invented reference).

  R2 determinism:  for each WRITE (plain indexed-assign, or wp.atomic_*):
                     - plain-assign -> EXCLUSIVITY check: does EVERY tid-axis variable (bound directly from
                       wp.tid()) appear as a BARE, unmodified Name in the write's index tuple? If yes ->
                       GATHER (deterministic by construction, no atomic needed). If no (the index was offset/
                       computed, e.g. i+cx[k]) and NOT atomic-protected -> RACE (unproven exclusivity; multiple
                       source threads/iterations can target the same address -- lost update, undefined result).
                       This is a conservative SUFFICIENT condition (identity-mapped tid axes), not a full
                       injectivity prover -- named explicitly in the static/runtime boundary section.
                     - wp.atomic_* -> classify the accumulator array's declared dtype: FLOAT-class (float32/64,
                       vecN/quat/matNN) -> order-dependent; INT-class (int32/64/uint*) -> order-invariant.
                       EXTERNALLY ANCHORED: CPU-forced (CUDA_VISIBLE_DEVICES='') re-run of this project's REAL,
                       committed det_accumulation_probe.py (not reimplemented) -- MEASURED below.

  R4 addressing:   file-scope write-provenance (per I's real corpus result: structural/fixed addressing -> a-
                   priori COMPUTED; input-KEYED addressing -> contention needs the input distribution -> PROVE-
                   NANCE-GATED). An array is STATE/DATA iff it is EVER a store-target (plain or atomic) anywhere
                   in the file; else CONSTANT/TABLE. A write's address is DATA-KEYED iff its index expression's
                   provenance traces to a Subscript-read on a STATE array; else STRUCTURAL (tid/loop-var/scalar-
                   param/CONSTANT-table lookups only -- reading a fixed lattice constant does not taint).

  R2-precision:    X = Y / Z (or Z-augdiv) where Z was accumulated via `+=` inside a `for` loop in the SAME
                   kernel (a moment-like reduction, e.g. rho = sum_k g[k]) -> PRECISION-SENSITIVE flag
                   (conditioning ~ 1/Z, blows up as Z -> 0, e.g. near-vacuum/void voxels). A caveat, not a
                   ship-blocker (matches the task's own expectation for C's real kernel).

VERDICT composition (mechanical precedence over the per-kernel finding-set):
  RACE anywhere                      -> DON'T-SHIP (silent lost-update; worse than nondeterminism, it's wrong)
  ATOMIC-FLOAT on a DATA-KEYED write  -> DON'T-SHIP vs the strict bit-exact bar (the SAME bar
                                         d_certvector_on_real_lbm.py's determinism-eye measures, epsilon==0).
                                         Escape hatch named, not taken: a relaxed convergence-tolerance cert
                                         target would need a RUNTIME multi-order residual/bit-diff check.
  ATOMIC-FLOAT on a STRUCTURAL write  -> ABSTAIN: order-dependence is real (measured), but whether it BREACHES
                                         the cert needs the QoI's sigma_min/conditioning -- a physics fact this
                                         gate cannot read off the source text. Refer to the upstream computation
                                         -level requirement-cert (d_computation_level_requirement_cert.py) for
                                         THIS QoI's determinism_needed().
  ATOMIC-INT (any addressing) or      -> SHIP (value-deterministic). DATA-KEYED+int additionally attaches a
  GATHER-only (no atomics, no race)      non-blocking caveat: throughput/contention is provenance-gated per I
                                         (worst-case traffic needs an input-distribution spec, separate axis
                                         from value-correctness).
  + R1 traffic report, + R2-precision division flags attached as non-blocking caveats on any SHIP.

TEST INSTANCES (verdict PRE-REGISTERED below, BEFORE running the analyzer):
  1. C/lbm_gpu_fast.py :: collide_stream        expect SHIP     (gather; 72B/voxel; ux=mx/rho flagged)
  2. C/lbm_gpu_fast.py :: macro_ux_soa          expect SHIP     (gather; ux=mx/rho flagged)
  3. C/lbm_voxel_aero_gpu.py :: collide         expect SHIP     (independent 2nd real impl, split-kernel style)
  4. C/lbm_voxel_aero_gpu.py :: stream          expect SHIP     (independent pull-stream, false-reject probe)
  5. C/lbm_voxel_aero_gpu.py :: macro_ux        expect SHIP     (independent macro-extraction, div flagged)
  6. C/lbm_fsi_gpu.py :: stream_bounce          expect ABSTAIN  (REAL float atomic_add(fx,0,..) -- structural
                                                                 fixed-const target 0 -- momentum-exchange force)
  7. C/warp_bodybody_jacobi_gpu.py :: k_jac_vel expect DON'T-SHIP (REAL float atomic_add(dv,bi,..), bi=cbi[c]
                                                                 DATA-KEYED -- a REAL committed contact-impulse
                                                                 scatter kernel; upgrades the "modeled" ask)
  8. constructed :: macro_scatter_broken        expect DON'T-SHIP (RACE: push-style neighbor-scatter, NOT
                                                                 atomic-protected, i+cx[k]/j+cy[k] not raw tid)

GATE-QUALITY: actual vs pre-registered verdict, per instance. False-reject = expect SHIP, got other. Miss =
expect DON'T-SHIP, got SHIP. Both reported as counts (the falsifiable, machine-checked claim), not narrated.

PLUS a 9th, SEPARATE force-check (not counted in gate-quality, a symmetric-QC audit of the gate ITSELF):
  reverse_copy :: dst[N-1-i]=src[i] -- a PROVABLY bijective, race-free-without-atomics write. PRE-REGISTERED
  PREDICTION: the exclusivity check requires a BARE tid Name in the index tuple; 'N-1-i' is a BinOp, so predict
  a FALSE-POSITIVE RACE flag (the named conservative-bias limitation, forced rather than merely asserted).

Run: python3 d_static_deployment_gate_real_lbm.py
(the analyzer itself needs no venv/warp; the venv is only used for the external-anchor subprocess re-run)
"""
import ast
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.dirname(os.path.abspath(__file__))
_W_DIR = os.path.join(os.path.dirname(C_DIR), "warp_gpu")
_OWN_VENV_PY = __import__("pathlib").Path(__file__).resolve().parents[2] / "python3"
VENV_PY = str(_OWN_VENV_PY) if _OWN_VENV_PY.exists() else "python3"

# =====================================================================================================
# dtype classification
# =====================================================================================================
FLOAT_HINTS = ("float", "vec2", "vec3", "vec4", "quat", "mat22", "mat33", "mat44", "spatial")
INT_HINTS = ("int8", "int16", "int32", "int64", "uint")


def dtype_class(ann_src: str) -> str:
    s = ann_src.lower()
    if any(h in s for h in FLOAT_HINTS):
        return "float"
    if any(h in s for h in INT_HINTS) or re.search(r"\bint\b", s):
        return "int"
    return "unknown"


def dtype_bytes(ann_src: str) -> int:
    s = ann_src.lower()
    if "vec3" in s or "vec2" in s:
        return 4 * (3 if "vec3" in s else 2)
    if "quat" in s:
        return 16
    if "64" in s:
        return 8
    return 4


# =====================================================================================================
# AST helpers: kernel discovery + signature parsing
# =====================================================================================================
def parse_file(path):
    src = open(path).read()
    return src, ast.parse(src, filename=path)


def find_kernels(tree):
    """All @wp.kernel-decorated FunctionDefs, in source order."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                dec_s = ast.unparse(dec)
                if "wp.kernel" in dec_s:
                    out.append(node)
                    break
    return out


def signature_params(fn):
    """-> (array_params: {name: dtype_class}, scalar_params: set(name))"""
    array_params, scalar_params = {}, set()
    for a in fn.args.args:
        ann = ast.unparse(a.annotation) if a.annotation is not None else ""
        if "wp.array" in ann:
            m = re.search(r"dtype\s*=\s*([\w.]+)", ann)
            dt = m.group(1) if m else ann
            array_params[a.arg] = dtype_class(dt)
        else:
            scalar_params.add(a.arg)
    return array_params, scalar_params


# =====================================================================================================
# File-scope STATE-vs-CONSTANT array classification (written-anywhere-in-file => STATE/DATA)
# =====================================================================================================
def collect_write_sites(fn, array_params):
    """All (array_name, index_component_nodes, is_atomic, lineno) write sites in a kernel, found via ast.walk
    (order doesn't matter for THIS pass -- we only need the SET of written array names + each site's index expr).
    """
    sites = []
    for node in ast.walk(fn):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id in array_params:
                    idx_nodes = _slice_components(t.slice)
                    sites.append((t.value.id, idx_nodes, False, node.lineno))
        if isinstance(node, ast.Call):
            fs = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
            if re.match(r"^wp\.atomic_(add|sub|min|max|cas|exch)$", fs):
                if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in array_params:
                    idx_nodes = list(node.args[1:-1]) if len(node.args) > 2 else []
                    sites.append((node.args[0].id, idx_nodes, True, node.lineno))
    return sites


def _slice_components(slice_node):
    """Normalize a Subscript's slice into a list of index-expression nodes (handles arr[a,b,c] as a Tuple)."""
    if isinstance(slice_node, ast.Tuple):
        return list(slice_node.elts)
    return [slice_node]


def _swap_pairs(tree):
    """host-level tuple swaps / reassignment-cycles: `fA, fB = fB, fA` (the ping-pong double-buffer idiom).
    A ping-ponged INPUT-role buffer is, BY DESIGN, never itself a Store-target at the kernel level (that's the
    point of ping-pong: avoid in-place read/write hazard) -- so the plain written-anywhere rule alone would
    misclassify it as a read-only CONSTANT table. Detect the swap and let STATE propagate through it."""
    pairs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Tuple) and len(tgt.elts) == len(node.value.elts):
                for a, b in zip(tgt.elts, node.value.elts):
                    if isinstance(a, ast.Name) and isinstance(b, ast.Name):
                        pairs.append((a.id, b.id))
    return pairs


def file_state_arrays(tree):
    """Array names that are EVER a write-target (plain or atomic) in ANY kernel in this file => STATE/DATA.
    Everything else that's an array-typed kernel parameter somewhere in the file => CONSTANT/TABLE. Propagated
    through host-level ping-pong swaps (see _swap_pairs) to a fixed point."""
    state, all_arrays = set(), set()
    for fn in find_kernels(tree):
        arr, _ = signature_params(fn)
        all_arrays |= set(arr)
        for name, _idx, _atomic, _ln in collect_write_sites(fn, arr):
            state.add(name)
    pairs = [(a, b) for a, b in _swap_pairs(tree) if a in all_arrays and b in all_arrays]
    changed = True
    while changed:
        changed = False
        for a, b in pairs:
            if a in state and b not in state:
                state.add(b); changed = True
            if b in state and a not in state:
                state.add(a); changed = True
    return state, all_arrays - state


# =====================================================================================================
# Provenance tracker: STRUCTURAL (tid / loop-var / scalar-param / CONSTANT-table lookup) vs DATA-KEYED
# (traces to a Subscript-read on a STATE/DATA array). Forward walk in program order; tid-axis names recorded.
# =====================================================================================================
class Provenance:
    def __init__(self, array_params, scalar_params, state_arrays):
        self.array_params = array_params
        self.scalar_params = scalar_params
        self.state_arrays = state_arrays
        self.bindings = {}      # name -> 'structural' | 'data-keyed'
        self.tid_axes = set()   # names bound directly from wp.tid()

    def of(self, node):
        if isinstance(node, ast.Constant):
            return "structural"
        if isinstance(node, ast.Name):
            if node.id in self.scalar_params:
                return "structural"
            return self.bindings.get(node.id, "structural")  # unresolved => conservative structural default
        if isinstance(node, ast.Call):
            fs = ast.unparse(node.func)
            if fs == "wp.tid()" or fs == "wp.tid":
                return "structural"
            # casts / math funcs (int(), float(), wp.round(), wp.cross(), wp.length(), wp.abs(), wp.min/max...):
            # propagate OR over args
            return "data-keyed" if any(self.of(a) == "data-keyed" for a in node.args) else "structural"
        if isinstance(node, ast.Subscript):
            base = node.value
            if isinstance(base, ast.Name):
                if base.id in self.state_arrays:
                    return "data-keyed"          # reading simulation STATE always taints
                if base.id in self.array_params:
                    return "structural"          # CONSTANT/TABLE lookup (e.g. lattice cx/cy/w/opp) -- no taint
            return "data-keyed"                  # unknown subscript base: conservative taint
        if isinstance(node, ast.BinOp):
            return "data-keyed" if (self.of(node.left) == "data-keyed" or self.of(node.right) == "data-keyed") else "structural"
        if isinstance(node, ast.UnaryOp):
            return self.of(node.operand)
        if isinstance(node, ast.Tuple) or isinstance(node, ast.List):
            return "data-keyed" if any(self.of(e) == "data-keyed" for e in node.elts) else "structural"
        if isinstance(node, ast.Compare):
            return "data-keyed" if any(self.of(x) == "data-keyed" for x in [node.left] + node.comparators) else "structural"
        if isinstance(node, ast.IfExp):
            return "data-keyed" if any(self.of(x) == "data-keyed" for x in [node.test, node.body, node.orelse]) else "structural"
        return "structural"

    def bind_assign(self, node):
        if not isinstance(node, ast.Assign):
            return
        val = node.value
        is_tid = False
        try:
            is_tid = ast.unparse(val) in ("wp.tid()",) or (isinstance(val, ast.Call) and ast.unparse(val.func) == "wp.tid")
        except Exception:
            pass
        prov = "structural" if is_tid else self.of(val)
        for t in node.targets:
            if isinstance(t, ast.Name):
                self.bindings[t.id] = prov
                if is_tid:
                    self.tid_axes.add(t.id)
            elif isinstance(t, ast.Tuple):
                for e in t.elts:
                    if isinstance(e, ast.Name):
                        self.bindings[e.id] = prov
                        if is_tid:
                            self.tid_axes.add(e.id)

    def walk_body(self, stmts):
        """Forward pass in program order (recursing into For/If) binding provenance as we go."""
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                self.bind_assign(stmt)
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                prior = self.bindings.get(stmt.target.id, "structural")
                self.bindings[stmt.target.id] = "data-keyed" if (prior == "data-keyed" or self.of(stmt.value) == "data-keyed") else "structural"
            elif isinstance(stmt, ast.For):
                if isinstance(stmt.target, ast.Name):
                    self.bindings[stmt.target.id] = "structural"  # literal-range loop vars are structural
                self.walk_body(stmt.body)
            elif isinstance(stmt, ast.If):
                self.walk_body(stmt.body)
                self.walk_body(stmt.orelse)


# =====================================================================================================
# Exclusivity check (sufficient, conservative): every tid-axis appears as a BARE Name in the index tuple
# =====================================================================================================
def is_exclusive_write(index_nodes, tid_axes):
    if not tid_axes:
        return False  # no tid found at all -> cannot prove exclusivity (single-thread probe kernels handled separately)
    found = {n.id for n in index_nodes if isinstance(n, ast.Name) and n.id in tid_axes}
    return found == tid_axes


# =====================================================================================================
# Division / precision facet: X = Y / Z where Z accumulated via `+=` inside a `for` loop in this kernel
# =====================================================================================================
def division_flags(fn):
    accumulated = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.For):
            for inner in ast.walk(node):
                if isinstance(inner, ast.AugAssign) and isinstance(inner.op, ast.Add) and isinstance(inner.target, ast.Name):
                    accumulated.add(inner.target.id)
    flags = []
    for node in ast.walk(fn):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and isinstance(node.right, ast.Name):
            if node.right.id in accumulated:
                flags.append(f"line {node.lineno}: '{ast.unparse(node)}' divides by '{node.right.id}' "
                             f"(accumulated via += over a loop) -> PRECISION-SENSITIVE as {node.right.id}->0")
    return flags


# =====================================================================================================
# R1 traffic counter: recursive-descent over statement lists, literal-range loop unrolling, branch handling
# (If containing a `return` => separate TERMINAL path; If without `return` => value-alternative, elementwise-MAX)
# =====================================================================================================
def _literal_range_len(iter_node):
    if isinstance(iter_node, ast.Call) and ast.unparse(iter_node.func) == "range" and len(iter_node.args) == 1:
        a = iter_node.args[0]
        if isinstance(a, ast.Constant) and isinstance(a.value, int):
            return a.value
    return None


def _touch_in_expr(node, arrays, mult, reads):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and sub.value.id in arrays:
            reads[sub.value.id] = reads.get(sub.value.id, 0) + mult


def _count_stmt(stmt, arrays, mult, reads, writes):
    """Count array touches in a single non-control-flow statement (Assign/AugAssign/Expr-Call)."""
    if isinstance(stmt, (ast.Assign, ast.AugAssign)):
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        for t in targets:
            if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id in arrays:
                writes[t.value.id] = writes.get(t.value.id, 0) + mult
                for comp in _slice_components(t.slice):
                    _touch_in_expr(comp, arrays, mult, reads)
            elif isinstance(t, ast.Subscript):
                for comp in _slice_components(t.slice):
                    _touch_in_expr(comp, arrays, mult, reads)
        _touch_in_expr(stmt.value, arrays, mult, reads)
    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        fs = ast.unparse(call.func)
        if re.match(r"^wp\.atomic_(add|sub|min|max|cas|exch)$", fs) and call.args and isinstance(call.args[0], ast.Name) and call.args[0].id in arrays:
            writes[call.args[0].id] = writes.get(call.args[0].id, 0) + mult
            for a in call.args[1:]:
                _touch_in_expr(a, arrays, mult, reads)
        else:
            _touch_in_expr(call, arrays, mult, reads)
    else:
        _touch_in_expr(stmt, arrays, mult, reads)


def _merge_max(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = max(out.get(k, 0), v)
    return out


def count_traffic_paths(stmts, arrays, mult, reads, writes):
    """Returns a list of (reads, writes) dict-pairs, one per distinct TERMINAL execution path through `stmts`
    (a top-level `if ...: <stmts>; return` produces 2 terminal paths; a value-alternative if/else without
    `return` is merged via elementwise-MAX and the walk continues as ONE path)."""
    reads, writes = dict(reads), dict(writes)
    for idx, stmt in enumerate(stmts):
        if isinstance(stmt, ast.For):
            n = _literal_range_len(stmt.iter)
            if n is None:
                n = 1  # symbolic trip-count: cannot unroll statically, report as-if-once (named limitation)
            sub = count_traffic_paths(stmt.body, arrays, mult * n, reads, writes)
            assert len(sub) == 1, "loop body with early-return not expected in these kernels"
            reads, writes = sub[0]
            continue
        if isinstance(stmt, ast.If):
            body_paths = count_traffic_paths(stmt.body, arrays, mult, reads, writes)
            else_paths = count_traffic_paths(stmt.orelse, arrays, mult, reads, writes) if stmt.orelse else [(dict(reads), dict(writes))]
            ends_in_return = any(isinstance(s, ast.Return) for s in stmt.body)
            if ends_in_return:
                assert len(else_paths) == 1
                reads, writes = else_paths[0]
                rest = count_traffic_paths(stmts[idx + 1:], arrays, mult, reads, writes)
                return body_paths + rest
            else:
                assert len(body_paths) == 1 and len(else_paths) == 1
                reads = _merge_max(body_paths[0][0], else_paths[0][0])
                writes = _merge_max(body_paths[0][1], else_paths[0][1])
                continue
        if isinstance(stmt, ast.Return):
            return [(reads, writes)]
        _count_stmt(stmt, arrays, mult, reads, writes)
    return [(reads, writes)]


# =====================================================================================================
# Per-kernel facet extraction + verdict
# =====================================================================================================
def analyze_kernel(fn, state_arrays, const_arrays):
    array_params, scalar_params = signature_params(fn)
    prov = Provenance(array_params, scalar_params, state_arrays)
    prov.walk_body(fn.body)

    write_sites = collect_write_sites(fn, array_params)
    findings = []  # list of dict(kind, array, addressing, dtype, lineno)
    for arr, idx_nodes, is_atomic, lineno in write_sites:
        addressing = "DATA-KEYED" if any(prov.of(n) == "data-keyed" for n in idx_nodes) else "STRUCTURAL"
        if is_atomic:
            dt = array_params.get(arr, "unknown")
            kind = "ATOMIC-FLOAT" if dt == "float" else ("ATOMIC-INT" if dt == "int" else "ATOMIC-UNKNOWN-DTYPE")
        else:
            excl = is_exclusive_write(idx_nodes, prov.tid_axes)
            kind = "GATHER" if excl else "RACE"
        findings.append({"array": arr, "kind": kind, "addressing": addressing, "lineno": lineno})

    # R1 traffic (STATE arrays only; CONSTANT tables excluded, matches roofline convention)
    paths = count_traffic_paths(fn.body, set(state_arrays) & set(array_params), 1, {}, {})
    traffic_bytes_per_path = []
    for reads, writes in paths:
        rb = sum(c * dtype_bytes(_ann_of(fn, a)) for a, c in reads.items())
        wb = sum(c * dtype_bytes(_ann_of(fn, a)) for a, c in writes.items())
        traffic_bytes_per_path.append({"reads": reads, "writes": writes, "read_bytes": rb, "write_bytes": wb, "total_bytes": rb + wb})

    div_flags = division_flags(fn)

    # ---- verdict precedence: RACE > ATOMIC-FLOAT+DATA-KEYED > ATOMIC-FLOAT+STRUCTURAL > else SHIP ----
    verdict, reason = "SHIP", "gather-only / int-atomics-only, no race"
    caveats = []
    if any(f["kind"] == "RACE" for f in findings):
        bad = [f for f in findings if f["kind"] == "RACE"]
        verdict = "DON'T-SHIP"
        reason = f"RACE: non-exclusive write(s) not atomic-protected: {[(f['array'], f['lineno']) for f in bad]}"
    elif any(f["kind"] == "ATOMIC-FLOAT" and f["addressing"] == "DATA-KEYED" for f in findings):
        bad = [f for f in findings if f["kind"] == "ATOMIC-FLOAT" and f["addressing"] == "DATA-KEYED"]
        verdict = "DON'T-SHIP"
        reason = (f"ATOMIC-FLOAT on DATA-KEYED target(s) {[(f['array'], f['lineno']) for f in bad]}: order-dependent "
                  f"value AND unbounded worst-case contention (provenance-gated, per I) -- fails the strict bit-exact "
                  f"bar. Escape hatch NOT taken here: a relaxed convergence-tolerance target needs a RUNTIME "
                  f"multi-order residual/bit-diff check (out of static scope).")
    elif any(f["kind"] == "ATOMIC-FLOAT" and f["addressing"] == "STRUCTURAL" for f in findings):
        bad = [f for f in findings if f["kind"] == "ATOMIC-FLOAT" and f["addressing"] == "STRUCTURAL"]
        verdict = "ABSTAIN"
        reason = (f"ATOMIC-FLOAT on STRUCTURAL (fixed) target(s) {[(f['array'], f['lineno']) for f in bad]}: "
                  f"order-dependence is real (externally measured) but whether it BREACHES the cert needs this "
                  f"QoI's sigma_min/conditioning -- not statically knowable from source. Refer upstream to "
                  f"d_computation_level_requirement_cert.py's determinism_needed() for this specific QoI.")
    else:
        dk_int = [f for f in findings if f["kind"] == "ATOMIC-INT" and f["addressing"] == "DATA-KEYED"]
        if dk_int:
            caveats.append(f"ATOMIC-INT on DATA-KEYED target(s) {[(f['array'], f['lineno']) for f in dk_int]}: "
                           f"value-deterministic (int, order-invariant), but THROUGHPUT/contention is "
                           f"provenance-gated per I (worst-case traffic needs an input-distribution spec).")
    if div_flags:
        caveats.extend(div_flags)

    return {
        "kernel": fn.name,
        "findings": findings,
        "traffic_paths": traffic_bytes_per_path,
        "division_flags": div_flags,
        "verdict": verdict,
        "reason": reason,
        "caveats": caveats,
    }


def _ann_of(fn, array_name):
    for a in fn.args.args:
        if a.arg == array_name and a.annotation is not None:
            return ast.unparse(a.annotation)
    return ""


# =====================================================================================================
# PATH B: independent regex census (cross-check for over-determination, not a replacement for path A)
# =====================================================================================================
def regex_cross_check(src, fn_name):
    m = re.search(rf"def {re.escape(fn_name)}\(([^)]*(?:\)[^)]*)*?)\):(.*?)(?=\n@wp\.kernel|\nif __name__|\Z)", src, re.S)
    if not m:
        # fallback: locate by name then take up to the next top-level def/decorator
        i = src.find(f"def {fn_name}(")
        j = src.find("\n@wp.kernel", i + 1)
        j2 = src.find("\ndef ", i + 1)
        end = min([x for x in (j, j2, len(src)) if x != -1])
        body_src = src[i:end]
    else:
        body_src = m.group(0)
    atomic_calls = re.findall(r"wp\.atomic_(?:add|sub|min|max)\(\s*(\w+)", body_src)
    # division candidates: '= <expr> / DIVISOR'; a MOMENT-division (path A's criterion) additionally requires
    # DIVISOR to appear with a literal '+=' token somewhere in the SAME body (independent textual re-check of
    # path A's "accumulated via += inside a loop" rule -- not just "any division exists")
    div_candidates = set(re.findall(r"=\s*[\w.]+\s*/\s*(\w+)", body_src))
    div_targets = [d for d in div_candidates if re.search(rf"\b{re.escape(d)}\s*\+=", body_src)]
    range9 = len(re.findall(r"for\s+\w+\s+in\s+range\(9\)", body_src))
    return {"atomic_targets": atomic_calls, "division_targets": div_targets, "range9_loops": range9}


# =====================================================================================================
# External anchor: CPU-forced re-run of the REAL, committed det_accumulation_probe.py
# =====================================================================================================
def external_anchor_float_vs_int():
    code = (
        "import sys, numpy as np\n"
        f"sys.path.insert(0, {C_DIR!r})\n"
        "import warp as wp\n"
        "wp.init()\n"
        "import det_accumulation_probe as dap\n"
        "dap.DEV = 'cpu'\n"
        "rng = np.random.default_rng(0)\n"
        "N=8; C=20000\n"
        "body = rng.integers(0, N, size=C).astype(np.int32)\n"
        "imp = (rng.standard_normal((C,3))*0.3).astype(np.float32)\n"
        "perm = rng.permutation(C)\n"
        "fA = dap.run_float(body, imp, N); fB = dap.run_float(body[perm], imp[perm], N)\n"
        "iA = dap.run_int(body, imp, N); iB = dap.run_int(body[perm], imp[perm], N)\n"
        "print('F_DIV', float(np.max(np.abs(fA-fB))))\n"
        "print('I_DIV', float(np.max(np.abs(iA-iB))))\n"
    )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        r = subprocess.run([VENV_PY, "-c", code], capture_output=True, text=True, env=env, timeout=180)
        f_div = float(re.search(r"F_DIV ([\d.e+-]+)", r.stdout).group(1))
        i_div = float(re.search(r"I_DIV ([\d.e+-]+)", r.stdout).group(1))
        return {"ok": True, "f_div": f_div, "i_div": i_div, "cpu_forced": True, "stderr_tail": r.stderr[-400:]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# =====================================================================================================
# constructed broken variant (embedded source text, NOT executed -- static analysis only)
# =====================================================================================================
BROKEN_SRC = '''
import warp as wp

@wp.kernel
def macro_scatter_broken(f: wp.array3d(dtype=wp.float32), cx: wp.array(dtype=wp.float32), cy: wp.array(dtype=wp.float32),
                          solid: wp.array2d(dtype=wp.int32),
                          rho_acc: wp.array2d(dtype=wp.float32), mx_acc: wp.array2d(dtype=wp.float32)):
    """'optimized' moment computation: each cell PUSHES its post-collision population's momentum contribution
    to its DOWNSTREAM neighbor's accumulator instead of GATHERING from upstream (classic push-vs-pull GPU port
    bug). Multiple upstream cells concurrently scatter-add into the SAME downstream accumulator with a plain
    += (no atomic) -- lost updates (RACE), not just reordering."""
    i, j = wp.tid()
    if solid[i, j] == 1:
        return
    for k in range(9):
        fk = f[i, j, k]
        di = i + int(cx[k])
        dj = j + int(cy[k])
        if 0 <= di and di < 999999 and 0 <= dj and dj < 999999:
            rho_acc[di, dj] = rho_acc[di, dj] + fk
            mx_acc[di, dj] = mx_acc[di, dj] + cx[k] * fk
'''

# a PROVABLY-SAFE bijective-but-OFFSET write (reversal permutation: N-1-i is unique per i, no two threads
# collide) -- used ONLY to FORCE-CHECK the exclusivity-checker's stated conservative-bias limitation (a
# sufficient-not-necessary condition) rather than merely asserting it in prose. Pre-registered PREDICTION:
# the gate will flag this as RACE (a FALSE POSITIVE) because `N-1-i` is not a BARE tid Name in the index.
REVERSAL_SRC = '''
import warp as wp

@wp.kernel
def reverse_copy(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32), N: int):
    """dst[N-1-i] = src[i] -- a permutation (bijective), hence race-free WITHOUT atomics: no two threads i,i'
    can ever target the same N-1-i. A human can prove this; the gate's syntactic exclusivity check cannot."""
    i = wp.tid()
    dst[N - 1 - i] = src[i]
'''


# =====================================================================================================
# main
# =====================================================================================================
# independently-established DAG-min reference (NOT invented here): 72 B/voxel = 9 f_i read + 9 write x4B,
# the D2Q9 LBM fluid-cell information floor, MEASURED in d_certvector_on_real_lbm.py against C's real kernel.
D2Q9_DAG_MIN_BYTES = 72

INSTANCES = [
    ("C/lbm_gpu_fast.py", os.path.join(C_DIR, "lbm_gpu_fast.py"), "collide_stream", "SHIP"),
    ("C/lbm_gpu_fast.py", os.path.join(C_DIR, "lbm_gpu_fast.py"), "macro_ux_soa", "SHIP"),
    ("C/lbm_voxel_aero_gpu.py", os.path.join(C_DIR, "lbm_voxel_aero_gpu.py"), "collide", "SHIP"),
    ("C/lbm_voxel_aero_gpu.py", os.path.join(C_DIR, "lbm_voxel_aero_gpu.py"), "stream", "SHIP"),
    ("C/lbm_voxel_aero_gpu.py", os.path.join(C_DIR, "lbm_voxel_aero_gpu.py"), "macro_ux", "SHIP"),
    ("C/lbm_fsi_gpu.py", os.path.join(C_DIR, "lbm_fsi_gpu.py"), "stream_bounce", "ABSTAIN"),
    ("C/warp_bodybody_jacobi_gpu.py", os.path.join(_W_DIR, "warp_bodybody_jacobi_gpu.py"), "k_jac_vel", "DON'T-SHIP"),
    ("constructed/broken", None, "macro_scatter_broken", "DON'T-SHIP"),
]


def main():
    print("=" * 116)
    print("DEPLOYMENT-GATE (static, CPU-only) -- extracts cert-facets from real kernel SOURCE, emits SHIP/DON'T-SHIP/ABSTAIN")
    print("=" * 116)

    print("\n[EXTERNAL ANCHOR] CPU-forced (CUDA_VISIBLE_DEVICES='') re-run of REAL det_accumulation_probe.py "
          "(genchi-genbutsu: the actual committed code, not reimplemented) -- underwrites the R2 float/int rule:")
    anchor = external_anchor_float_vs_int()
    if anchor["ok"]:
        print(f"   float atomic_add, order A vs shuffled B: max|delta| = {anchor['f_div']:.3e}  "
              f"{'-> order-DEPENDENT (nonzero, confirms rule)' if anchor['f_div'] > 0 else '-> UNEXPECTED: zero'}")
        print(f"   int64 atomic_add, order A vs shuffled B: max|delta| = {anchor['i_div']:.3e}  "
              f"{'-> order-INVARIANT (zero, confirms rule)' if anchor['i_div'] == 0.0 else '-> UNEXPECTED: nonzero'}")
        anchor_pass = anchor["f_div"] > 0.0 and anchor["i_div"] == 0.0
    else:
        print(f"   ANCHOR SUBPROCESS FAILED: {anchor['error']} -- falling back to the committed script's own "
              f"documented measured result (float non-zero / int64 exactly zero) as a CITED, not re-executed, anchor.")
        anchor_pass = None
    print(f"   ANCHOR {'CONFIRMED (machine-measured this run)' if anchor_pass else ('CITED-ONLY' if anchor_pass is None else 'FAILED')}")

    results = []
    for label, path, kname, expected in INSTANCES:
        if path is not None:
            src, tree = parse_file(path)
        else:
            src, tree = BROKEN_SRC, ast.parse(BROKEN_SRC)
        state_arrays, const_arrays = file_state_arrays(tree)
        fn = next(f for f in find_kernels(tree) if f.name == kname)
        res = analyze_kernel(fn, state_arrays, const_arrays)
        rb = regex_cross_check(src, kname)

        # path A vs path B cross-check on the headline facts
        ast_atomic_targets = sorted({f["array"] for f in res["findings"] if f["kind"].startswith("ATOMIC")})
        agree_atomics = set(ast_atomic_targets) == set(rb["atomic_targets"])
        ast_has_div = len(res["division_flags"]) > 0
        rb_has_div = len(rb["division_targets"]) > 0
        agree_div = ast_has_div == rb_has_div

        print(f"\n{'-'*112}\n[{label} :: {kname}]  STATE-arrays(file)={sorted(state_arrays)}  CONSTANT-tables(file)={sorted(const_arrays)}")
        for f in res["findings"]:
            print(f"   write '{f['array']}' @line {f['lineno']}: {f['kind']:22s} addressing={f['addressing']}")
        for p in res["traffic_paths"]:
            vs_min = ""
            if p["total_bytes"] > 0:
                ratio = p["total_bytes"] / D2Q9_DAG_MIN_BYTES
                if abs(ratio - 1.0) < 1e-9:
                    vs_min = f"  == D2Q9 DAG-min ({D2Q9_DAG_MIN_BYTES}B) -> AT the information floor, no redundant traffic"
                elif kname in ("collide_stream", "collide", "stream", "macro_ux_soa", "macro_ux"):
                    vs_min = f"  = {ratio:.2f}x the D2Q9 DAG-min ({D2Q9_DAG_MIN_BYTES}B) -> {'redundant re-read, non-blocking perf flag' if ratio>1 else 'below the fluid-cell floor (lighter diagnostic kernel, not a full update)'}"
            print(f"   R1 traffic (one path): reads={p['reads']} writes={p['writes']} -> {p['total_bytes']} B{vs_min}")
        if res["division_flags"]:
            for d in res["division_flags"]:
                print(f"   R2-precision FLAG: {d}")
        print(f"   path-A/path-B cross-check: atomics {ast_atomic_targets} vs {rb['atomic_targets']} "
              f"agree={agree_atomics}; division-flag-present {ast_has_div} vs {rb_has_div} agree={agree_div}")
        match = res["verdict"] == expected
        print(f"   VERDICT: {res['verdict']}  (pre-registered expected: {expected})  {'MATCH' if match else '*** MISMATCH ***'}")
        print(f"   reason: {res['reason']}")
        for c in res["caveats"]:
            print(f"   caveat: {c}")

        results.append({
            "label": label, "kernel": kname, "expected": expected, "actual": res["verdict"], "match": match,
            "reason": res["reason"], "caveats": res["caveats"], "findings": res["findings"],
            "traffic_paths": res["traffic_paths"], "crosscheck_atomics_agree": agree_atomics, "crosscheck_div_agree": agree_div,
        })

    print(f"\n{'-'*112}\n[KNOWN-LIMITATION FORCE-CHECK, not counted in gate-quality] reverse_copy: dst[N-1-i]=src[i] "
          f"-- PROVABLY bijective (a human can show N-1-i is unique per i), so race-free WITHOUT atomics.")
    print(f"   PRE-REGISTERED PREDICTION (before running): the gate's exclusivity check requires a BARE tid Name in")
    print(f"   the index tuple; 'N-1-i' is a BinOp, not a bare Name -> predict RACE (a FALSE POSITIVE, conservative bias).")
    rev_tree = ast.parse(REVERSAL_SRC)
    rev_state, rev_const = file_state_arrays(rev_tree)
    rev_fn = find_kernels(rev_tree)[0]
    rev_res = analyze_kernel(rev_fn, rev_state, rev_const)
    fp_confirmed = rev_res["verdict"] != "SHIP" and any(f["kind"] == "RACE" for f in rev_res["findings"])
    print(f"   ACTUAL: verdict={rev_res['verdict']}, finding={[(f['array'], f['kind']) for f in rev_res['findings']]}")
    print(f"   PREDICTION {'CONFIRMED' if fp_confirmed else 'REFUTED'}: the stated conservative-bias limitation "
          f"{'is REAL (machine-forced, not just asserted in prose) -- a correct kernel would be held for human review.' if fp_confirmed else 'did NOT reproduce -- prose claim was WRONG, retracting it.'}")

    n = len(results)
    n_match = sum(r["match"] for r in results)
    false_rejects = [r for r in results if r["expected"] == "SHIP" and r["actual"] != "SHIP"]
    misses = [r for r in results if r["expected"] == "DON'T-SHIP" and r["actual"] == "SHIP"]
    crosscheck_all_agree = all(r["crosscheck_atomics_agree"] and r["crosscheck_div_agree"] for r in results)

    print("\n" + "=" * 116)
    print(f"GATE-QUALITY: {n_match}/{n} pre-registered verdicts matched. FALSE-REJECTS (expect SHIP, got other) = "
          f"{len(false_rejects)} {[r['kernel'] for r in false_rejects]}. MISSES (expect DON'T-SHIP, got SHIP) = "
          f"{len(misses)} {[r['kernel'] for r in misses]}. path-A/path-B (2-way over-determination) full agreement = {crosscheck_all_agree}")

    print("\n" + "=" * 116)
    print("STATIC vs RUNTIME boundary (named, not hand-waved):")
    print("  CAN certify a-priori (this gate, source-only, no execution):")
    print("   - determinism-by-construction: gather (exclusive-write) vs atomics vs unprotected-scatter (RACE)")
    print("   - addressing structural-vs-data-keyed (file-scope write-provenance)")
    print("   - presence of a moment-division precision-sensitivity (X = Y/Z, Z accumulated over a loop)")
    print("   - a byte-count LOWER-BOUND traffic figure (STATE-array touches x dtype-bytes), compared to an")
    print("     independently-established DAG-min where one exists (72B/voxel, D2Q9, from d_certvector_on_real_lbm.py)")
    print("  CANNOT certify a-priori (named honestly, needs a RUNTIME measurement instead):")
    print("   - ACHIEVED bandwidth/roofline-% (measured 79% in d_certvector_on_real_lbm.py -- a runtime figure;")
    print("     this gate only certifies the byte-count NUMERATOR, not the achieved GB/s)")
    print("   - actual worst-case CONTENTION/occupancy for a DATA-KEYED write (k_jac_vel: how many contacts")
    print("     can pile onto one body this frame?) -- provenance-gated per I, needs an input-distribution spec")
    print("     or a runtime histogram, not derivable from source text")
    print("   - whether a float-atomic non-determinism inside a CONVERGING iterative solver (k_jac_vel's Jacobi")
    print("     relaxation) stays within a physical tolerance -- needs a RUNTIME multi-order bit-diff/residual")
    print("     check (the same technique d_certvector_on_real_lbm.py's determinism-eye and det_accumulation_probe's")
    print("     order-shuffle test already run) -- this gate can only say 'not bit-exact', never 'close enough'")
    print("   - the exclusivity check is a conservative SUFFICIENT condition (identity-mapped tid axes), not a")
    print("     full injectivity prover: a provably-bijective but OFFSET address map (e.g. arr[N-1-i]) would be")
    print("     conservatively flagged for review even if a human could prove it race-free -- sound-direction bias")
    print("   - R1's byte-count is a SOURCE-TEXT touch count (an upper bound on hardware traffic), not the compiled/")
    print("     achieved DRAM traffic: lbm_voxel_aero_gpu.py's older `collide` textually reads f[i,j,k] TWICE in one")
    print("     expression (line 57: val = f[i,j,k] - omega*(f[i,j,k]-feq)) -> 27 reads measured vs the 9-read floor")
    print("     (144B vs 72B, 2.0x) in its fluid branch -- a real source-level redundant re-read the newer fused")
    print("     lbm_gpu_fast.py::collide_stream eliminates by caching into a vec9 register (g[k]) instead. Whether")
    print("     the CUDA/NVRTC compiler's own CSE folds the textual double-read into one hardware load anyway is a")
    print("     COMPILER fact this source-only pass cannot see -- confirming ACHIEVED traffic needs compiled-IR")
    print("     inspection or a runtime profiler (nsight-compute), not source analysis. Non-blocking PERF flag.")
    print("=" * 116)

    verdict = (n_match == n) and (len(false_rejects) == 0) and (len(misses) == 0) and crosscheck_all_agree and (anchor_pass is not False)
    print(f"\nOVERALL: {'PASS' if verdict else 'FAIL'}")

    out_path = os.path.join(HERE, "artifacts", "d_static_deployment_gate_evidence.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({"anchor": anchor, "anchor_pass": anchor_pass, "instances": results, "n": n, "n_match": n_match,
               "false_rejects": [r["kernel"] for r in false_rejects], "misses": [r["kernel"] for r in misses],
               "crosscheck_all_agree": crosscheck_all_agree,
               "known_limitation_forcecheck_reverse_copy": {"verdict": rev_res["verdict"], "findings": rev_res["findings"],
                                                             "false_positive_confirmed": fp_confirmed},
               "verdict": "PASS" if verdict else "FAIL"},
              open(out_path, "w"), indent=1, default=str)
    print(f"EVIDENCE -> {out_path}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
