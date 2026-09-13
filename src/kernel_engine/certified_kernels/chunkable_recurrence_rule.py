"""Fifth a-priori requirement: is a task's recurrence CHUNKABLE, and what is its byte floor?

The a-priori requirement cert reads four numbers off a task (fewest bytes that must move, input-keyed
addressing / identifiability, tolerance, write contention). A task that carries state across a sequence
adds a fifth: the STRUCTURE of the state transition, which decides whether the sequence loop has to run
step by step or can be cut into chunks.

Rule. A state update S_t = f(S_{t-1}, x_t) is CHUNKABLE when the transition is linear in the state and the
transition operators compose in closed form, i.e.

  * diagonal:                 S_t = diag(a_t) S_{t-1} + b_t          (per-channel gate / selective SSM)
  * diagonal-plus-low-rank:   S_t = S_{t-1}(I - b_t k_t k_t^T) + b_t v_t k_t^T   (delta rule, and its
                              gated variant S_t = S_{t-1} diag(g_t)(I - b_t k_t k_t^T) + b_t v_t k_t^T)
  * an explicitly associative scan (cumsum / cumprod / cummax / logcumsumexp / associative_scan).

Then an outer sequential loop over chunks, carrying only the chunk-boundary state, plus an inner loop that
is fully parallel inside the chunk (matmuls + one unit-triangular solve of chunk size) reproduces the
step-by-step result to floating-point round-off. The chunkwise form of the rank-1 delta-rule update used
here is the WY representation of Kimi Delta Attention / gated DeltaNet (arXiv 2510.26692).

A recurrence whose state map is nonlinear (tanh/sigmoid RNN cell, LSTM, GRU) does not compose in closed
form and stays SEQUENTIAL: its byte floor is the step-by-step one.

Byte floors. Step-by-step, the state is read and written once per step:
    sequential = T * 2 * |S| * dtype + T * (per-step inputs and outputs) * dtype
Chunked, the state crosses a chunk boundary T/C times instead of T times:
    chunked    = (T/C) * 2 * |S| * dtype + T * (per-step inputs and outputs) * dtype
so the state term shrinks by the chunk size C, which is what makes the task portable at all when |S| is
large (|S| = d_v * d_k for a matrix-valued state).

Input: a kernel/task source string, the same input the a-priori cert reads (KernelBench-form reference
module text) plus optional parsed shapes. Output of the selftest:
`artifacts/chunkable_recurrence_rule.json` and the gate verdicts on stdout.

Gates:
  G1 chunked == sequential for a synthetic gated-deltanet layer (per-channel diagonal gate, d=64, T=512,
     chunk 64): max abs error at fp32 round-off level, on both the outputs and the final state.
  G2 a nonlinear recurrence (h_t = tanh(W h_{t-1} + x_t)) is classed `sequential`, and the same chunked
     evaluator applied to it does NOT reproduce the sequential result - the class is discriminating, not
     a label.
  G3 the byte floors: the chunked floor's state term is C times smaller than the sequential one, and the
     suggested chunk size keeps the inner working set inside a fixed on-chip budget.

  python3 chunkable_recurrence_rule.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
import re

import numpy as np
from scipy.linalg import solve_triangular

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
DTYPE_BYTES = 4                                                                               # float32
ONCHIP_BUDGET_BYTES = 96 * 1024                                                               # per-block on-chip working-set budget the chunk must fit in
CHUNK_MIN, CHUNK_MAX = 16, 256

# ---- source-level detection of the transition structure ----
# nonlinear state map: the state passes through a saturating nonlinearity every step => no closed-form composition
NONLINEAR_RECURRENCE = re.compile(
    r"\bnn\.RNN\b|\bnn\.LSTM\b|\bnn\.GRU\b|\bnn\.RNNCell\b|\bnn\.LSTMCell\b|\bnn\.GRUCell\b"
    r"|\bF\.(?:rnn|lstm|gru)\b|VanillaRNN|\bLSTM\b|\bGRU\b"
    # an explicit cell written out: the carried state is reassigned through a saturating nonlinearity
    r"|\b(?:h|hidden|hx|state)\s*=\s*(?:self\.|torch\.|F\.)?(?:tanh|sigmoid|relu)\(")
# explicitly associative scan
ASSOCIATIVE_SCAN = re.compile(
    r"\btorch\.cumsum\(|\.cumsum\(|\btorch\.cumprod\(|\.cumprod\(|\btorch\.cummax\(|\.cummax\("
    r"|\btorch\.cummin\(|\.cummin\(|logcumsumexp|associative_scan|\bsegsum\b")
# diagonal (per-channel gated) linear state, incl. selective state-space models
DIAGONAL_LINEAR = re.compile(
    r"selective_scan|\bssd\b|Mamba|mamba|state_decay|decay_chunk|decay_states|A_cumsum"
    r"|\bS4\b|\bS6\b|state_space|\bssm\b|\bSSM\b")
# diagonal-plus-low-rank: the rank-1 delta-rule update (and its gated variant)
DPLR = re.compile(
    r"delta_rule|DeltaNet|deltanet|delta_net|gated_delta|\bWY\b|\bDPLR\b"
    r"|I\s*-\s*beta|\(1\s*-\s*beta\s*\*|beta\s*\*\s*k\b")

CHUNKABLE, SEQUENTIAL, NO_RECURRENCE = "chunkable", "sequential", "none"


def detect_recurrence(src):
    """classify a task source. Returns (class, form, evidence token).

    class is one of `chunkable` (composable transition), `sequential` (nonlinear state map) or `none`
    (no state carried across a sequence). The nonlinear test is applied first: an LSTM that happens to
    contain a cumsum elsewhere is still sequential."""
    m = NONLINEAR_RECURRENCE.search(src)
    if m:
        return SEQUENTIAL, "nonlinear state map (saturating cell)", m.group(0)
    m = DPLR.search(src)
    if m:
        return CHUNKABLE, "diagonal-plus-low-rank (delta rule, rank-1 WY form)", m.group(0)
    m = DIAGONAL_LINEAR.search(src)
    if m:
        return CHUNKABLE, "diagonal transition (per-channel gate / selective state space)", m.group(0)
    m = ASSOCIATIVE_SCAN.search(src)
    if m:
        return CHUNKABLE, "explicitly associative scan", m.group(0)
    return NO_RECURRENCE, "no state carried across a sequence", ""


def suggest_chunk_size(d_k, d_v, T=None, budget_bytes=ONCHIP_BUDGET_BYTES, dtype_bytes=DTYPE_BYTES):
    """largest power-of-two chunk C whose inner working set C*(2*d_k + 2*d_v)*dtype + C*C*dtype fits the
    on-chip budget (the C x C term is the within-chunk triangular system), clamped to [16, 256] and to T."""
    c = CHUNK_MIN
    while c * 2 <= CHUNK_MAX:
        nxt = c * 2
        if T is not None and nxt > T:
            break
        if (nxt * (2 * d_k + 2 * d_v) + nxt * nxt) * dtype_bytes > budget_bytes:
            break
        c = nxt
    if T is not None:
        c = min(c, T)
    return c


def byte_floors(T, d_k, d_v, chunk, dtype_bytes=DTYPE_BYTES, per_step_io_elems=None):
    """sequential and chunked minimum-traffic floors for a sequence recurrence with a d_v x d_k state.

    per_step_io_elems defaults to the delta-rule layer's own per-step I/O (q, k, g read, v read, o write)."""
    state = d_v * d_k
    io_elems = per_step_io_elems if per_step_io_elems is not None else (3 * d_k + 2 * d_v)
    io_bytes = T * io_elems * dtype_bytes
    seq = T * 2 * state * dtype_bytes + io_bytes
    chunked = (T + chunk - 1) // chunk * 2 * state * dtype_bytes + io_bytes
    return {"state_elems": state, "sequential_bytes": int(seq), "chunked_bytes": int(chunked),
            "state_traffic_ratio": float(T / ((T + chunk - 1) // chunk)),
            "total_ratio": float(seq / chunked), "io_bytes": int(io_bytes)}


SCAN_FORM = "explicitly associative scan"


def _module_ints(src):
    """module-level int assignments and 1-tuples of the task file (seq_length = 128, input_shape = (4096,))."""
    ns = {}
    for line in src.splitlines():
        m = re.match(r"^([A-Za-z_]\w*)\s*=\s*([0-9][0-9\s\*\+\-/]*)\s*(#.*)?$", line)
        if m:
            try:
                ns[m.group(1)] = int(eval(m.group(2), {"__builtins__": {}}, ns))
            except Exception:
                pass
            continue
        m = re.match(r"^([A-Za-z_]\w*)\s*=\s*\(\s*([0-9]+)\s*,?\s*\)\s*(#.*)?$", line)
        if m:
            ns[m.group(1)] = int(m.group(2))
    return ns


def _shape_hints(src):
    """(sequence length, state width) read off the task's own module-level constants; (None, None) if absent."""
    ns = _module_ints(src)
    T = next((ns[n] for n in ("seq_length", "sequence_length", "seq_len", "input_shape", "length") if n in ns), None)
    d = next((ns[n] for n in ("hidden_size", "d_state", "d_head", "d_model", "features") if n in ns), None)
    return T, d


def recurrence_requirement(src, shapes=None, T=None, d_k=None, d_v=None):
    """the fifth a-priori requirement for one task: transition class, chunk size and both byte floors.

    Shapes are used only to size the floors; when they are unknown the class and form are still reported
    and the floors are None (the class is read off the transition structure, not off the shapes)."""
    klass, form, evidence = detect_recurrence(src)
    req = {"recurrence_class": klass, "transition_form": form, "evidence": evidence,
           "chunk_size": None, "floors": None}
    if klass == NO_RECURRENCE:
        return req
    # the task's own named constants first (seq_length / hidden_size / d_state name the sequence and the state
    # explicitly); positional tensor shapes only as a fallback, where the last two dims are taken as (T, width)
    if T is None or d_k is None:
        hT, hd = _shape_hints(src)
        T = T if T is not None else hT
        d_k = d_k if d_k is not None else hd
    if (T is None or d_k is None) and shapes:
        longest = max(shapes, key=len)
        if len(longest) >= 2:
            T = T if T is not None else longest[-2]
            d_k = d_k if d_k is not None else longest[-1]
    if T and d_k is None and form == SCAN_FORM:
        d_k = 1                                                                               # a scalar scan carries one state element per lane
    if T and d_k:
        d_v = d_v or d_k
        chunk = suggest_chunk_size(d_k, d_v, T) if klass == CHUNKABLE else 1
        io = 2 * d_v if form == SCAN_FORM else None                                           # a scan reads one element and writes one per step
        req["chunk_size"] = chunk
        req["floors"] = byte_floors(T, d_k, d_v, chunk, per_step_io_elems=io)
    return req


# ---------------------------------------------------------------------------------------------------
# reference implementations: a gated delta-rule layer, step by step and chunkwise (WY form)
# ---------------------------------------------------------------------------------------------------
def gated_delta_sequential(q, k, v, g, beta, S0=None):
    """step-by-step reference. S_t = S_{t-1} diag(g_t) (I - beta_t k_t k_t^T) + beta_t v_t k_t^T,
    o_t = S_t q_t. q,k: (T,d_k); v: (T,d_v); g: (T,d_k) per-channel gate; beta: (T,).
    Returns (O (T,d_v), S_final (d_v,d_k))."""
    T, d_k = k.shape
    d_v = v.shape[1]
    S = np.zeros((d_v, d_k), dtype=np.float32) if S0 is None else S0.astype(np.float32).copy()
    O = np.zeros((T, d_v), dtype=np.float32)
    for t in range(T):
        S = S * g[t][None, :]
        S = S - beta[t] * np.outer(S @ k[t], k[t])
        S = S + beta[t] * np.outer(v[t], k[t])
        O[t] = S @ q[t]
    return O, S


def _chunk_forward(q, k, v, g, beta, S0):
    """one chunk, fully parallel inside: cumulative-gate change of variables + WY (unit-triangular) solve.

    With gamma_t = prod_{s<=t} g_s (within the chunk) and S~_t = S_t diag(gamma_t)^-1 the gated update
    becomes the ungated rank-1 form S~_t = S~_{t-1}(I - beta_t a_t b_t^T) + beta_t v_t b_t^T with
    a_t = gamma_t * k_t and b_t = k_t / gamma_t, whose product of transitions is I - Cmat B and whose
    forced part is Umat B, both obtained from the SAME unit-triangular system (arXiv 2510.26692)."""
    C = k.shape[0]
    gamma = np.cumprod(g, axis=0).astype(np.float32)                                          # (C, d_k)
    A = (gamma * k).astype(np.float32)                                                        # rows a_t
    B = (k / gamma).astype(np.float32)                                                        # rows b_t
    Qt = (gamma * q).astype(np.float32)                                                       # rows gamma_t * q_t
    M = (B @ A.T).astype(np.float32)                                                          # M[s,t] = b_s . a_t
    L = (np.tril(beta[:, None] * M.T, -1)).astype(np.float32)                                 # strictly lower, = beta_t * M[s,t], s<t
    Ident = np.eye(C, dtype=np.float32)
    rhs_a = (beta[:, None] * A).astype(np.float32)                                            # (C, d_k) = (diag(beta) A)
    rhs_v = (beta[:, None] * v).astype(np.float32)                                            # (C, d_v)
    Cmat = solve_triangular(Ident + L, rhs_a, lower=True, unit_diagonal=True).T.astype(np.float32)   # (d_k, C)
    Umat = solve_triangular(Ident + L, rhs_v, lower=True, unit_diagonal=True).T.astype(np.float32)   # (d_v, C)
    Z = (B @ Qt.T).astype(np.float32)                                                         # Z[s,t] = b_s . (gamma_t q_t)
    Zm = np.triu(Z, 0).astype(np.float32)                                                     # keep s <= t (output after the t-th update)
    O = (S0 @ Qt.T - (S0 @ Cmat) @ Zm + Umat @ Zm).T.astype(np.float32)                       # (C, d_v)
    S_end = ((S0 - (S0 @ Cmat) @ B + Umat @ B) * gamma[-1][None, :]).astype(np.float32)
    return O, S_end


def gated_delta_chunked(q, k, v, g, beta, chunk, S0=None):
    """outer sequential loop over chunks carrying the state, inner loop fully parallel within the chunk."""
    T, d_k = k.shape
    d_v = v.shape[1]
    S = np.zeros((d_v, d_k), dtype=np.float32) if S0 is None else S0.astype(np.float32).copy()
    O = np.zeros((T, d_v), dtype=np.float32)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        O[s:e], S = _chunk_forward(q[s:e], k[s:e], v[s:e], g[s:e], beta[s:e], S)
    return O, S


def nonlinear_sequential(x, W, h0=None):
    """h_t = tanh(W h_{t-1} + x_t): the non-composable control."""
    T, d = x.shape
    h = np.zeros(d, dtype=np.float32) if h0 is None else h0.astype(np.float32).copy()
    H = np.zeros((T, d), dtype=np.float32)
    for t in range(T):
        h = np.tanh(W @ h + x[t]).astype(np.float32)
        H[t] = h
    return H


def nonlinear_chunked_attempt(x, W, chunk):
    """the same task evaluated in chunks under the (wrong) assumption that the transition composes:
    each chunk restarts from the previous chunk's boundary state but treats the map inside the chunk as
    linear. Used only to show that the `sequential` class is discriminating."""
    T, d = x.shape
    h = np.zeros(d, dtype=np.float32)
    H = np.zeros((T, d), dtype=np.float32)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        acc = h.copy()
        for t in range(s, e):
            acc = (W @ acc + x[t]).astype(np.float32)                                         # linear composition inside the chunk
            H[t] = np.tanh(acc)
        h = H[e - 1]
    return H


# ---- KernelBench-form stand-ins used by the selftest's classifier check ----
STAND_IN_SOURCES = {
    "gated_deltanet": "class Model(nn.Module):\n    def forward(self, q, k, v):\n        return gated_delta_rule(q, k, v, beta)\n",
    "mamba2_ssd": "    def segsum(self, x):\n        x_cumsum = torch.cumsum(x, dim=-1)\n        A_cumsum = torch.cumsum(A_blocks, dim=-1)\n",
    "cumsum": "class Model(nn.Module):\n    def forward(self, x):\n        return torch.cumsum(x, dim=self.dim)\n",
    "lstm": "self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)\n",
    "vanilla_rnn": "        self.hidden = torch.tanh(self.i2h(combined))\n        return self.h2o(self.hidden)\n",
    "matmul": "class Model(nn.Module):\n    def forward(self, A, B):\n        return torch.matmul(A, B)\n",
}
EXPECTED_CLASS = {"gated_deltanet": CHUNKABLE, "mamba2_ssd": CHUNKABLE, "cumsum": CHUNKABLE,
                  "lstm": SEQUENTIAL, "vanilla_rnn": SEQUENTIAL, "matmul": NO_RECURRENCE}


def main():
    print("=" * 122)
    print("chunkable-recurrence rule: the fifth a-priori requirement (transition structure -> chunkable vs sequential)")
    print("=" * 122)
    rng = np.random.default_rng(0)
    T, d, chunk = 512, 64, 64

    # ---- G1: synthetic gated-deltanet layer, chunked vs step-by-step ----
    q = rng.standard_normal((T, d)).astype(np.float32)
    k = rng.standard_normal((T, d)).astype(np.float32)
    k = (k / np.linalg.norm(k, axis=1, keepdims=True)).astype(np.float32)                     # delta-rule keys are unit-norm
    v = rng.standard_normal((T, d)).astype(np.float32)
    g = (0.95 + 0.05 * rng.random((T, d))).astype(np.float32)                                 # per-channel diagonal gate in [0.95, 1)
    beta = rng.uniform(0.1, 0.9, size=T).astype(np.float32)

    O_seq, S_seq = gated_delta_sequential(q, k, v, g, beta)
    O_chk, S_chk = gated_delta_chunked(q, k, v, g, beta, chunk)
    err_o = float(np.max(np.abs(O_seq - O_chk)))
    err_s = float(np.max(np.abs(S_seq - S_chk)))
    scale_o = float(np.max(np.abs(O_seq)))
    scale_s = float(np.max(np.abs(S_seq)))
    eps = float(np.finfo(np.float32).eps)
    roundoff_o = eps * scale_o * np.sqrt(chunk * d)                                           # fp32 round-off budget for the chunk's matmul depth
    roundoff_s = eps * scale_s * np.sqrt(T * d)
    g1 = bool(err_o <= roundoff_o and err_s <= roundoff_s)
    print(f"\n[G1] SYNTHETIC GATED-DELTANET (per-channel diagonal gate, d={d}, T={T}, chunk={chunk}): chunked vs sequential recurrence")
    print(f"     outputs:     max abs err {err_o:.3e}   (|O|max {scale_o:.3f}, fp32 round-off budget eps*|O|*sqrt(C*d) = {roundoff_o:.3e})")
    print(f"     final state: max abs err {err_s:.3e}   (|S|max {scale_s:.3f}, fp32 round-off budget eps*|S|*sqrt(T*d) = {roundoff_s:.3e})")
    print(f"     ⇒ the chunkwise WY form reproduces the step-by-step recurrence at fp32 round-off -> {g1}")

    # ---- G2: a nonlinear recurrence must be classed sequential, and must NOT chunk ----
    W = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    x = rng.standard_normal((T, d)).astype(np.float32)
    H_seq = nonlinear_sequential(x, W)
    H_chk = nonlinear_chunked_attempt(x, W, chunk)
    err_nl = float(np.max(np.abs(H_seq - H_chk)))
    nl_class, nl_form, nl_ev = detect_recurrence("h = torch.tanh(self.W @ h + x[t])")
    classes = {name: detect_recurrence(src)[0] for name, src in STAND_IN_SOURCES.items()}
    classifier_ok = classes == EXPECTED_CLASS
    g2 = bool(nl_class == SEQUENTIAL and err_nl > 1e-2 and classifier_ok)
    print(f"\n[G2] NONLINEAR RECURRENCE h_t = tanh(W h_(t-1) + x_t) (known-bad control for the rule):")
    print(f"     class = '{nl_class}' (evidence {nl_ev!r}); chunking it anyway gives max abs err {err_nl:.3e} (order 1, not round-off)")
    print(f"     classifier on KernelBench-form stand-ins: {classes}")
    print(f"     ⇒ the class is discriminating: nonlinear stays SEQUENTIAL and does not survive chunking -> {g2}")

    # ---- G3: byte floors and chunk size ----
    fl = byte_floors(T, d, d, chunk)
    suggested = suggest_chunk_size(d, d, T)
    working_set = (suggested * (2 * d + 2 * d) + suggested * suggested) * DTYPE_BYTES
    g3 = bool(abs(fl["state_traffic_ratio"] - chunk) < 1e-6 and fl["total_ratio"] > 1.0
              and working_set <= ONCHIP_BUDGET_BYTES and CHUNK_MIN <= suggested <= CHUNK_MAX)
    print(f"\n[G3] BYTE FLOORS (T={T}, state {d}x{d} = {fl['state_elems']} elems, fp32):")
    print(f"     sequential floor {fl['sequential_bytes']:,} B   chunked floor {fl['chunked_bytes']:,} B   total ratio {fl['total_ratio']:.2f}x")
    print(f"     state traffic shrinks by exactly the chunk size ({fl['state_traffic_ratio']:.0f}x); per-step I/O ({fl['io_bytes']:,} B) is the same in both")
    print(f"     suggested chunk {suggested} keeps the inner working set at {working_set/1024:.1f} KiB <= {ONCHIP_BUDGET_BYTES/1024:.0f} KiB on-chip budget -> {g3}")

    ok = g1 and g2 and g3
    os.makedirs(ARTIFACTS, exist_ok=True)
    json.dump({
        "module": "chunkable_recurrence_rule",
        "claim": ("A fifth a-priori requirement: the STRUCTURE of a task's state transition decides whether "
                  "its sequence loop is chunkable. A diagonal, diagonal-plus-low-rank (rank-1 delta rule, "
                  "with or without a diagonal gate) or explicitly associative transition composes in closed "
                  "form, so an outer loop over chunks plus a fully parallel inner loop reproduces the "
                  "step-by-step recurrence to fp32 round-off and the task's byte floor is the chunked one "
                  "(state traffic divided by the chunk size). A nonlinear state map (tanh/LSTM/GRU cell) "
                  "does not compose and keeps the sequential floor."),
        "G1_equivalence": {"T": T, "d": d, "chunk": chunk, "max_abs_err_outputs": err_o,
                           "max_abs_err_final_state": err_s, "output_scale": scale_o, "state_scale": scale_s,
                           "fp32_roundoff_budget_outputs": roundoff_o, "fp32_roundoff_budget_state": roundoff_s,
                           "pass": g1},
        "G2_nonlinear_control": {"class": nl_class, "evidence": nl_ev, "chunked_attempt_max_abs_err": err_nl,
                                 "classifier_on_stand_ins": classes, "pass": g2},
        "G3_byte_floors": dict(fl, suggested_chunk=suggested, inner_working_set_bytes=int(working_set),
                               onchip_budget_bytes=ONCHIP_BUDGET_BYTES, pass_=g3),
        "gates": {"G1_chunked_equals_sequential_at_roundoff": g1, "G2_nonlinear_stays_sequential": g2,
                  "G3_byte_floors_and_chunk_size": g3, "verdict": "PASS" if ok else "FAIL"},
        "honest_scope": ("MEASURED: the equivalence, the nonlinear control and the floors are computed here "
                         "in float32 on a synthetic gated delta-rule layer (SYNTHETIC INPUT); no GPU, no "
                         "measured kernel time. The byte floors are analytic minimum-traffic counts (state "
                         "term + per-step I/O), not measured DRAM traffic. The source-level classifier is "
                         "token-based: it reads the transition structure off the task text and can only see "
                         "recurrences written with the listed constructs."),
        "provenance": ("Chunkwise (WY) form of gated rank-1 delta-rule updates: arXiv 2510.26692. Synthetic "
                       "inputs only; CPU, OMP_NUM_THREADS=4."),
    }, open(os.path.join(ARTIFACTS, "chunkable_recurrence_rule.json"), "w"), indent=1)
    print("=" * 122)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} — G1(chunked==sequential at fp32 round-off)={g1} G2(nonlinear stays sequential)={g2} G3(byte floors + chunk size)={g3}")
    print("EVIDENCE -> artifacts/chunkable_recurrence_rule.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
