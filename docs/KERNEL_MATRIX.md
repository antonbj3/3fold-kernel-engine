# Kernel-family cloud determinism matrix

The matrix covers eight existing selftests, ten fixed-point reduction sites and
two sizes of the native/Warp LBM stream export. It uses Warp 1.17.0 on Modal L4,
A10G and H100. It does not include other repositories or the local RTX 5070.

Each GPU capture runs the existing selftests in two independent processes,
records every readback and tape primal/gradient array, and preserves their own
numerical gates. Reduction capsules include actual int64 values and input-array
hashes. A zero difference within one GPU is insufficient for cross-GPU identity.
The export uses the unchanged C host and generated CUDA body, compiled for each
GPU's actual architecture. Its 10% bandwidth criterion is separately reported;
passing numerical determinism does not imply passing that timing criterion.

To capture using the configured external Modal runner, invoke its `run.py::run`
entrypoint for this repository with this command on each GPU, sequentially:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python src/kernel_engine/certified_kernels/kernel_matrix_capture_v1.py
```

The per-GPU outputs are `reports/kernel_matrix_v1/capture.json.gz` and
`summary.json`. Keep them in separate GPU directories. The gzip stores complete
records; its uncompressed SHA-256 is checked against the summary before auditing.
The auditor performs no GPU work:

```sh
python src/kernel_engine/certified_kernels/kernel_matrix_report_v1.py \
  reports/kernel_matrix_v1/l4 \
  reports/kernel_matrix_v1/a10g \
  reports/kernel_matrix_v1/h100
```

It writes `matrix.json` and `matrix.md`. Acceptance requires all 20 rows, all
three expected GPUs, matching source/version identities, all own numerical
gates, and exact observations across all three pairwise GPU comparisons.
The expected observation counts are frozen explicitly; truncated matching
records cannot silently pass. Numeric int64 differences are computed from stored
values. For other arrays, a differing hash names the first differing record and
leaves numeric delta unmeasured until a raw-array replay. Equal full-byte hashes
are reported as observed zero delta.

For a single sequential capture-and-audit command, set `MODAL_RUNNER_ROOT` to the
directory of the already configured external runner:

```sh
python scripts/kernel_matrix_modal_v1.py
```

The runner preserves dated capsules and matrices under
`reports/kernel_matrix_v1/runs/`. It checks numerical reports rather than treating
the transport's exit code alone as evidence of success. This command can be
invoked by an existing nightly scheduler; no schedule is installed by this work.
The external runner provides GPU isolation, dependencies and account credentials.

See `docs/RUNNING.md` for retained measured results and scope limitations.
