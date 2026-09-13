# Signed kernel observation certificates

Twenty independently verifiable JSON files bind the existing four-device matrix
rows to pinned source and input provenance, per-device output hashes, Warp/driver
versions and passed numerical gates. These are eight selftests, ten reduction
sites and two export sizes, not twenty distinct numerical modules. All attestations
retain the finite-fixture scope of the underlying measurements.

Each envelope signs canonical compact, sorted UTF-8 JSON with Ed25519 via OpenSSL.
The private runner key is stored outside the repository and is never uploaded to
Modal. The shipped public-key PEM SHA-256 is:

`f6e84af329dc5a9d38776715bbef90d947d24964d41138d66970bec2e80bab9d`

This is a runner assertion, not an independent laboratory or hardware attestation.
Obtain the expected public key or fingerprint through a trusted channel. Replacing
both the certificate and its colocated key defeats trust based solely on that
folder. The verifier therefore requires an explicit trusted-key argument and does
not automatically trust a key embedded in a certificate.

```sh
python scripts/kernel_certificates_v1.py verify --trusted-key TRUSTED_PUBLIC_KEY.pem
```

Default verification checks all signatures, exact inventory, source manifest,
four-device coverage, numerical gate status and agreement with the complete
retained capsules. `--signature-only` omits the capsule comparison but still checks
the inventory, signed claims and source manifest; it does not rerun numerical work.
Only NumPy and an OpenSSL executable are required for this CPU-only verification.

To replay a selected export row through the configured Modal runner, use this
repository command on L4 with a600-second bound:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/kernel_certificates_v1.py replay \
  --trusted-key reports/kernel_certificates_v1/public_key.pem --row export:512
```

First pin the shipped public key independently. Replay checks current frozen
source files and Warp version, then runs the original two independent native/Warp
export legs with their exact CPU-reference and repeat gates. Both original export
sizes execute, but only the selected row is compared to its certificate. The fresh
report records the observed driver; driver equality is not an acceptance gate.
Replay currently supports the two export rows only. It does not claim a fresh
rerun of all twenty rows.

Input identity is deliberately typed. Selftests hash their pinned deterministic
fixture definition and source manifest; their old capsules did not independently
record every input array. Reduction sites hash the recorded array-input manifest;
scalar parameters and dimensions remain bound through source provenance. Export
rows record the actual input-file byte hash. Output hashes cover the canonical
complete observation-record sequence, including underlying full-array hashes;
int64 records additionally contain the actual output values.

The issuer can reproduce identical certificate bytes using the same external key:

```sh
python scripts/kernel_certificates_v1.py build --private-key PRIVATE_KEY.pem
```

Signatures are deterministic. No timestamp or performance claim is smuggled into
the numerical payload. The source manifest binds original numerical files; the
issuer/verifier implementation is recorded separately in `runner_sources.json`.

| Certified matrix row | Status | Certificate |
|---|---|---|
| export:1024 | VERIFIED-FRESH | [00.json](../reports/kernel_certificates_v1/00.json) |
| export:512 | VERIFIED-FRESH | [01.json](../reports/kernel_certificates_v1/01.json) |
| int64:lbm/differentiable_flow_control:probe_ux | VERIFIED-FRESH | [02.json](../reports/kernel_certificates_v1/02.json) |
| int64:lbm/differentiable_flow_control:track_loss | VERIFIED-FRESH | [03.json](../reports/kernel_certificates_v1/03.json) |
| int64:lbm/differentiable_fsi_chain:drag_force | VERIFIED-FRESH | [04.json](../reports/kernel_certificates_v1/04.json) |
| int64:lbm/differentiable_lbm_probe:objective | VERIFIED-FRESH | [05.json](../reports/kernel_certificates_v1/05.json) |
| int64:lbm/lbm3d_immersed_boundary:spread | VERIFIED-FRESH | [06.json](../reports/kernel_certificates_v1/06.json) |
| int64:wave_fdtd/diff_wave_3d:energy | VERIFIED-FRESH | [07.json](../reports/kernel_certificates_v1/07.json) |
| int64:wave_fdtd/diff_wave_substrate:focus_loss | VERIFIED-FRESH | [08.json](../reports/kernel_certificates_v1/08.json) |
| int64:wave_fdtd/xray_3d_dda:backproject | VERIFIED-FRESH | [09.json](../reports/kernel_certificates_v1/09.json) |
| int64:wave_fdtd/xray_tomography_sigma:sensitivity | VERIFIED-FRESH | [10.json](../reports/kernel_certificates_v1/10.json) |
| int64:wave_fdtd/xray_tomography_sigma:sq_resid | VERIFIED-FRESH | [11.json](../reports/kernel_certificates_v1/11.json) |
| selftest:lbm/differentiable_flow_control | VERIFIED-FRESH | [12.json](../reports/kernel_certificates_v1/12.json) |
| selftest:lbm/differentiable_fsi_chain | VERIFIED-FRESH | [13.json](../reports/kernel_certificates_v1/13.json) |
| selftest:lbm/differentiable_lbm_probe | VERIFIED-FRESH | [14.json](../reports/kernel_certificates_v1/14.json) |
| selftest:lbm/lbm3d_immersed_boundary | VERIFIED-FRESH | [15.json](../reports/kernel_certificates_v1/15.json) |
| selftest:wave_fdtd/diff_wave_3d | VERIFIED-FRESH | [16.json](../reports/kernel_certificates_v1/16.json) |
| selftest:wave_fdtd/diff_wave_substrate | VERIFIED-FRESH | [17.json](../reports/kernel_certificates_v1/17.json) |
| selftest:wave_fdtd/xray_3d_dda | VERIFIED-FRESH | [18.json](../reports/kernel_certificates_v1/18.json) |
| selftest:wave_fdtd/xray_tomography_sigma | VERIFIED-FRESH | [19.json](../reports/kernel_certificates_v1/19.json) |

## Complete fresh replay

The separate `kernel_certificate_replay_v2.py` re-executes all twenty signed rows
on one of the recorded GPU architectures, instead of replaying only an export row.
Use the configured Modal runner on L4 with a2000-second bound and this command:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/kernel_certificate_replay_v2.py \
  --trusted-key reports/kernel_certificates_v1/public_key.pem
```

Pin the public key as described above before using the bundled path. The new
runner first verifies all signatures against the retained four-device capsules
and checks every original numerical source file. It then calls the frozen complete
capture entrypoint: sixteen independent selftest processes, all ten actual int64
sites with two launches each, and the two original independent native/Warp export
legs at both sizes. Fresh observation-record digests and typed input identities
must match their signed device records, with matching Warp version and architecture.
The actual driver is retained even when it differs from the earlier observation.

The output directory `reports/kernel_certificate_replay_v2/` retains the full fresh
capsule, its checked summary, capture stdout/stderr and `verification.json` with
per-row acceptance gates. Original certificates and cloud/local capsules are not
rewritten. Source files added since the original snapshot are listed separately;
every file in the original signed source manifest must still match exactly.

A reader can independently rebuild the verification report from the fetched
capsule without opening a GPU context:

```sh
CUDA_VISIBLE_DEVICES='' python scripts/kernel_certificate_replay_v2.py \
  --trusted-key reports/kernel_certificates_v1/public_key.pem \
  --capture reports/kernel_certificate_replay_v2 \
  --output /tmp/kernel-certificate-audit
```

This command audits the supplied capture; it does not prove when or where that
capture was generated. The recorded Modal execution supplies the fresh-run evidence.
The twenty certificate rows remain a bounded subset of this repository's modules;
this command is not an all-repository `make verify` implementation or a universal
hardware-determinism guarantee.

A new invocation removes its prior verification result before checking signatures,
so a failed rerun cannot leave a stale successful report. Capture logs and numerical
evidence remain available for diagnosis. This failure-path safeguard was added
after the measured Modal upload; the exact executed source is retained alongside
the fresh capture. The final verifier independently audits that same fresh capsule
on CPU, with identical numerical results.
