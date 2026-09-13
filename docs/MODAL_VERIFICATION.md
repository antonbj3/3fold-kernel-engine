# Fresh Modal verifier and CPU environment boundary

The five declared recipes from commit95b383e were executed on two fresh Modal GPUs through the existing runner. Both selected runs are OWN-GATE-FAIL4/5 (make exit2). The full repository gate is also incomplete; this checkpoint does not promote its284 unmapped declarations.

| Fresh environment | Command time | Signed GPU rows | Selected recipes | Exact CPU witness |
|---|---:|---:|---:|---:|
| L4, Warp1.17.0 |321.9s|20/20 pass|4/5|600.667236328125|
| A10, Warp1.17.0 |192.2s|20/20 pass|4/5|600.667236328125|
| Recorded local CPU | not a GPU run | not run here |4/4 CPU pass|437.3193359375|

Times are complete command durations, not GPU throughput comparisons. Both fresh GPU captures match all20 signed device records and each other exactly. Independent CPU audits of each fetched capsule reproduce its complete verification report byte-for-byte. Original signatures, sources and tolerances remain unchanged.

The only failed recipe check is the local exact-number pin for the CPU recurrence guard. All seven original guard gates pass on both cloud hosts, including refusal of expansive transitions and retention of a stable roundoff failure. Two of four complete CPU reports match locally/cloud; the stability and roundoff observers differ. In all12 stability cases, input and sequential hashes match but chunked output/state hashes differ. This includes the three gain0.99 cases that still pass their original numerical budgets. These are CPU observations, outside the existing certified fixed-order GPU catalogue; its empty difference list is not a universal portability claim.

## Locate the arithmetic difference

A separate CPU observer captures actual locals from the frozen `_chunk_forward` function at return. It does not replace the arithmetic. It records21 input/intermediate arrays plus sequential output/state. Both environments pass5/5 capture controls: complete stages, exact independent trace repeats, equality with an uninstrumented call, unchanged inputs and the original expansive refusal.

The scalar gate1.2/chunk128 witness has identical arrays through `rhs_v`. The first differing stage is `Cmat`, the first unit-triangular solve result:

| Stage | Differing elements | First index | Local value | Cloud value |
|---|---:|---|---:|---:|
| Cmat |80|[0,36]|0.36717987060546875|0.3671722412109375|
| Umat |96|[0,32]|-1.0441304445266724|-1.0441302061080933|
| O |94|[33,0]|7.695234298706055|7.695235252380371|
| S_end |1|[0,0]|0|101.71659088134766|

Transplanting only the captured cloud Cmat/Umat into the unchanged function on the local host reproduces the entire cloud output array exactly. The local-injection control also matches its original full outputs. Endpoint transfer still fails: local replay produces-101.71659088134766 against cloud+101.71659088134766. Thus the solve outputs explain the output-array difference in this witness, while endpoint arithmetic has an additional environment dependence. The comparison remains OWN-GATE-FAIL4/6; both negative checks are retained. Its two independent complete reports are identical.

Local environment: Python3.13.11, NumPy2.5.3 with scipy-openblas0.3.34.106.0, SciPy1.18.1 with scipy-openblas0.3.31.dev. Cloud CPU observer: Python3.12.1, NumPy2.5.2 with scipy-openblas0.3.34.0.0, SciPy1.18.1 with scipy-openblas0.3.31.dev. Python, wheel builds and CPU dispatch were not independently varied. This does not identify one library version, FMA instruction or unique root cause, nor does this scalar probe localize every stability-case difference.

## Reproduce and inspect

From the existing configured Modal runner directory, the exact selected command was:

```sh
modal run run.py::run --repo 3fold-kernel-engine \
  --cmd 'make verify-declared RUNTIME=modal' --gpu L4 \
  --timeout-s 1200 --extra-pip 'warp-lang==1.17.0 cryptography'
```

Use A10G for the second device. Both retained runs exit2. Keep the trusted public-key pin described in KERNEL_CERTIFICATES.md; the default verifier recipe requires that explicit key.

Capture the CPU recurrence intermediates from the repository root:

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/roundoff_environment_v1.py
```

The capture writes the current host's arrays under `reports/roundoff_environment_v1`; preserve the shipped local snapshot before generating a replacement. The shipped cloud snapshot lives in its `cloud/` subdirectory. Compare in the recorded local environment:

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/roundoff_environment_compare_v1.py
```

Expected exit1 retains the endpoint/whole-environment differences. The report records the replay environment as well as both capture environments; a different replay host can change transplantation results. Four tests verify restoration of tracing/solver hooks and refusal of corrupted arrays/source metadata:

```sh
python -m pytest -q tests/test_roundoff_environment.py
```

`reports/verification_modal_v1/` contains the exact recipe snapshot, local CPU receipts, selected fresh reports/captures per device, original run metadata and independent CPU audits. Only selected fresh artifacts are copied: the generic runner also downloads old repository reports, which are not fresh evidence for this run. `reports/roundoff_environment_v1/` retains complete local/cloud arrays, hashes, versions, comparison and repeat receipt. All three Modal jobs ended and their results were fetched. No local GPU was used and nothing was pushed.
