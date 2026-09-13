# CPU dispatch isolates the recurrence differences

An explicit common-dispatch profile resolves the measured local/cloud differences without changing numerical source, expected numbers or tolerances. The original native-profile failure remains recorded. The controlled probe passes8/8 mechanism gates; both hosts produce exactly the same56 captured arrays when NumPy and SciPy both select Haswell code.

The local native environment selects Haswell for both OpenBLAS runtimes. The cloud native environment selects SkylakeX. Within each host, all four intervention cases use the same library binaries. SciPy's binary SHA256 is also identical across the two hosts. NumPy's binary and package version differ across hosts, but this does not prevent exact agreement under the matched dispatch for these fixtures.

Each case runs in two independent processes, with one BLAS thread and CUDA hidden. NumPy is loaded and initialized before selecting SciPy's dispatch; the actual architectures are read back from the loaded libraries, and requested selections must match. All56 arrays per worker are retained and independently checked against their hashes, dimensions and dtypes. The original frozen function is executed; endpoint terms are reconstructed with its unchanged operator order and required to equal its captured endpoint exactly.

| Cloud NumPy dispatch | Cloud SciPy dispatch | Max output error | Original endpoint | Endpoint with fixed cloud solve matrices |
|---|---|---:|---:|---:|
| SkylakeX | SkylakeX |600.667236328125|101.71659088134766|101.71659088134766|
| Haswell | SkylakeX |600.667236328125|-101.71659088134766|-101.71659088134766|
| SkylakeX | Haswell |437.3193359375|305.1497802734375|101.71659088134766|
| Haswell | Haswell |437.3193359375|0|-101.71659088134766|

SciPy dispatch changes the two triangular-solve outputs; changing only NumPy leaves those solve arrays unchanged. To isolate the later endpoint arithmetic, every case receives the same captured cloud Cmat/Umat matrices. Its S0, B, state-times-C and correction term are identical. NumPy dispatch alone changes `Umat @ B` from+2^-27 to-2^-27. The frozen cumulative-gate scaling amplifies this to the endpoint sign difference above. Both effects are therefore observed dispatch effects within fixed library builds. This does not identify a particular instruction, prove universal bit determinism, or make the expansive recurrence numerically acceptable: its original guard still refuses it.

Local/cloud Haswell/Haswell compares56/56 complete arrays exactly, including original and transplanted endpoint terms. Three refusal tests cover missing arrays, modified bytes and a reported successful worker whose actual architecture disagrees with the requested selection. Independent complete comparison reports are exact.

## Reproduce the mechanism

From the repository root in the recorded CPU environment:

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/roundoff_dispatch_v1.py
python scripts/roundoff_dispatch_compare_v1.py
python -m pytest -q tests/test_roundoff_dispatch.py
```

The probe requires threadpoolctl. It writes the current host's captures under `reports/roundoff_dispatch_v1/`; preserve the shipped captures before regenerating them. The `cloud/` subdirectory retains the independent Modal result. The audit rejects missing/modified arrays, wrong actual dispatch, source drift and unequal independent workers. The recorded environments are bounded supported x86-64/OpenBLAS combinations, not a promise about arbitrary library releases or architectures.

## Verify the explicit profile

From the existing configured Modal runner directory:

```sh
modal run run.py::run --repo 3fold-kernel-engine \
  --cmd 'OPENBLAS_CORETYPE=Haswell make verify-declared RUNTIME=modal' \
  --gpu L4 --timeout-s 1200 \
  --extra-pip 'warp-lang==1.17.0 cryptography threadpoolctl'
```

This invocation passes5/5 declared recipes in319.1seconds, including the fresh20-row signed GPU replay. All four complete CPU reports are byte-identical to the recorded local reports. An independent CPU audit reproduces the complete fetched GPU verification report exactly, and all20 fresh GPU output-record hashes equal the native-profile capture. Haswell is an explicit process environment selection; the default command and its native4/5 negative are unchanged. The recipe snapshot still contains five recipes. The two new diagnostic declarations increase the current inventory to291 rows, of which286 remain unmapped; full `make verify` is not green.

`reports/roundoff_dispatch_v1/` retains the complete intervention data and comparison. `reports/verification_haswell_v1/` retains selected fresh profile reports, capture, runtime metadata and an independent CPU audit. These tests establish numerical agreement, not a throughput benefit. Original native negatives remain under `reports/verification_modal_v1/` and `reports/roundoff_environment_v1/`. No local GPU or persistent machine setting was used.
