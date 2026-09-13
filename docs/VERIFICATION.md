# Explicit verification recipes

The full repository gate is OWN-GATE-FAIL: 5/289 declared VERIFIED-FRESH rows have explicit recipes; 284 remain unmapped. This is partial queue item27. All declared source members resolve and there are no duplicate declarations. A main guard identifies only a candidate, never numerical success.

Run from the repository root with its installed environment:

```sh
make verify-inventory PYTHON=python
make verify-declared PYTHON=python
make verify PYTHON=python
```

The inventory exits1 for incomplete coverage. The selected target executes CPU recipes by default: 4 declarations passed in each of two independent runs, with exact generated numerical report hashes. Full `make verify` refuses incomplete coverage before running children (script exit1, make exit2). Ten focused tests pass:

```sh
python -m pytest -q tests/test_verification_inventory.py tests/test_verify_declared.py
```

`docs/VERIFY_RECIPES.json` binds each declaration to exact source hashes and the decisive RUNNING table-note hash. Recipes specify repository-relative working directory, argv, runtime and expected exit code plus typed JSON-pointer equality, collection lengths or exact pytest pass count. Changing a source or status note invalidates its recipe. These pins cover declared source members, not the full transitive dependency graph or environment. Numerical modules retain their own input and gate checks.

The runner removes each expected report before execution, limits each child to900seconds and records checks, exit code, report SHA256 and diagnostic output hashes. CPU children hide CUDA devices and use one BLAS/OpenMP thread. Successful process exit alone does not replace declared evidence checks. No tolerances are widened.

`RUNTIME=modal` selects both CPU and Modal recipes. Run that only through the existing external Modal runner with the required CUDA environment; the label itself does not provision a GPU or acquire a local lock. This checkpoint used no GPU. Kernel's declared cloud replay was not rerun for this CPU checkpoint; its prior signed replay evidence remains separately recorded.

Evidence files:

- `reports/verification_inventory_v1.json`: source inventory, explicit coverage failure, no numerical execution.
- `reports/verification_complete_v1.json`: strict full-verification refusal.
- `reports/declared_verification_v1.json`: fresh selected recipe checks and hashes.
- `reports/declared_verification_repeat_v1.json`: independent repeat equality of numerical evidence; pytest timing text is excluded.

The selected result's VERIFIED-FRESH status applies only to its listed recipes. It does not promote unmapped rows, hardware not exercised here, or failed original physics models. Completing item27 requires explicit recipes and fresh evidence for every remaining declaration, including native/group/library rows.

Recipes may declare `pythonpath` as a list of existing repository directories. The runner resolves these to absolute paths before spawning workers, so temporary working directories preserve package imports. Outside paths, files and missing directories are rejected.

The added recurrence stability recipe checks all nine original observer gates and12 cases. Expansive transitions remain refused, including the existing output-budget violation; this is a bounded synthetic CPU screen, not a general stability certificate.

Identical argv, working directory, runtime, explicit Python import paths and report destination share a single child execution within one verifier invocation. All selected declaration pins are validated before any child starts. Each row retains its own expected gates and values, and records `executed_by`, `reused_execution` and the fresh report hash. A changed report invalidates reuse. The cache is empty on every new invocation; prior reports never qualify. Failed shared children remain failed on every row expecting success.

The legacy roundoff guard recipe uses `scripts/verify_roundoff_artifact_v1.py` to remove the old JSON, run the unchanged artifact producer and copy its fresh receipt into reports. Its7 original gates, two8-case legs and measured error437.3193359375 are required. The original JSON/NPZ artifacts remain byte-identical. Four CPU declarations use4 child commands; the remaining declared recipe requires Modal and was not rerun here.

The floating-point catalogue recipe re-audits the saved four-device matrix on CPU:3/3 gates,20rows,120comparisons and an empty observed-difference list. This is fresh analysis of archived device evidence, not fresh GPU execution or a universal portability claim. Its existing narrative status now has an explicit inventory table declaration.
