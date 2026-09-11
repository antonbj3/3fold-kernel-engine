# Running

1. `python -m venv .venv && .venv/bin/pip install -r requirements.txt` (add `wgpu` for the Vulkan port).
2. Every module under `src/kernel_engine/` is also a script: run it directly and it executes its own gates and prints the verdicts.
3. `pytest tests/` wraps those self-tests; the CUDA-only ones skip automatically when no CUDA device is present.
4. The allocation-law modules are CPU-only and take 5-25 s each; they write their evidence JSON next to themselves under `artifacts/`.
5. Reference evidence JSONs from earlier runs are in `examples/`; the four allocation ones there were regenerated on a machine without a GPU.
