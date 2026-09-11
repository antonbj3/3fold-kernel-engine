# Renames and provenance

Code identifiers, file names, function names, JSON keys and CLI flags are **unchanged** from the
internal platform, so files can be copied in either direction without edits. Only prose
(docstrings, comments, printed text, JSON string values) was translated and scrubbed.

## Jargon table (internal term -> public term, prose only)

| internal | public |
| --- | --- |
| varv | round |
| kernelvarv | kernel variant round |
| grind | gate |
| korrekthetsgrind / benchmarkgrind / determinismgrind | correctness gate / benchmark gate / determinism gate |
| fallbevis | planted-fault test |
| atoms, atoms_runner | machine-checked assertions, assertion runner |
| preflight | startup check |
| fold | record |
| operator, operatorsorder | the author, the author's request |
| lane A-L | an agent worktree |
| poxel (prose only; identifiers keep the word) | goal-derived allocation |
| faltkarna / lastfalt / tillverkningsfalt | field kernel / load field / manufacturing field |
| luftflode | airflow |
| Swedish docstrings and comments throughout | English |

Also removed from prose: absolute paths, process ids, dates, commit hashes, internal document
paths, model names, and any reference to a laboratory, customer or person.

## Files copied (destination -> source path relative to the platform repository root)

| in this repository | source |
| --- | --- |
| src/kernel_engine/allocation/d_poxel_waterfilling_unification.py | scripts/physics_exp/d_poxel_waterfilling_unification.py |
| src/kernel_engine/allocation/d_fpga_bitwidth_waterfilling.py | scripts/physics_exp/d_fpga_bitwidth_waterfilling.py |
| src/kernel_engine/allocation/d_self_tuning_sensitivity_kernel.py | scripts/physics_exp/d_self_tuning_sensitivity_kernel.py |
| src/kernel_engine/allocation/d_goal_derived_representation.py | scripts/physics_exp/d_goal_derived_representation.py |
| src/kernel_engine/certified_loop/d_1c_v_autonomy_loop_end_to_end_real_cuda.py | scripts/physics_exp/d_1c_v_autonomy_loop_end_to_end_real_cuda.py |
| src/kernel_engine/certified_loop/d_lbm_soa_certified_roofline_close.py | scripts/physics_exp/d_lbm_soa_certified_roofline_close.py |
| src/kernel_engine/certified_loop/d_vulkan_port_certified_kernel.py | scripts/physics_exp/d_vulkan_port_certified_kernel.py |
| src/kernel_engine/certified_loop/d_cuda_scene_eyes_determinism_real_gpu.py | scripts/physics_exp/d_cuda_scene_eyes_determinism_real_gpu.py |
| src/kernel_engine/certified_loop/d_privatized_reduction_close_abstain.py | scripts/physics_exp/d_privatized_reduction_close_abstain.py |
| src/kernel_engine/kernel_variants/kernelvarv_v1_f2_matvec.py | scripts/kernel/kernelvarv_v1_f2_matvec.py |
| src/kernel_engine/kernel_variants/kernelvarv_v1_f4_csg.py | scripts/kernel/kernelvarv_v1_f4_csg.py |
| src/kernel_engine/kernel_variants/kernelvarv_v2_f4_closing.py | scripts/kernel/kernelvarv_v2_f4_closing.py |
| src/kernel_engine/lbm/lbm_lattice.py | scripts/physics_exp/lbm_lattice.py |
| src/kernel_engine/lbm/lbm_gpu_fast.py | scripts/physics_exp/lbm_gpu_fast.py |
| src/kernel_engine/lbm/lbm3d_gpu.py | scripts/physics_exp/lbm3d_gpu.py |
| src/kernel_engine/lbm/lbm3d_poiseuille.py | scripts/physics_exp/lbm3d_poiseuille.py |
| src/kernel_engine/lbm/gpu_lbm_luftflode_v1.py | scripts/phys/gpu_lbm_luftflode_v1.py |
| src/kernel_engine/_vendor/lastfalt_v1_fem.py | scripts/kernel/lastfalt_v1_fem.py |
| src/kernel_engine/_vendor/render_match_scaffold.py | scripts/physics_exp/render_match_scaffold.py |
| examples/d_poxel_waterfilling_unification_evidence.json | scripts/physics_exp/artifacts/ (regenerated here) |
| examples/d_fpga_bitwidth_waterfilling_evidence.json | scripts/physics_exp/artifacts/ (regenerated here) |
| examples/d_self_tuning_sensitivity_kernel_evidence.json | scripts/physics_exp/artifacts/ (regenerated here) |
| examples/d_goal_derived_representation_evidence.json | scripts/physics_exp/artifacts/ (regenerated here) |
| examples/d_lbm_soa_certified_evidence.json | scripts/physics_exp/artifacts/d_lbm_soa_certified_evidence.json (scrubbed) |
| examples/d_vulkan_certified_port_evidence.json | scripts/physics_exp/artifacts/d_vulkan_certified_port_evidence.json (scrubbed) |
| examples/kernelvarv_v1.json | reports/probes/kernelvarv_v1.json (scrubbed) |
| examples/kernelvarv_v2.json | reports/probes/kernelvarv_v2.json (scrubbed) |
| examples/gpu_lbm_validation_poiseuille.json | reports/probes/gpu_lbm_validation_poiseuille.json (scrubbed) |
| examples/gpu_lbm_validation_bend90.json | reports/probes/gpu_lbm_validation_bend90.json (scrubbed) |

## Deliberate code changes (the only ones)

1. Three files gained one line inserting `../_vendor` on `sys.path` so the vendored dependencies
   (`lastfalt_v1_fem`, `render_match_scaffold`) resolve from this layout: `kernelvarv_v1_f2_matvec.py`,
   `lbm3d_gpu.py`, `lbm3d_poiseuille.py`.
2. `kernelvarv_v2_f4_closing.py`: the first test case read a production profile from a data file that
   is not published. It now rasterises a synthetic stepped turned profile of the same size class
   (about 1243 x 1201 pixels). The other two cases and all gates are unchanged.
3. `render_match_scaffold.py`: the self-test used an unpublished domain module; it now runs a
   self-contained linear example.
