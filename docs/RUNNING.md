# Running

1. `python -m venv .venv && .venv/bin/pip install -r requirements.txt` (add `wgpu` for the Vulkan port).
2. Every module under `src/kernel_engine/` is also a script: run it directly and it executes its own gates and prints the verdicts.
3. `pytest tests/` wraps those self-tests; the CUDA-only ones skip automatically when no CUDA device is present.
4. The allocation-law modules are CPU-only and take 5-25 s each; they write their evidence JSON next to themselves under `artifacts/`.
5. Reference evidence JSONs from earlier runs are in `examples/`; the four allocation ones there were regenerated on a machine without a GPU.

6. `src/kernel_engine/` is split into packages by solver class (`lbm/`, `euler_hllc/`, `amr_poisson/`,
   `wave_fdtd/`, `fem/`, `warp_gpu/`, `certified_kernels/`, `reductions/`, `allocation/`, `kernel_gen/`,
   `thermo/`, `tropical_sdf/`, `kernel_variants/`, `certified_loop/`, `_vendor/`). Run the scripts with
   `src/kernel_engine` as the working directory: a few of them read and write `data/` and `artifacts/`
   relative to it.
7. Three FEM modules need CAD meshes first: run `fem/cad_to_femmesh.py`, `fem/cad_to_tetmesh.py` and
   `fem/cad_kirsch_mesh.py` (they need `gmsh`, see requirements.txt) before
   `fem/warpfem_cad_elasticity.py`, `fem/warpfem_mms_cad.py`, `fem/warpfem_kirsch.py`,
   `fem/warpfem_mms_cad3d.py` and `fem/warpfem_stress3d.py`. The pytest suite does this automatically.
8. `tests/test_pass3_selftests.py` wraps the modules added in the second pass; the CUDA-only ones skip
   when no device is present.

## Status

One row per shipped module. VERIFIED-FRESH = its self-test ran in this repository's venv on CPU and
reproduced its numbers. CUDA-ONLY = not runnable here until the GPU driver is fixed (either it refuses
without a device, or its CPU fallback exceeds the time budget); last verified in the source project.
SYNTHETIC-ONLY = the method was demonstrated on private input, and it ships with a synthetic input and a test.

| module | status | note |
| --- | --- | --- |
| src/kernel_engine/_vendor/goal_oriented_culling.py | VERIFIED-FRESH |  |
| src/kernel_engine/_vendor/lastfalt_v1_fem.py | VERIFIED-FRESH |  |
| src/kernel_engine/_vendor/render_match_scaffold.py | SYNTHETIC-ONLY |  |
| src/kernel_engine/_vendor/report_sigma.py | VERIFIED-FRESH |  |
| src/kernel_engine/_vendor/uq.py | VERIFIED-FRESH |  |
| src/kernel_engine/allocation/d_fpga_bitwidth_waterfilling.py | VERIFIED-FRESH |  |
| src/kernel_engine/allocation/d_goal_derived_representation.py | VERIFIED-FRESH |  |
| src/kernel_engine/allocation/d_pair_rep_waterfill_goalderived.py | VERIFIED-FRESH |  |
| src/kernel_engine/allocation/d_poxel_waterfilling_unification.py | VERIFIED-FRESH |  |
| src/kernel_engine/allocation/d_self_tuning_sensitivity_kernel.py | VERIFIED-FRESH |  |
| src/kernel_engine/allocation/fpga_bitwidth_waterfilling_is_precision_floor_allocation_marginal_modes_need_more_bits.py | VERIFIED-FRESH |  |
| src/kernel_engine/allocation/l_phase1_quant_x_cert_waterfill.py | VERIFIED-FRESH |  |
| src/kernel_engine/allocation/l_waterfill_deploygap.py | VERIFIED-FRESH |  |
| src/kernel_engine/allocation/l_waterfill_sufficiency.py | VERIFIED-FRESH |  |
| src/kernel_engine/allocation/price_vector_waterfilling_multicommodity_capacity_allocation.py | VERIFIED-FRESH |  |
| src/kernel_engine/allocation/probe_fair_vs_waterfill_knobs.py | CUDA-ONLY | aborted by own idle guard (desktop GPU processes present) |
| src/kernel_engine/allocation/reservation_waterfill_survival.py | VERIFIED-FRESH |  |
| src/kernel_engine/allocation/sigmin_price_vector_waterfilling.py | VERIFIED-FRESH |  |
| src/kernel_engine/allocation/sigmin_waterfill_bstar_stop.py | VERIFIED-FRESH |  |
| src/kernel_engine/amr_poisson/amr_octree_fv.py | VERIFIED-FRESH |  |
| src/kernel_engine/amr_poisson/amr_sigma_scaling.py | VERIFIED-FRESH |  |
| src/kernel_engine/amr_poisson/lbm_poisson.py | VERIFIED-FRESH |  |
| src/kernel_engine/amr_poisson/lightning_dbm.py | CUDA-ONLY | own verdict: charge-em DBM stays BUILDLIST -- mass-radius D=1.57+-0.12 at N=300 is variance-dominated, not the 1.71 target; eta-morphology D 1.78 -> 1.60 -> 1.23 validated |
| src/kernel_engine/amr_poisson/poisson_dispatch.py | VERIFIED-FRESH |  |
| src/kernel_engine/certified_kernels/d_1c_iv_end_to_end_real_cuda_cert.py | VERIFIED-FRESH | reduce_det (tree) 509 GB/s = 76 % peak, run-to-run spread 0 -> CERTIFY; reduce_atomic spread 5.3e+01 -> REJECT |
| src/kernel_engine/certified_kernels/d_avbd_gpu_certified_roofline.py | VERIFIED-FRESH | colored Gauss-Seidel CERTIFIED at 104.3 % of the 603 GB/s copy roofline (Jacobi 51.2 %, not certified); stiffness-ratio advantage grows 5.03x, mass-ratio advantage does not (0.75); 215 s |
| src/kernel_engine/certified_kernels/d_coupled_knob_ordering_advantage_real_cuda.py | CUDA-ONLY | own verdict: "ORDERING CAUSALLY BUYS ITERATIONS ON REAL COUPLED CUDA KNOBS: REFUTED" (exit 1 by design); guided certified 0/3, naive-primary 0/3 at 6 certs, blind-random 14/20 |
| src/kernel_engine/certified_kernels/d_cuda_transcendental_parity.py | VERIFIED-FRESH | bit-exact vs IEEE: cos 0.809, sin 0.839, exp 0.697, rsqrt 0.000; max abs err exp 3.81e-06 (~32 ULP); CUDA repeat x3 reproducible |
| src/kernel_engine/certified_kernels/apriori_requirement_cert_on_real_kernelbench.py | VERIFIED-FRESH |  |
| src/kernel_engine/certified_kernels/kernelbench_addressing_census_provenance_gate_absent.py | VERIFIED-FRESH |  |
| src/kernel_engine/certified_kernels/chunkable_recurrence_rule.py | VERIFIED-FRESH | chunked vs sequential gated delta rule (d=64, T=512, chunk 64): max abs err 3.34e-06 on the outputs, 3.58e-07 on the final state, both inside the fp32 round-off budget; the nonlinear control stays sequential |
| src/kernel_engine/certified_kernels/d_ensemble_coexecution_cert_contended_roofline.py | VERIFIED-FRESH |  |
| src/kernel_engine/certified_kernels/d_r2_r4_dissociation_real_cuda.py | VERIFIED-FRESH | float-atomic R2 FAIL/R4 FAIL 3 GB/s; int64-atomic R2 PASS/R4 FAIL 8 GB/s; tree+atomic-final R2 FAIL/R4 PASS 558 GB/s (83 %) -> R2 and R4 dissociate |
| src/kernel_engine/certified_kernels/d_roofline_scene_eye.py | VERIFIED-FRESH |  |
| src/kernel_engine/certified_kernels/d_wave92_B_friction_gate_roofline.py | VERIFIED-FRESH |  |
| src/kernel_engine/certified_kernels/diag_fp16_bandwidth.py | VERIFIED-FRESH | half2 peaks 8641 MLUPS at N=1024 and settles at ~6400; plain FP16 46 % of FP32 bandwidth, half2 67 %, ~0.87x FP32-fused MLUPS at production grids |
| src/kernel_engine/certified_kernels/fem_sass_roofline.py | VERIFIED-FRESH | FEM quadrature crosses the ridge at AI~6 (nq=4..256 compute-bound); LBM stencil AI 0.23, bandwidth-capped at 2239 GB/s (L2-optimistic) |
| src/kernel_engine/certified_kernels/l_phase0_c2_roofline.py | CUDA-ONLY | own verdict: G3 "BW anchored to spec" FALSE (measured read 676 GB/s outside 0.5-1.0x the 672 GB/s spec -> instrument or spec wrong); G1/G2 pass, MBU sweep 0.07 -> 0.92 |
| src/kernel_engine/certified_kernels/probe_cuda_machinery_port_first_node.py | VERIFIED-FRESH | max_rel 3.22e-07 vs the true MLE; fp64 GPU 9.6x a single CPU core (0.6x per 16 fair cores); verdict PORTED-AND-WORTH-IT |
| src/kernel_engine/certified_kernels/probe_rung3_warp_scheduler.py | VERIFIED-FRESH | occupancy equals the resource-partition formula exactly (incl. 1024 -> 66.67 %); Hill 0.35-0.62 vs null 0.034; own ABSTAIN on rung-3 tail structure (xi gap 1.00, not sign-stable) |
| src/kernel_engine/certified_kernels/probe_tensorcore_precision_rung.py | CUDA-ONLY | aborted by own idle guard (desktop GPU processes present) |
| src/kernel_engine/certified_kernels/u_compute_twin_roofline_ceiling_v1.py | VERIFIED-FRESH | bf16 GEMM 58604.6 GFLOP/s sustained (drift +0.02 %), device-to-device copy 564.1 GB/s, ridge 103.89 FLOP/byte |
| src/kernel_engine/certified_kernels/u_h5_thermal_roofline_twin_state.py | CUDA-ONLY | own verdict: not-C -- ridge 1.930 -> 2.072 FLOP/byte is a 7.4 % shift, below the pre-registered 10 % ("-> FAIL"); all 6 instrument gates pass; 336 s |
| src/kernel_engine/certified_kernels/u_v870_roofline_ceiling_long_window_sustained_throttle_recheck.py | VERIFIED-FRESH | 45 s sustained: 54149.8 GFLOP/s (late decile 53465.4, drift -11.1 %) and 500.6 GB/s (late decile 515.7, drift +2.2 %); ridge 108.18 vs the 3 s value 115.28 FLOP/byte |
| src/kernel_engine/certified_kernels/u_v874_roofline_soft_knee_sweep_tunable_arithmetic_intensity.py | VERIFIED-FRESH | near-ridge efficiency 59.2 % vs 102.7 % far from the ridge -> soft knee confirmed on the roofline axis |
| src/kernel_engine/certified_loop/d_1c_v_autonomy_loop_end_to_end_real_cuda.py | VERIFIED-FRESH | deficit-guided certified in 2 iterations, min-of-10 roofline 93.6 % >= 85 %; random mutation 100 %/20 seeds, mean 4.4 iterations (2.2x); wrong-tile bug caught (relerr 50 %); 131.6 s |
| src/kernel_engine/certified_loop/d_cuda_scene_eyes_determinism_real_gpu.py | VERIFIED-FRESH |  |
| src/kernel_engine/certified_loop/d_lbm_soa_certified_roofline_close.py | VERIFIED-FRESH | fusion 2601 -> 6416 MLUPS (2.47x), layout AoS -> SoA 6416 -> 8185 MLUPS (1.28x); Poiseuille <1 %, bit-repeat x3 true; SUPPORTED |
| src/kernel_engine/certified_loop/d_privatized_reduction_close_abstain.py | VERIFIED-FRESH |  |
| src/kernel_engine/certified_loop/d_vulkan_port_certified_kernel.py | CUDA-ONLY | own verdict: "Gate-1 bit-exact sweep: FAIL" on subnormal inputs (Vulkan flushes to zero, CUDA does not); the other facets transfer (CUDA 90.0 %, Vulkan 94.1 % of theoretical) |
| src/kernel_engine/euler_hllc/axisym_ns_solver.py | VERIFIED-FRESH |  |
| src/kernel_engine/euler_hllc/detonation_cellular.py | VERIFIED-FRESH |  |
| src/kernel_engine/euler_hllc/detonation_znd.py | VERIFIED-FRESH |  |
| src/kernel_engine/euler_hllc/gas_flow_engine.py | VERIFIED-FRESH |  |
| src/kernel_engine/euler_hllc/goc_baseline_honesty.py | VERIFIED-FRESH |  |
| src/kernel_engine/euler_hllc/jeans_instability.py | VERIFIED-FRESH |  |
| src/kernel_engine/euler_hllc/pa_adjoint_vs_cheap_goal.py | VERIFIED-FRESH |  |
| src/kernel_engine/euler_hllc/pa_adjoint_vs_cheap_proxies.py | VERIFIED-FRESH |  |
| src/kernel_engine/euler_hllc/phenomenon_registry.py | VERIFIED-FRESH |  |
| src/kernel_engine/euler_hllc/sigma_solver_routing.py | VERIFIED-FRESH |  |
| src/kernel_engine/euler_hllc/substrate_compose.py | VERIFIED-FRESH | hot-swap radiation changes cloud dT by +0.262 and blocks 16 % as shadow; chemistry module burns 100 % of the fuel (dT +0.228); tracer conserved to 1e-02 |
| src/kernel_engine/euler_hllc/traffic_flow_lwr.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/buckling_euler_column.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/cad_kirsch_mesh.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/cad_to_femmesh.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/cad_to_tetmesh.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/euler_buckling.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/fatigue_life.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/fem3d_elasticity.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/fem3d_modal.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/fem3d_orthotropic.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/fem3d_thermoelastic.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/fracture_lefm.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/spectral_fatigue.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/viscoelasticity_creep.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_acoustic_design.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_acoustic_modal.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_adjoint_arbitrary.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_cad_elasticity.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_compliant_inverter.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_compute_router.py | VERIFIED-FRESH | in-distribution conformal coverage 0.905+-0.020, high-frequency OOD AUROC 1.00 (caught 100 %), structured OOD 14 %; 6293x faster in distribution |
| src/kernel_engine/fem/warpfem_coupled_adjoint.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_design_gradient.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_design_objective.py | VERIFIED-FRESH | point objective J 6.645e-04 -> 3.859e-05 (94 %, monotone), residual guard max 7.8e-11; distance to the compliance design 0.20 |
| src/kernel_engine/fem/warpfem_elasticity_validate.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_em_magnetostatics.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_field_surrogate.py | VERIFIED-FRESH | held-out field-shape median relative error 10.6 % (p90 15.3 %) at 751x (21.2 ms -> 28.3 us) |
| src/kernel_engine/fem/warpfem_field_surrogate_fallbevis_v1.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_kirsch.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_mms_3d.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_mms_cad.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_mms_cad3d.py | VERIFIED-FRESH | field error 2.43e-05 (tol 0.005) on the gmsh tet volume, CG residual 1.00e-12, 10059 P2 nodes |
| src/kernel_engine/fem/warpfem_mms_elasticity.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_modal.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_navier_stokes.py | VERIFIED-FRESH | Kovasznay Re=40: field error 7.63e-04 (tol 0.03), saddle residual 7.01e-07 |
| src/kernel_engine/fem/warpfem_stokes_poiseuille.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_stress3d.py | VERIFIED-FRESH | equilibrium gate: mean sigma_xx 1.1058e+06 vs applied, rel 4.23e-07; max von Mises 3.186e+06 Pa at the hole |
| src/kernel_engine/fem/warpfem_surrogate.py | CUDA-ONLY | own verdict: "USABLE (BUT MARGINAL / seed-dependent) ... NOT a robust pass" -- held-out median 7.6 % grazes the 10 % gate, p90 19.7 %, fair speedup 172x |
| src/kernel_engine/fem/warpfem_thermoelastic.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_topopt.py | VERIFIED-FRESH | compliance 2.531e+04 -> 3.330e+03 (87 % stiffer) at volume fraction 0.40 |
| src/kernel_engine/fem/warpfem_transient_heat_energy.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/warpfem_transient_structural.py | VERIFIED-FRESH |  |
| src/kernel_engine/kernel_gen/d_crossbackend_kernel_port_verify.py | VERIFIED-FRESH | CUDA vs Vulkan bit-exact 1.0000, max abs delta 0.00e+00 on the 3-point fma stencil (needs wgpu) |
| src/kernel_engine/kernel_gen/d_l1_kernel_certvec_compose.py | VERIFIED-FRESH |  |
| src/kernel_engine/kernel_variants/kernelvarv_v1_f2_matvec.py | CUDA-ONLY | own verdict: overall_pass false, exit 1 by design -- correctness/determinism/benchmark gates pass (1.54x) but the race counter-evidence gate did not fire |
| src/kernel_engine/kernel_variants/kernelvarv_v1_f4_csg.py | CUDA-ONLY | own verdict: overall_pass false, exit 1 by design -- no live variant clears the 1.2x benchmark gate; winner 1d_baseline |
| src/kernel_engine/kernel_variants/kernelvarv_v2_f4_closing.py | SYNTHETIC-ONLY |  |
| src/kernel_engine/lbm/acoustic_streaming.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/cloud_morphology.py | CUDA-ONLY | own verdict: honest-negative, exit 1 by design -- cloud fraction 0.6 % is a degenerate one-row layer so box-D=1.000 is trivial and the fractal instrument is not exercised; water drift 1.8e-03 conserved; 495 s |
| src/kernel_engine/lbm/d_certvector_on_real_lbm.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/d_integration_stitch_lbm_contact_scenario_cert.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/d_per_variable_precision_cert_lbm.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/d_static_deployment_gate_real_lbm.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/differentiable_lbm_probe.py | VERIFIED-FRESH | int64 fixed-point objective: two-launch max abs difference 0.0 (float atomics 3.58e-07) |
| src/kernel_engine/lbm/double_diffusive_lbm.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/excitable_media_fhn.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/gpu_lbm_luftflode_v1.py | VERIFIED-FRESH | pipe Poiseuille: umax/umean 2.0085 vs 2.0 (0.42 % error), dp/dx error 3.45 %; both 5 % gates pass |
| src/kernel_engine/lbm/gpu_lbm_utilization_cell.py | SYNTHETIC-ONLY | measures 2037-3198 MLUPS and 64162 pJ/site, then reads reports/probes/asic_fallback_feasibility.json; its producer asic_fallback_feasibility_cell.py is not shipped, so gate G2 cannot be evaluated here |
| src/kernel_engine/lbm/hartmann_mhd_lbm.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/lbm3d_gpu.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/lbm3d_immersed_boundary.py | VERIFIED-FRESH | IBM drag 6.7013e-02 inside the Stokes-Hasimoto band [5.878e-02, 7.759e-02], confinement K(c)=1.530; int64 fixed-point force spreading: selftest reproduces bit-identically over two runs (float atomics: drag differs 1e-06) |
| src/kernel_engine/lbm/lbm3d_poiseuille.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/lbm_aero_v0.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/lbm_compressible_boundary.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/lbm_fsi_bouzidi.py | VERIFIED-FRESH | interpolated bounce-back changes St by 1 % only (Re93 0.1744, Re139 0.1915 vs Williamson 0.1606/0.1800); the modules own hypothesis "Bouzidi tightens St below 2 %" is falsified |
| src/kernel_engine/lbm/lbm_fsi_gpu.py | VERIFIED-FRESH | St 0.178/0.189/0.213 at Re 93/139/209, ~7 % median vs Williamson-1989; no shedding at Re~56 (correct onset) |
| src/kernel_engine/lbm/lbm_fsi_viv.py | CUDA-ONLY | own verdict: exit 1 by design -- frequency capture validated but no amplitude peak (A/D 0.025-0.043, monotone); large-amplitude lock-in needs higher Re |
| src/kernel_engine/lbm/lbm_gpu_fast.py | VERIFIED-FRESH | fused SoA 10352 MLUPS = 111 % of the FP32 ceiling 9333 (5.8x the baseline); Poiseuille L2 0.0 % |
| src/kernel_engine/lbm/lbm_gpu_fp16.py | CUDA-ONLY | own verdict: "FP16 storage wins = PARTIAL (precision holds no - throughput 8923 MLUPS yes)", Poiseuille L2 9.22 %, exit 1 by design |
| src/kernel_engine/lbm/lbm_gpu_fp16_half2.py | VERIFIED-FRESH | half2 7858 MLUPS = 42 % of the FP16 ceiling, 1.22x plain FP16 and 1.07x FP32-fused; Poiseuille L2 9.22 % |
| src/kernel_engine/lbm/lbm_lattice.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/lbm_mach_ceiling.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/lbm_mrt_stability.py | VERIFIED-FRESH | MRT stays stable to >=23.3x lower viscosity than BGK (diverges at nu=2.33e-03) at the same grid/Mach/float32; TRT ties BGK |
| src/kernel_engine/lbm/lbm_voxel_aero.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/lbm_voxel_aero_gpu.py | VERIFIED-FRESH | Strouhal 0.175 (Roshko), Poiseuille validated, 156x the CPU implementation |
| src/kernel_engine/lbm/magnetic_induction_lattice.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/moist_convection_lbm.py | VERIFIED-FRESH | cloud forms (q_c max 6.911, cloudy fraction 0.83 %), water drift 2e-03, heat 3e-16; enhancement stays 1.00x -> the moist-boundary-layer fix is falsified (condensation is pinned to the cold-top sink); 416 s |
| src/kernel_engine/lbm/probe_lbm_fp16_steady_dither.py | VERIFIED-FRESH | G1 golden-stride dither 295.2x pass, G2 win (deterministic bias is DC); G3: not 1/K (ramp bowl 0.064 % at K=3 up to 6.49 %); G4 control: dither hurts the unsteady case 7.15x |
| src/kernel_engine/lbm/rayleigh_taylor_lbm.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/reaction_diffusion_turing.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/rte_lbm.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/thermofluid_lbm_rayleigh.py | VERIFIED-FRESH | marginal Ra_c 1825/1782/1761 at ny 40/52/64, Richardson p=2 (R2=1.000) -> Ra_inf 1719, 0.7 % from the continuum 1707.76 |
| src/kernel_engine/reductions/d_1c_real_reduction_roofline_torch.py | VERIFIED-FRESH | cub reduction 658 GB/s = 98 % of the 672 GB/s theoretical peak; copy 613 GB/s = 91 % |
| src/kernel_engine/reductions/probe_kernel_determinism_cert.py | VERIFIED-FRESH |  |
| src/kernel_engine/reductions/u_v54_gather_law_deterministic_reduction_vs_atomic_int64.py | VERIFIED-FRESH |  |
| src/kernel_engine/kernel_gen/export/ (build.sh, host_int64_reduce.cu, warp_reference_run.py) | VERIFIED-FRESH | exported reduce_int64_atomic compiled with nvcc: output bit-identical to the Warp-launched original; 70.4/70.8 GB/s Warp vs 73.2/73.0 GB/s nvcc = 0.111/0.112 vs 0.116/0.116 of the 632 GB/s identity |
| src/kernel_engine/thermo/biot_transient_conduction.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/bohm_sheath_criterion.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/d_nonnormal_thermo_price_of_amplification_probe.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/d_thermo_computing_equals_fusion.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/d_thermo_datahole_law.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/d_thermo_voi_law_multiworld.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/debye_specific_heat.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/diffusion_induced_stress.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/explosion_combustion.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/fizeau_drag.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/fourier_conduction.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/heat_pipe_capillary_limit.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/heat_pipe_sigma_budget.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/heat_pump_cop.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/heat_pump_sigma_budget.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/hopf_reaction_diffusion.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/induction_heating_skin.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/ising_thermo.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/joule_heating_thermal.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/kapitza_acoustic_mismatch.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/p8_combustion_cert_pod_a90.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/p8_combustion_cert_watertight.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/p8_combustion_state_certification.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/p8_delft_conditional_variance.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/p8_delft_flamelet_render_match.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/p8_delft_species_flamelet.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/p8_ecn_combustion_energy.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/p8_flame_mixture_fraction_render_match.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/p8_sandia_extinction_flamelet_breakdown.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/photoelasticity_isochromatics.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/photon_diffusion_escape.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/saffman_delbruck_diffusion.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/sommerfeld_electron_heat.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/soret_thermodiffusion.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/thermocouple_seebeck.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/thermoelectric_seebeck.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/tidal_heating.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/wheatstone_bridge.py | VERIFIED-FRESH |  |
| src/kernel_engine/thermo/wkb_quantization.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/2jet_sdf_curvature_decouples_v6.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/JxA_contact_param_over_determination_material_physics_leg_lifts_the_motion_regime_null.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/certified_generative_support_placement_worstcase_sigmamin_robust_dfc.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/contact_cert_chi_from_sdf_error.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/contact_dof_cert_identify_or_abstain.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/contact_orientation_gauge_emergence.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/contact_penalty_conditioning_omega_max_side_accuracy_cost_tradeoff_completes_sim_conditioning.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/engine_contact_manifold_rank_twist_selfstress_2point_reduction.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/engine_manifold_reduction_needs_3points_not_2_wrench_span.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/engine_which_3_points_Eoptimal_sigmamin_manifold_not_max_area.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/friction_mu_is_a_sigma_min_null_under_stick_lifted_only_by_slip_events_contact_sim_ready.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/mma_capstone_validity_band_sweep.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/mma_topopt_capstone_stress_constrained_dfc_positive.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/restitution_null_lifted_by_impact_friction_and_restitution_are_decorrelated_contact_nulls.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/so_arm100_contact_sim_cost_from_tropical_sdf_backend_hertz_stiffness_sets_stable_dt_stability_verified.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/stress_constrained_compliance_topopt_dfc_positive.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/stress_objective_topopt_design_for_certifiability_positive.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/topopt_stress_frontier_forced_number_not_honest_label.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/topopt_stress_vs_compliance_design_for_certifiability.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/tropical_sdf_SIMT_tax_cost_model_compute_bound_vs_voxel_bandwidth_bound_crossover_Kstar.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/tropical_sdf_backend_computes_full_v6_contact_geometry_ladder_2jet_curvature_anisotropy_on_so_arm100.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/tropical_sdf_backend_min_of_capsules_beats_voxel_3x_bytes_matched_penetration_so_arm100.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/tropical_sdf_so_arm100_geometry_backend_v2_config_driven_posing_plus_contact_normal_gradient.py | VERIFIED-FRESH |  |
| src/kernel_engine/tropical_sdf/tropical_sdf_swept_volume_continuous_collision_detection_min_over_trajectory.py | VERIFIED-FRESH |  |
| src/kernel_engine/warp_gpu/gpu_fracture_determinism_sigma.py | VERIFIED-FRESH |  |
| src/kernel_engine/warp_gpu/mujoco_warp_bench.py | VERIFIED-FRESH | MuJoCo-Warp 306-320 steps/s; 5.01M box-steps/s at N=16384 (needs mujoco + mujoco-warp, now in requirements.txt) |
| src/kernel_engine/warp_gpu/rigid2d_primal_vbd_gpu.py | VERIFIED-FRESH | penetration <=1.5e-8, top-angle difference 0.0 deg, force error <=0.1 %, finite 100 % at mass ratio 1e4; 214977 stacks/s at B=65536 |
| src/kernel_engine/warp_gpu/warp_bodybody_colored_gpu.py | VERIFIED-FRESH | colored Gauss-Seidel stable over mass ratios 30-1000x; 196 mm at 100x with 40 iterations matches relaxed Jacobi at ~300 (7x fewer iterations) |
| src/kernel_engine/warp_gpu/warp_bodybody_jacobi_gpu.py | VERIFIED-FRESH | N=125: 49582 box-steps/s (1.65x real time); Coulomb drift 0.404 m vs 0.408 analytic (1 %); K=8 tower stable |
| src/kernel_engine/warp_gpu/warp_bodybody_scale_gpu.py | VERIFIED-FRESH | N=125: 71.2 steps/s, 8904 box-steps/s, 1123 contacts, 27 colours, stable |
| src/kernel_engine/warp_gpu/warp_cylinder_roll_gpu.py | VERIFIED-FRESH | travel 0.575/1.132/1.656 m at 10/20/30 deg vs theory 0.57/1.12/1.63 |
| src/kernel_engine/warp_gpu/warp_granular_friction.py | VERIFIED-FRESH | repose angle 1.4/5.0/6.4/6.5 deg for mu 0/0.3/0.6/1.0 |
| src/kernel_engine/warp_gpu/warp_granular_gpu.py | VERIFIED-FRESH | 7207 steps/s at 10k particles (30x real time), 2110 steps/s at 200k |
| src/kernel_engine/warp_gpu/warp_mesh_roll_gpu.py | VERIFIED-FRESH | sphere-mesh travel 0.485/1.019/1.560 m at 10/20/30 deg (theory 0.61/1.20/1.75 for a=5/7 g sin), v ~ omega r |
| src/kernel_engine/warp_gpu/warp_rigid_ramp_gpu.py | VERIFIED-FRESH | transition angle follows atan(mu) within 0.5 deg for mu 0.2-1.0; controls: mu=0 slides, mu=2 sticks |
| src/kernel_engine/warp_gpu/warp_rigid_ramp_gpu_v2.py | VERIFIED-FRESH | 3-D cone brakes lateral slip to 3.4 cm drift with atan(mu) preserved (<=1.5 deg); warm start 2.35 vs cold 2.88 deg at one iteration |
| src/kernel_engine/warp_gpu/warp_rl_env_gpu.py | VERIFIED-FRESH | 46.6M env-steps/s at 131072 envs; P-controller mean distance 1.52 -> 0.38 m |
| src/kernel_engine/wave_fdtd/acoustic_emission.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/acoustic_sigma_renderer.py | SYNTHETIC-ONLY |  |
| src/kernel_engine/wave_fdtd/acoustic_fdtd.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/diag_acoustic_mode.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/diag_acoustic_seed.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/diag_acoustic_spectrum.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/diff_wave_3d.py | VERIFIED-FRESH | int64 fixed-point window energy: two-launch max abs difference 0.0 (float atomics 1.91e-06); the printed gradient still comes from Warp's backward pass |
| src/kernel_engine/wave_fdtd/diff_wave_substrate.py | VERIFIED-FRESH | int64 fixed-point focus energy: two-launch max abs difference 0.0 (float atomics 2.38e-07); the printed gradient still comes from Warp's backward pass |
| src/kernel_engine/wave_fdtd/elastodynamics_kache.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/goc_wave_transient.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/optics_coating.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/optics_fdtd.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/persona_design_acoustic.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/persona_design_acoustic_cavity_shape.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/persona_design_acoustic_inverse.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/persona_design_acoustic_shape_adjoint.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/persona_design_lbm_acoustic.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/s1_acoustic_metamaterial_bandgap.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/thermoacoustic_rijke_dde.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/vibroacoustic.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/wave_fdtd_3d.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/wave_fdtd_kache.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/wave_fdtd_verify.py | VERIFIED-FRESH |  |
| src/kernel_engine/certified_kernels/aa_micro_bench.py | VERIFIED-FRESH | median own/pull 0.99x (26907 vs 26872 MLUPS at N=2048) -> conversion-bound; AA co-location not worth building |
| src/kernel_engine/certified_kernels/d_energy_exponent_substrate_invariant.py | VERIFIED-FRESH | p_CPU mean 0.942 (cv 0.109), p_GPU mean 1.869 (cv 0.020), between-substrate dp 0.927 -> ACCEPT_invariant, NULL does not fire |
| src/kernel_engine/certified_kernels/gpu_duty_torch_v1.py | VERIFIED-FRESH |  |
| src/kernel_engine/certified_kernels/module_const_launch_tune.py | VERIFIED-FRESH |  |
| src/kernel_engine/certified_kernels/morton_3d_stencil.py | VERIFIED-FRESH |  |
| src/kernel_engine/certified_kernels/morton_sparse_gather.py | VERIFIED-FRESH |  |
| src/kernel_engine/certified_kernels/probe_compute_ladder_descent.py | VERIFIED-FRESH | atomic_f64 spread 3.702e-08 (122.6 ulp) vs tree_f64 0; coalesced 608 vs strided 147 GB/s (4.13x); launch latency p99 120.8 us |
| src/kernel_engine/certified_kernels/probe_exclusive_sweep_block.py | CUDA-ONLY | aborted by own idle guard (desktop GPU processes present) |
| src/kernel_engine/certified_kernels/probe_l2_inband_endgame.py | CUDA-ONLY | aborted by own idle guard (desktop GPU processes present) |
| src/kernel_engine/certified_kernels/probe_sync_density_victim_model.py | CUDA-ONLY | aborted by own idle guard (desktop GPU processes present) |
| src/kernel_engine/certified_kernels/probe_sync_penalty_vs_priority.py | VERIFIED-FRESH | sync penalty 3.277 ms at priority 0 vs 0.491 ms at high priority (85 % reduction); H_A true, H_B (15 % invariance) false |
| src/kernel_engine/certified_kernels/probe_timeslice_dma_fartail.py | CUDA-ONLY | aborted by own idle guard (desktop GPU processes present) |
| src/kernel_engine/certified_kernels/ser_fracture_compaction.py | VERIFIED-FRESH | compaction up to 8.6x, peak at active fraction ~0.031 (one active lane per warp); 2.0x on the dense set |
| src/kernel_engine/euler_hllc/lubrication_reynolds_bearing.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/fsi_added_mass.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/fsi_pipe_flutter.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/neuber_notch_plasticity.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/plasticity_3d_j2.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/plasticity_return_mapping.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/thermal_buckling.py | VERIFIED-FRESH |  |
| src/kernel_engine/fem/viscoelastic_preload_relaxation.py | VERIFIED-FRESH |  |
| src/kernel_engine/lbm/coupled_design_aero_struct.py | VERIFIED-FRESH | combined aero+structure gradient matches central FD on 5/5 cells (<10 %); CG converged (<0.1 %); +89 % stiffness costs +89 % drag |
| src/kernel_engine/lbm/differentiable_flow_control.py | VERIFIED-FRESH | int64 fixed-point loss and probe: selftest reproduces bit-identically over two runs (float atomics 7.45e-09 / 5.96e-08 per launch) |
| src/kernel_engine/lbm/differentiable_fsi_chain.py | VERIFIED-FRESH | int64 fixed-point drag seam: two-launch max abs difference 0.0 (float atomics 4.77e-07); the calibration loop is driven by Warp's backward pass and still varies |
| src/kernel_engine/lbm/g18_cfd_nilss_prereq_wake_chaos.py | VERIFIED-FRESH | the positive global-field lambda is convective amplification, not temporal chaos -> the 2-D laminar wake is not a NILSS bed |
| src/kernel_engine/lbm/g19_forced_2d_wake_nilss_prereq.py | VERIFIED-FRESH | forced 2-D wake is lock-in or quasi-periodic over A/D 0.2-0.5 and f_e/f_shed 0.7-1.3, no robust chaos; honest-negative verdict = pass |
| src/kernel_engine/lbm/g20_3d_wake_chaos_nilss_prereq.py | VERIFIED-FRESH | G1: rms(uz)/U 0.019 -> 0.074 at Re=300, lambda_z 4.0D vs Barkley-Henderson mode-A 3.96D; G2: broadband fraction 0.01/0.12 vs the 2-D baseline 0.25, twin-lambda 1.8e-05 -> boundary, honest-negative = pass; 678 s |
| src/kernel_engine/lbm/g21_3d_wake_chaos_highRe.py | VERIFIED-FRESH | reduced case only (--validate, the modules own short self-test): Re=150 St 0.210, <C_D> 1.62, broadband 0.04; Re=400 ramp 16.11 ms/step, rms(uz)/U 0.0012 finite. The full 288x176x224 sweep exceeds 1500 s here (no phase completed) |
| src/kernel_engine/lbm/probe_kam_resonance_dither_strides.py | CUDA-ONLY | own verdict: G1 (KAM ordering) FAIL -0.609 and G2 (non-coprime tier) FAIL 0.48x; G3 14.5x and G4 4/5 pass -> "literal-metric FALSIFIED, signal in spirit" |
| src/kernel_engine/reductions/d_1c_iv_best_in_class_float4.py | VERIFIED-FRESH | scalar tree 597 GB/s (89 %), float4 tree 638 GB/s (95 %), cub 650 GB/s (97 %); all deterministic and correct |
| src/kernel_engine/reductions/det_accumulation_probe.py | VERIFIED-FRESH |  |
| src/kernel_engine/wave_fdtd/goc_wave_verify.py | VERIFIED-FRESH |  |
| src/kernel_engine/_vendor/_emit.py | VERIFIED-FRESH | evidence-JSON writer used by the wake cells; exercised by g39/g42/g61 |
| src/kernel_engine/certified_kernels/probe_dma_single_stream_gap.py | VERIFIED-FRESH | H1 (per-transfer overhead) false, H2 (stream-depth cap) true: 16/64/256 MiB chunks give 24.07/23.33/24.09 GB/s and the queued-depth arm gains 0.6 % |
| src/kernel_engine/certified_kernels/probe_dma_split_shape_decisive.py | VERIFIED-FRESH | the instrument decides the split shape: interleaved-windowed enqueue 11.69/11.69 GB/s (even, agg 23.4), batch-enqueue-per-stream 23.4/11.7 GB/s (uneven, same 23.3-23.5 aggregate) |
| src/kernel_engine/certified_kernels/determinism_sweep_int64.py | VERIFIED-FRESH | 10 int64 accumulation sites bit-identical over two launches, 5/8 selftests bit-identical over two runs, 3 adjoint-carrying reported |
| src/kernel_engine/certified_kernels/probe_l2_arbitration_endgame.py | CUDA-ONLY | own verdict: "G-ctrl FAILED - sensors not calibrated; matrix uninterpretable" (hog +0.8/-0.8, stream +0.8/-0.825), the spatial-locality artefact its own header documents |
| src/kernel_engine/fem/fno_3d.py | VERIFIED-FRESH | SpektralConv3d linear 3.9e-07, high mode (k=5 > kept 3) damped to 2.0e-16, held-out rel L2 0.982 -> 0.059, mesh-flexible on a 16^3 grid |
| src/kernel_engine/fem/fno_tpu_compatible.py | VERIFIED-FRESH | truncated-DFT matmul equals the FFT spectral convolution to 1.9e-15 (mode block 2.2e-15), no torch.fft in the matmul path, autograd finite |
| src/kernel_engine/lbm/g39_cfd_force_nilss_bed.py | CUDA-ONLY | --validate green (grid 416x272x200 = 22.6M, 31.31 ms/step, rms(uz)/U 0.0009 finite, twin 6.9 GB); the act itself is ~106000 step-equivalents ~ 55 min, past the run cap here |
| src/kernel_engine/lbm/g42_wake_benettin_dim.py | CUDA-ONLY | --smoke exceeds the 300 s run cap on the shared GPU |
| src/kernel_engine/lbm/g61_modeA_Rec_cylinder_vs_ellipse.py | CUDA-ONLY | --validate exceeds the 300 s run cap on the shared GPU |
| src/kernel_engine/reductions/d_1c_v_atomic_saturation_check.py | VERIFIED-FRESH | at N=67M a single-float32 atomic accumulator is bit-identical over 20 reps (16777216.0 every time) and 50 % wrong: saturation at 2^24, inside the (N-1)*eps = 8.0 naive-summation bound |
| src/kernel_engine/warp_gpu/substep_value_probe.py | VERIFIED-FRESH | substepping win not confirmed: the substep path blows up at mass ratio 100x+ while the baseline is stable; per-body adaptive under-relaxation (relax <= 1/contacts) is stable 30x-1000x (106/342/420/420 mm) where fixed relax=0.25 blows up |
| src/kernel_engine/warp_gpu/warmstart_massratio.py | VERIFIED-FRESH | honest negative: pre-applying the cached normal impulse destabilises relaxed Jacobi at every ratio (cold 151/421/420/421 mm at 20/50/100/300x, warm blows up) |
| src/kernel_engine/warp_gpu/warmstart_value_probe.py | VERIFIED-FRESH | its own instrument guard fires: the warm branch diverges (KE 705.7 at vit4 vs cold 0.145 at vit40), so the value question is unresolved rather than negative |
| src/kernel_engine/wave_fdtd/xray_3d_dda.py | VERIFIED-FRESH | internal 3-D field recovered from 24 external views at rel 8 %; error rises 6.6x from low-sigma to high-sigma regions, monotone; int64 fixed-point back-projection: selftest reproduces bit-identically over two runs (float atomics 1.25e-06 per launch) |
| src/kernel_engine/wave_fdtd/xray_tomography_sigma.py | VERIFIED-FRESH | 3/3 gates; mean absolute error by coverage-sigma quartile 0.015/0.016/0.017/0.059, monotone, top/bottom band 4.0x; int64 fixed-point sensitivity and residual loss: selftest reproduces bit-identically over two runs (float atomics 1.53e-05 / 4.06e-01 per launch); gate-1 finite-difference agreement improved from 8.58e-03 to 4.45e-05 |

293 module rows: 267 VERIFIED-FRESH, 22 CUDA-ONLY, 4 SYNTHETIC-ONLY.


## Real datasets

The dataset files go in `data/<slug>/` at the repository root. `data/` is in `.gitignore` and nothing under it ships
with the repository; each slug below names the public source and the exact files a module opens, so the runs recorded
below can be reproduced. Sizes are what the 2026-09-12 run used (160 MB total); the upstream datasets are larger, only
the files named here are needed. When a slug directory is absent the module prints `SYNTHETIC INPUT` and runs the same
pipeline on generated input.

| slug | public source | files needed | size | modules |
| --- | --- | --- | --- | --- |
| tnf-flames | TNF Workshop archives, tnfworkshop.org: Sandia piloted flames C-F (`pmCDEF.zip`) and Delft Flame III (`DATA_BASE_DELFT_FLAME_III_April_2003.zip`) | `pmCDEF.zip` (7.7 MB), `DATA_BASE_DELFT_FLAME_III_April_2003.zip` (170 KB) | 7.9 MB | p8_flame_mixture_fraction_render_match, p8_sandia_extinction_flamelet_breakdown, p8_delft_flamelet_render_match, p8_delft_species_flamelet, p8_delft_conditional_variance, p8_combustion_state_certification, p8_combustion_cert_pod_a90, p8_combustion_cert_watertight |
| ecn-spray-a | Engine Combustion Network Spray A, ecn.sandia.gov (constant-volume pressure traces) | `press_reacting.txt` | 671 KB | p8_ecn_combustion_energy |
| kernelbench | KernelBench, github.com/ScalingIntelligence/KernelBench (MIT), ICML 2025 | `level1/*.py` (100), `level2/*.py` (100), `level3/*.py` (50), `level4/*.py` (20) | 1.2 MB | apriori_requirement_cert_on_real_kernelbench (level1), kernelbench_addressing_census_provenance_gate_absent (all four levels) |
| paderborn_kat_severity | Paderborn University KAt bearing data centre, mb.uni-paderborn.de (KAt-DataCenter) | `<code>/N15_M07_F10_<code>_*.mat` for the nine codes K001, KA01, KI01, K002, KA03, KI03, K003, KA04, KI04 (6 records each); only K001, KA01, KI01 are present here | 150 MB | acoustic_sigma_renderer (SYNTHETIC-ONLY, see below) |

Real-data results reproduced in this repository on 2026-09-12 (CPU, `.venv-kernel`):

- `p8_flame_mixture_fraction_render_match`: Sandia D30.Ycnd, ξ_st = 0.352 vs measured T-peak at ξ = 0.381; Burke-Schumann
  over-predicts the peak T by 12%; peak 2022 K against ~300 K frozen mixing.
- `p8_sandia_extinction_flamelet_breakdown`: conditional T rel-RMS at ξ_st D 0.047 -> E 0.059 -> F 0.219 (n = 9317 / 8576 /
  10345 samples), monotone, F is 4.7x worse than D.
- `p8_delft_flamelet_render_match`: DNG ξ_st = 0.072 vs measured T-peak ξ = 0.070; B-S over-predicts the peak by 1.29x;
  lean-side NRMSE 18%.
- `p8_delft_species_flamelet`: H2O peaks at ξ = 0.080 (ξ_st 0.072), lean-branch NRMSE 10%, B-S/measured peak ratio 1.28.
- `p8_delft_conditional_variance`: H2O bulk median rel-RMS 0.24; at ξ_st CO 0.64 vs H2O 0.33.
- `p8_ecn_combustion_energy`: Δp = 0.39 bar (cross-checked against the 0.25 bar absolute-column rise), ΔQ = 142 J vs fuel
  energy 166 J, η_c = 0.85, identifiability range [0.48, 1.55].
- `p8_combustion_state_certification`: 3 scalars x 6 ξ-bins = 18 constraints; RMS-z D 0.0 -> E 0.3 -> F 2.0σ; in F the
  temperature is 3.6x further off the manifold than CO.
- `p8_combustion_cert_pod_a90`: a90 = 1.18σ per constraint at 5% false-alarm; POD 0.05 for burning D, 1.00 for extinction F.
- `p8_combustion_cert_watertight`: RMS-z vs Reynolds number correlation 0.94, monotone D 0.0 < E 0.3 < F 2.0σ, T/CO
  decoupling 3.6x in F.
- `apriori_requirement_cert_on_real_kernelbench`: 89 of 100 level-1 problems parsed, 0 input-keyed, all structural ->
  computed a-priori; histogram worst case 256x its uniform typical. Fifth requirement (R5, on by default,
  `--no-chunkable-rule` turns it off): 5 of the 100 level-1 files carry state across a sequence and all 5 are chunkable
  (associative scan), byte floor 524,288 -> 264,192 B at chunk 128; the G1-G3 verdicts are identical with the flag on
  and off.
- `kernelbench_addressing_census_provenance_gate_absent`: 270 kernels (100/100/50/20), input-keyed write count 0 on all
  three paths; the injected bincount control routes to `distributional` and abstains. Fifth requirement (R5, chunkable
  recurrence): 17 of the 270 kernels carry state across a sequence and were all classed sequential before the rule; 7 of
  them change class to chunkable (level1 5: `89_cumsum`, `90_cumprod`, `91_cumsum_reverse`, `92_cumsum_exclusive`,
  `93_masked_cumsum`, byte floor 524,288 -> 264,192 B at chunk 128; level3 2: `48_Mamba2ReturnY`,
  `49_Mamba2ReturnFinalState`, byte floor 303,104 -> 43,008 B at chunk 128), and 10 stay sequential (2 VanillaRNN, 4
  LSTM, 4 GRU - nonlinear state map). Levels 2 and 4 hold no recurrence. Gates G1-G3 unchanged.
- `chunkable_recurrence_rule`: synthetic gated-deltanet layer (per-channel diagonal gate, d=64, T=512, chunk 64),
  chunked vs step-by-step max abs error 3.34e-06 on the outputs (|O|max 12.2, fp32 round-off budget 9.30e-05) and
  3.58e-07 on the final state (budget 2.64e-05); the nonlinear recurrence h_t = tanh(W h_(t-1) + x_t) is classed
  sequential and chunking it anyway gives max abs error 2.0; byte floors for a 64x64 state over T=512: 17,432,576 B
  sequential -> 917,504 B chunked (19.0x total, state traffic 64x), suggested chunk 64 at an 80.0 KiB inner working set.

`acoustic_sigma_renderer` stays SYNTHETIC-ONLY: the archive copy holds only the three TRAIN bearing codes, so the module's
own data-availability gate would refuse the real run with

    RuntimeError: DEGENERATE CALIB SET -- missing bearing .mat files for ['K002', 'KA03', 'KI03'] (CALIB collected 0
    samples). Re-fetch data/paderborn_kat_severity/ before trusting this script's output.

It ships with the stand-in signal generator instead; no tolerance in it was changed.


## Deterministic accumulation

Float atomic addition is not associative, so a reduction built from `wp.atomic_add` on a float array returns a
value that depends on the order in which blocks reach the accumulator, and that order varies from run to run.
Each affected module now carries a module-level switch `DETERMINISTIC_ACCUMULATION = True`: every contribution is
rounded in float64 to an int64 fixed point, summed with integer atomics (associative and commutative, so
order-invariant), and converted back once. The scale is a power of two derived from the quantity's declared range,
with an overflow assertion `|sum| x scale < 2^62` next to it. Setting the switch to `False` restores the float
path for A/B comparison; no tolerance was changed.

Measured on an RTX 5070 (Warp 1.17, CUDA 12.9). "site delta" = same input, two launches of the accumulation
kernel, max abs difference of the result array. "time" = mean over 20 launches after 3 warm-up launches.

| module | sites | float delta | int64 delta | time ratio int64/float |
| --- | --- | --- | --- | --- |
| lbm/differentiable_flow_control | track_loss, probe_ux | 7.45e-09, 5.96e-08 | 0.0, 0.0 | 0.98, 1.08 |
| lbm/differentiable_lbm_probe | objective | 3.58e-07 | 0.0 | 1.41 |
| lbm/differentiable_fsi_chain | drag_force | 4.77e-07 | 0.0 | 1.22 |
| lbm/lbm3d_immersed_boundary | spread | 5.82e-11 | 0.0 | 1.58 |
| wave_fdtd/diff_wave_3d | energy | 1.91e-06 | 0.0 | 1.09 |
| wave_fdtd/diff_wave_substrate | focus_loss | 2.38e-07 | 0.0 | 1.22 |
| wave_fdtd/xray_tomography_sigma | sensitivity, sq_resid | 1.53e-05, 4.06e-01 | 0.0, 0.0 | 1.50, 1.09 |
| wave_fdtd/xray_3d_dda | backproject | 1.25e-06 | 0.0 | 2.41 |

Whole-selftest reproduction (two runs, every printed number compared; float path = switch set to `False`):

| module | float path max abs difference | int64 path max abs difference |
| --- | --- | --- |
| lbm/differentiable_flow_control | 1.0e-11 | 0.0 |
| lbm/differentiable_lbm_probe | 0.0 | 0.0 |
| lbm/differentiable_fsi_chain | 1.0e-02 | 3.5e-01 (gradient-descent loop, see below) |
| lbm/lbm3d_immersed_boundary | 1.0e-06 | 0.0 |
| wave_fdtd/diff_wave_3d | 4.2e-03 | 1.0e-06 (adjoint, see below) |
| wave_fdtd/diff_wave_substrate | 6.4e-04 | 1.0e-07 (adjoint, see below) |
| wave_fdtd/xray_tomography_sigma | 1.95 | 2.0e-07 (adjoint, see below) |
| wave_fdtd/xray_3d_dda | 0.0 | 0.0 |

Gates are unchanged or tighter: the immersed-boundary drag stays 6.8697e-02 inside the Stokes-Hasimoto band, the
3-D X-ray recovery stays at 8.02 % with the same monotone sigma quartiles, the tomography gates stay 3/3 while the
gate-1 adjoint-vs-finite-difference agreement improves from 8.58e-03 to 4.45e-05, the 3-D wave finite-difference
agreement improves from 1.14e-03 to 1.62e-04, and the chained-FSI parameter recovery improves from 2.70 % to
0.03-0.19 % error (its verdict moves from PARTIAL to VALIDATED because the finite-difference noise is gone).
`differentiable_flow_control` still reports PARTIAL for the same reason as before (the jet recovers 0 % of the wake
deficit); that is unrelated to accumulation.

What was NOT changed, and why:

- Gradient values printed after `tape.backward()`. Warp's generated adjoint kernels accumulate gradients with
  float atomics inside Warp itself, not at a call site in this repository, and a quantiser has no useful
  derivative, so the forward kernels used inside a tape keep the float accumulation. That is the residual in the
  int64 column above: the forward and finite-difference numbers are bit-identical, the adjoint-driven iterates
  (the 80-step parameter recovery in `differentiable_fsi_chain`, the 400-step reconstruction in
  `xray_tomography_sigma`) are not.
- `struct_loss` in `differentiable_fsi_chain` is launched with `dim=1`: one thread, no cross-thread accumulation,
  already order-fixed.
- Warp's own tree reductions are order-fixed: `wp.utils.array_sum` over 2^20 float32 values returned one distinct
  value in 20 consecutive calls, and `torch.Tensor.sum()` over 2^24 float32 values likewise 1 distinct value in 20.
- `certified_kernels/d_coupled_knob_ordering_advantage_real_cuda.py:248` builds its reference with a float64
  `index_add_`; measured two-run divergence is 2.13e-14 absolute, 1.08e-15 relative, far below the tolerance the
  cell checks, so it is left as is.
- The probes that exist to DEMONSTRATE float-atomic non-determinism keep their float atomics by construction:
  `reductions/det_accumulation_probe.py`, `reductions/u_v54_gather_law_deterministic_reduction_vs_atomic_int64.py`,
  `certified_loop/d_cuda_scene_eyes_determinism_real_gpu.py`, `certified_loop/d_privatized_reduction_close_abstain.py`,
  `certified_kernels/probe_compute_ladder_descent.py`, `certified_kernels/probe_l2_arbitration_endgame.py`,
  `certified_kernels/probe_l2_inband_endgame.py`, `certified_kernels/d_roofline_scene_eye.py`,
  `kernel_variants/kernelvarv_v1_f2_matvec.py` (whose float `atomic_add` is a planted defect for a detector).
- The momentum-exchange force sums in the long wake/FSI sweeps (`lbm/lbm_fsi_gpu.py`, `lbm/lbm_fsi_viv.py`,
  `lbm/lbm_fsi_bouzidi.py`, `lbm/g18_cfd_nilss_prereq_wake_chaos.py`, `lbm/g19_forced_2d_wake_nilss_prereq.py`,
  `lbm/g20_3d_wake_chaos_nilss_prereq.py`, `lbm/g21_3d_wake_chaos_highRe.py`,
  `lbm/g61_modeA_Rec_cylinder_vs_ellipse.py`, `lbm/coupled_design_aero_struct.py`) are the same pattern and do
  affect reported numbers, but their selftests are multi-minute to multi-hour sweeps, so the two-run A/B that
  justifies the change was not run for them in this pass; they keep the float path.
- `fem/`, `euler_hllc/`, `thermo/` and `amr_poisson/` contain no `wp.atomic_add` on a float array; the FEM
  assembly goes through `warp.fem`, and the torch reductions in those modules are `.sum()` (order-fixed above).

Sweep cell:

    PYTHONPATH=src python3 src/kernel_engine/certified_kernels/determinism_sweep_int64.py

It launches each touched accumulation kernel twice (float twin and int64 twin), runs each touched selftest twice as
a subprocess and compares the normalised stdout hashes, writes `reports/determinism_sweep_int64_result.json`, and
exits non-zero if any int64 site or any non-adjoint-carrying selftest differs. `tests/test_determinism_sweep_int64.py`
wraps it and skips without a CUDA device.

### Innovation target 5: frozen adjoint baseline, before design

Run the existing eight site/selftest modules on Modal L4, unchanged. Use the
existing site inputs and stdout normalizer. Each selftest executes twice in its
own process; compare the full normalized text and report numerical deltas only
when the nonnumeric text aligns. Printed deltas are diagnostic, not full-array
adjoint certification. Preserve unaligned/failed outputs as failures, not zero.

Fixed target gates: all int64 site deltas exactly0; every selftest exits0 twice;
all eight normalized selftest hashes identical, including adjoint-carrying cases.
The old sweep exempts four adjoint-carrying modules and does not gate returncodes;
this new observer records those distinctions and exempts none. No baseline code,
physical thresholds or tolerances change. No timing claim, no local GPU use.

| module | status | evidence |
|---|---|---|
| `certified_kernels/innovation_adjoint_baseline.py` | VERIFIED-FRESH | L4 five/eight stdout pairs identical; FSI0.49, wave3D9e-8, tomography2e-7 printed deltas; flow-control exits1/1. |

Measured frozen baseline on L4, two runs per selftest:

| module | returncodes | normalized text identical | max aligned printed delta |
|---|---|---|---|
| lbm/differentiable_flow_control | 1/1 | True | 0 |
| lbm/differentiable_lbm_probe | 0/0 | True | 0 |
| lbm/differentiable_fsi_chain | 0/0 | False | 0.49 |
| lbm/lbm3d_immersed_boundary | 0/0 | True | 0 |
| wave_fdtd/diff_wave_3d | 0/0 | False | 9e-08 |
| wave_fdtd/diff_wave_substrate | 0/0 | True | 0 |
| wave_fdtd/xray_tomography_sigma | 0/0 | False | 2e-07 |
| wave_fdtd/xray_3d_dda | 0/0 | True | 0 |

Evidence: `reports/innovation_adjoint_baseline_l4.json`. All int64 site deltas
are0; the all-eight text-identity and all-zero-returncode gates FAIL. Flow control's
repeatable failure remains visible and is not an adjoint-order success claim.
The 2-D wave's printed numbers agree on these runs; that is not proof that its
unprinted full gradient arrays are identical. No numerical tolerance changed.
3-D wave inspection shows reversed velocity stencils scatter up to six additions
into a pressure-gradient cell. A separate gather adjoint can give each gradient
cell one writer while accumulating time steps in a fixed reverse order; preserve
the frozen forward kernels and its existing finite-difference gate.

### Target 5 first fixed-order adjoint: design and gates before execution

The L4 baseline wave3D printed delta9e-8 and source stencil fan-in motivate a
separate gather adjoint. Reuse frozen forward kernels, same seed,N30,T46 andcs0.4.
Store forward velocity history. Reverse pressure gathers both neighbouring
contributions for each velocity gradient and accumulates the local material
contribution in fixed reverse time order. Reverse velocity gathers its six
pressure-gradient contributions in fixed axis order. One writer per output cell;
no float atomic reductions in the new backward kernels. Seed the window-energy
gradient analytically. The deterministic forward energy helper remains unchanged.

Fixed gates: two entire gradient/forward arrays and result dictionaries byte-
identical; exact final forward array vs baseline; finite gradient; existing three
FD cells,eps1e-2 and relative error<0.05 unchanged. This is one fixed synthetic
fixture, not closure of all eight selftests or a performance claim. Additional
history storage is explicit; no baseline sources or tolerances change.

| module | status | evidence |
|---|---|---|
| `wave_fdtd/diff_wave_3d_gather.py` | VERIFIED-FRESH | L4 two whole gradients/forwards identical; frozen forward exact; max FD relative error0.000258836922<0.05 PASS. |

Measured gather-adjoint table, L4, two complete runs identical:

| cell | gather derivative | unchanged FD derivative | relative error |
|---|---|---|---|
| 15,15,15 | 0.000385787629057 | 0.000385850047735 | 0.000161769264 |
| 10,15,15 | -0.000683061603922 | -0.000683057481865 | 0.000006034715 |
| 15,20,15 | -0.000131959473947 | -0.000131993638774 | 0.000258836922 |

All four gates PASS; exact frozen forward. Full gradient hash
`a786c35e0967b6417b3d6951ffa13d5ec70c51dfb54206e7504e9a5661c63eec`
is identical in both runs. Evidence: `reports/diff_wave_3d_gather_l4.json`.
This closes the measured one-fixture wave3D gradient repeatability experiment,
not all eight modules; no timing claim or default replacement.

### Target 5 runtime segmented reduction, registered after baseline table

Local Warp1.17 runtime inspection exposes RUN_TO_RUN mode: scatter records,
sort by destination/thread, then fixed-order reduction, including generated
backward kernels. A separate harness sets that mode before importing each frozen
selftest, keeping the reference module list/normalizer and all original code.
This tests runtime-provided machinery, not a new implementation of its algorithm.
Use default record bounds; overflow or unsupported operations remain failures.

Fixed gates: all eight selftests run twice, all normalized hashes identical,
all returncodes0 and no runtime exceptions. Unchanged physics thresholds; flow
control's known repeatable exit1 will still fail the all-zero-returncode gate.
Do not call identical error traces successful determinism. Capability in the
cloud package must be demonstrated, not inferred from its version string.
No performance claim and no baseline/default modification.

| module | status | evidence |
|---|---|---|
| `certified_kernels/innovation_runtime_adjoint.py` | VERIFIED-FRESH | L4 six/eight pairs exit0 and identical; flow-control repeatable exit1, tomography repeated RuntimeError; target FAIL despite matching hashes. |

### Innovation target 6: frozen LBM stream mechanism before export design

Measure the existing differentiable_lbm_probe.stream kernel, without collision,
on512x512 and1024x1024 D2Q9 populations. Seed20260912, fixed wall mask and existing
direction/opposite arrays. Compare every output byte against an independent CPU
index/gather construction. Two isolated workers, two correctness launches each,
then10 warmup +30 timed launches, both CUDA events and synchronized wall time.
Strict empty-context guard before and after workers; no in-context device polling.
Population read+write byte floor=72*N*N; direction/mask traffic and cache effects
are excluded from that floor, so no peak-bandwidth fraction is claimed yet.

Fixed instrument gates: exact CPU reference, identical repeated launch and worker
hashes. Write the measured table before the native export design. The later C-host
export gate remains exact bytes and bandwidth ratio within10% of the Warp launch.
No baseline or tolerance changes; no foreign-engine integration claimed.

| module | status | evidence |
|---|---|---|
| `kernel_gen/innovation_lbm_stream_baseline.py` | VERIFIED-FRESH | L4 both sizes exact CPU/output bytes in two workers;1024 event0.348362/0.349317ms,216.72/216.13 payload GB/s. |

Measured runtime-mode table, L4, two runs:

| module | exit codes | identical hashes | exceptions |
|---|---|---|---|
| lbm/differentiable_flow_control | 1/1 | True | none |
| lbm/differentiable_lbm_probe | 0/0 | True | none |
| lbm/differentiable_fsi_chain | 0/0 | True | none |
| lbm/lbm3d_immersed_boundary | 0/0 | True | none |
| wave_fdtd/diff_wave_3d | 0/0 | True | none |
| wave_fdtd/diff_wave_substrate | 0/0 | True | none |
| wave_fdtd/xray_tomography_sigma | 1/1 | True | RuntimeError |
| wave_fdtd/xray_3d_dda | 0/0 | True | none |

Evidence: `reports/innovation_runtime_adjoint_l4.json`. Six/eight pairs pass both
returncode and identity; FSI and both waves now reproduce. Flow control retains
its physical exit1. Tomography produces matching RuntimeError traces: NOT a
successful determinism result. All-eight successful completion remains open.
A bounded single-worker diagnostic will expose its runtime failure before any
record-capacity adjustment; default bounds and all tolerances remain unchanged.

### Target 5 tomography record bound, registered before completed execution

The isolated L4 diagnostic fails in `project` backward with deterministic scatter
buffer overflow: default capacity yields zero completed selftests. Source mechanism:
110 samples per ray/thread, four bilinear field reads per sample, therefore at most
440 field-gradient records per thread. This is an allocation bound, not a numeric
tolerance. The separate bounded wrapper sets capacity440 before module creation;
the default-mode comparator and frozen tomography source remain unchanged.

Fixed gates: two normalized outputs exactly identical; both original selftests
return0 (FD<0.05, covered error<0.25, monotonic quartiles and band>2 unchanged);
no runtime exceptions; all three printed metrics present. Overflow remains a hard
failure. Printed-output identity does not certify every hidden gradient byte.
An initial submission was aborted during setup because the documentation append
used the wrong working directory. Zero completed measurements from that submission
are accepted; this registered text precedes the replacement submission.

| module | status | evidence |
|---|---|---|
| `certified_kernels/innovation_tomography_record_bound.py` | VERIFIED-FRESH | L4 capacity440: two exit0 and exact output hashes; FD4.41e-5, covered error5.85%, quartile ratio4.0; original gates PASS. |

Bounded tomography evidence: `reports/innovation_tomography_record_bound_l4.json`.
Both normalized hashes `13ef262907a11b88666150abc33db30ff14d74fb66ed4aeae5ecf9b83f2ec693`.
Default overflow remains a measured failure; seven/eight successful pairs are now
observed across the separate runs. No throughput or hidden-array identity claim.

### Target 5 composed sweep, gates before execution

Run all eight frozen selftests twice through the measured runtime wrapper,
selecting the separately measured440-record wrapper only for tomography.
Fixed independent gates: all eight normalized texts byte-identical; no runtime
exceptions; all eight original physics selftests return0. The last gate retains
flow control's physical negative without exemptions. Repeatability success and
physical correctness are reported separately, and a failed physics gate still
makes the harness return1. No hidden-array or cross-architecture identity claim.

| module | status | evidence |
|---|---|---|
| `certified_kernels/innovation_runtime_bounded_sweep.py` | VERIFIED-FRESH | L4 all8 normalized outputs identical,0 runtime exceptions;7/8 physical passes, flow-control exit1 retained; overall physical gate FAIL. |

Measured stream table before native export design, L4:

| side | payload bytes | event ms, runs1/2 | wall ms, runs1/2 | payload GB/s, runs1/2 |
|---|---|---|---|---|
| 512 | 18874368 | 0.0433493/0.0431808 | 0.0442241/0.0439996 | 435.402/437.101 |
| 1024 | 75497472 | 0.348362/0.349317 | 0.349239/0.350174 | 216.722/216.129 |

All three instrument gates PASS. Evidence: `reports/innovation_lbm_stream_baseline_l4.json`.
The512 payload-rate elevation is consistent with cache reuse; no off-chip peak
fraction is inferred. The1024 case carries75.5 MB per launch and gives a more
useful bandwidth-oriented export comparison. Both sizes remain fixed test cases.

### Target 6 native stream export, gates before execution

Export only the frozen stream forward function emitted by Warp1.17, preserving
its generated operations and launch shape. Remove the NVRTC-only WP_NO_CRT macro
for offline nvcc, strip source-location comments, retain existing licensed header
closure. A separate CUDA shim constructs the same array/launch descriptors and
exposes a C ABI. A gcc-compiled C host reads raw fixture arrays and calls the shim;
no Warp or Python runtime is linked into the resulting executable.

Fixed gates: both sizes match every frozen Warp output byte; two native runs
byte-identical; two isolated Warp/native timing legs with empty-context checks
between processes; native/Warp payload-bandwidth ratio in[0.9,1.1] in both legs
at both sizes. Same10 warmups/30 CUDA-event and wall repetitions, allocations and
transfers excluded on both sides. No clock, tolerance or baseline changes. These
are same-device bandwidth ratios, not a DRAM peak-fraction measurement. A C-host
result does not establish integration into a separate decode engine.

| module | status | evidence |
|---|---|---|
| `kernel_gen/innovation_lbm_stream_export.py` | CUDA-ONLY | Separate C host/generated stream export ready for registered two-leg L4 correctness and bandwidth gates. |
| `kernel_gen/stream_export_v1/stream_generated.cu` | CUDA-ONLY | Frozen forward operations exported; execution and exact-output comparison pending. |
| `kernel_gen/stream_export_v1/stream_shim.cu` | CUDA-ONLY | Native C ABI with repeated-output and event/wall measurement prepared; pending execution. |
| `kernel_gen/stream_export_v1/host_stream.c` | CUDA-ONLY | C11 host prepared for offline build and two-run execution. |

Measured composed sweep, L4: all8/8 normalized selftest hashes match twice,
0 runtime exceptions,7/8 original selftests exit0. Flow control alone exits1/1
with identical hash and its previously recorded zero wake recovery. The harness
therefore returns1. Target5 whole-selftest text repeatability is met under the
explicit runtime/bound configuration; all-eight physical success is NOT met.
Evidence: `reports/innovation_runtime_bounded_sweep_l4.json`. Hidden full arrays
remain outside this sweep's scope; only the separate wave3D gather experiment
checks its whole gradient and forward arrays. No defaults are promoted.
