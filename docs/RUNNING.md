# Running

Status values: VERIFIED-FRESH = the module's own gates passed in the recorded run; OWN-GATE-FAIL = one or more own gates failed, result retained and unpromoted; CUDA-ONLY = GPU execution not yet verified in the recorded scope; SYNTHETIC-ONLY = passing evidence limited to synthetic inputs. Earlier attempts are notes, not additional current module rows.

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

One row per shipped module. VERIFIED-FRESH requires every own gate to pass in the
recorded scope; OWN-GATE-FAIL retains measured own-gate failures. CUDA-ONLY means
GPU execution is not yet verified; SYNTHETIC-ONLY limits passing evidence to
synthetic inputs. The hardware and decisive values are stated with each new result.

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
| src/kernel_engine/lbm/differentiable_flow_control.py | OWN-GATE-FAIL | Own selftest exit1/1, zero printed wake recovery; int64 fixed-point loss and probe: selftest reproduces bit-identically over two runs (float atomics 7.45e-09 / 5.96e-08 per launch) |
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
| `certified_kernels/innovation_adjoint_baseline.py` | OWN-GATE-FAIL | L4 five/eight stdout pairs identical; FSI0.49, wave3D9e-8, tomography2e-7 printed deltas; flow-control exits1/1. |

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
| `certified_kernels/innovation_runtime_adjoint.py` | OWN-GATE-FAIL | L4 six/eight pairs exit0 and identical; flow-control repeatable exit1, tomography repeated RuntimeError; target FAIL despite matching hashes. |

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
| `certified_kernels/innovation_runtime_bounded_sweep.py` | OWN-GATE-FAIL | L4 all8 normalized outputs identical,0 runtime exceptions;7/8 physical passes, flow-control exit1 retained; overall physical gate FAIL. |

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
| `kernel_gen/innovation_lbm_stream_export.py` | OWN-GATE-FAIL | L4 all output bytes exact twice;1024 bandwidth ratio1.002739/1.002741 PASS;512 ratio1.890411/1.924847 FAIL fixed10% band. |
| `kernel_gen/stream_export_v1/stream_generated.cu` | VERIFIED-FRESH | L4 generated body matches frozen Warp; both512/1024 outputs exact CPU/Warp bytes twice. |
| `kernel_gen/stream_export_v1/stream_shim.cu` | VERIFIED-FRESH | L4 native event1024 .348877/.348672ms; exact outputs, both repeated launches identical. |
| `kernel_gen/stream_export_v1/host_stream.c` | VERIFIED-FRESH | L4 actual gcc C11 caller built and ran twice per size; all output bytes exact; no Warp runtime linked. |

Measured composed sweep, L4: all8/8 normalized selftest hashes match twice,
0 runtime exceptions,7/8 original selftests exit0. Flow control alone exits1/1
with identical hash and its previously recorded zero wake recovery. The harness
therefore returns1. Target5 whole-selftest text repeatability is met under the
explicit runtime/bound configuration; all-eight physical success is NOT met.
Evidence: `reports/innovation_runtime_bounded_sweep_l4.json`. Hidden full arrays
remain outside this sweep's scope; only the separate wave3D gather experiment
checks its whole gradient and forward arrays. No defaults are promoted.

Measured native export table, L4, same payload floor and isolated idle guards:

| side | Warp event ms1/2 | C-host event ms1/2 | C-host wall ms1/2 | native/Warp bandwidth1/2 | fixed band |
|---|---|---|---|---|---|
| 512 | 0.0423936/0.0428373 | 0.0224256/0.0222549 | 0.0227301/0.0225956 | 1.890411/1.924847 | FAIL |
| 1024 | 0.349833/0.349628 | 0.348877/0.348672 | 0.349258/0.349026 | 1.002739/1.002741 | PASS |

All full outputs exact against the independent CPU construction and frozen Warp
hashes; native and Warp repeats identical. The bandwidth-oriented1024 case meets
the original10% target. The small case is faster but OUTSIDE the preregistered
symmetric band, so the combined gate and harness FAIL (exit1). No tolerance was
relaxed to reinterpret a speedup as a pass. Evidence:
`reports/innovation_lbm_stream_export_l4.json`. Hypothesis, not measured attribution:
Python submission gaps may dominate the small event interval; cache reuse also
prevents treating its payload rate as DRAM throughput. Next mechanism test would
compare captured launch batches. No external decode-engine integration yet.

## Headless RT toolchain proof, before backend design

The installed OptiX SDK's unmodified optixTriangle target configures and builds
successfully with a single CPU build worker. No SDK source, binary or library is
vendored by this experiment. This is a build observation, not an RT engine result.

Preregistered runtime gates: launch the existing sample twice at128x96 with file
output, both exits0; both outputs valid P6 RGB images of exactly128x96; each image
has at least16 distinct colors and both hit-blue255 and non-hit-blue pixels;
complete output bytes identical. No performance claim, no SER or field-distance
certificate. A caller supplies OPTIX_TRIANGLE_BINARY; output reports contain
hashes, not local installation paths. Local launches require an externally held
shared GPU lock; optional journal guard rejects a current-boot Xid before each
launch. No retries or hardware/settings changes after a fault.

| module | status | evidence |
|---|---|---|



### Installed SDK runtime negative

| observation | result |
|---|---|
| unmodified SDK9.1 configure/build | exit0 / exit0 |
| first headless launch | exit1, OPTIX_ERROR_UNSUPPORTED_ABI_VERSION |
| accepted images |0|
| second launch |not attempted|
| current-boot Xid guard |no fault observed|

Evidence: reports/optix_sample_proof_sdk91.json. Runtime/valid-image/exact-repeat
gates FAIL; the fault guard passes. No driver, module or system setting changes.

| module | status | evidence |
|---|---|---|


Official compatibility sources: [9.1 release](https://forums.developer.nvidia.com/t/optix-9-1-release/354119)
requires R590; [9.0 release](https://github.com/NVIDIA/optix-sdk/releases/tag/v9.0.0)
requires R570 or later. The installed R580 driver cannot provide the9.1 ABI.
A separately downloaded official9.0 SDK is the next controlled toolchain variant;
the installed9.1 SDK and failed report stay frozen. Announced download, no driver
change, no SDK source/binary vendoring. Same sample source from that release and
same128x96/two-run/image/fault gates; no tolerance change. Build outside this repo.


### Compatible SDK runtime result

| quantity | first leg | second leg |
|---|---|---|
| process exit |0|0|
| image dimensions / bytes |128x96 /36878|128x96 /36878|
| distinct colors |1626|1626|
| hit-blue pixels |1625|1625|
| non-hit pixels |10663|10663|
| image SHA256 |75bf1e37c2d3fb133770115870282ed8bc9bcbcd3651c8d382f5e9c021013b02|same|

All4 preregistered gates PASS with the unmodified official SDK9.0 sample.
No current-boot fault observed before, between or after launches. The installed
SDK9.1 comparator and its zero-image ABI failure remain unchanged.
Archive provenance: official v9.0.0,58314739 bytes,
SHA2563fa55fb41e9f5b86bc5b26006ea56de8848b7380cf334132e0e87b92efb3a17e.
SDK source and binaries stay external; no driver or hardware setting changed.

Only image bytes are certified identical. Driver diagnostic stderr is not:
lengths3484/993, differing byte positions3195; its hashes are
retained in the report. No full-log determinism or timing claim is made.
Evidence: reports/optix_sample_proof_sdk90.json and optix_sample_1.ppm/2.ppm.

| module | status | evidence |
|---|---|---|
| `kernel_gen/optix_sample_probe.py` | VERIFIED-FRESH | SDK9.0 all4 gates PASS; two128x96 image payloads exact,1625 hit pixels, no observed fault. SDK9.1 runtime failure retained separately. |

The toolchain is ready for a separate ray-query mechanism table. No shared field,
motion or photon RT backend has yet been implemented, and no Euclidean distance,
ray parity, fluence, SER or throughput comparison is implied by this triangle.

## Signed RT winding candidate: design after contract measurement

The field contract probe (field commit6b20bad) measured2520 parity mismatches for
an overlapping box soup and512 for a nested soup. A vertical ray hit distance2
also differs from nearest face distance0.25. Accordingly this separate native
candidate uses built-in triangle traversal with any-hit signed-normal summation,
not parity, and exports int32 winding counts. The host field adapter retains the
frozen EDT and half-pitch correction. This is a winding seam, not a true-distance
backend. Triangle/ray inputs are float32; precision losses are measured rather
than hidden. Each primitive requests one any-hit invocation per trace; every hit
is ignored after accumulation so traversal continues. Hits at ray parameter0
are excluded, preserving the strict-above boundary convention in real arithmetic.

Preregistered correctness gates, before native execution: two independent process
runs return0; complete winding arrays repeat exactly; occupancy matches the frozen
CPU path at every voxel in all five contract fixtures; complete reconstructed
float32 EDT arrays match the frozen arrays; no current-boot Xid. No tolerance,
timing claim, automatic retry, default promotion or true-distance claim. SDK and
compiler are external inputs. The caller holds one shared GPU lock.

Native signed RT winding v1 result: all five fixed gates PASS. Ten independent
process runs return0; current-boot guards observe no fault. The five complete
int32 winding arrays repeat exactly and their EDT outputs reproduce every frozen
float32 sample. This covers these synthetic fixtures only.

| case | voxels | occupancy mismatches, both legs | distance mismatches, both legs | full winding repeat |
|---|---|---|---|---|
| box | 9261 | 0 / 0 | 0 / 0 | exact |
| offset | 10648 | 0 / 0 | 0 / 0 | exact |
| reversed | 9261 | 0 / 0 | 0 / 0 | exact |
| overlap | 12650 | 0 / 0 | 0 / 0 | exact |
| nested | 9261 | 0 / 0 | 0 / 0 | exact |

Evidence: field artifacts/field_rt_winding_v1.json and matching full-array NPZ.
No throughput, general-mesh, float64 equivalence, true-distance or integration
claim. Next: non-axis-aligned and scale/translation precision stress before timing.

| module | status | evidence |
|---|---|---|
| `kernel_gen/rt_winding_v1/params.h` | SYNTHETIC-ONLY | Native build0; field integration probe five gates PASS on ten process runs, zero mask/EDT differences. |
| `kernel_gen/rt_winding_v1/program.cu` | SYNTHETIC-ONLY | Native build0; field integration probe five gates PASS on ten process runs, zero mask/EDT differences. |
| `kernel_gen/rt_winding_v1/host.cpp` | SYNTHETIC-ONLY | Native build0; field integration probe five gates PASS on ten process runs, zero mask/EDT differences. |
| `kernel_gen/rt_winding_v1/build.sh` | SYNTHETIC-ONLY | Native build0; field integration probe five gates PASS on ten process runs, zero mask/EDT differences. |

## Native lifecycle phase observer: preregistered

Field timestamped lifecycle table (a17c0dc): eight total-speed failures, native
process308.656–395.597ms versus CPU whole8.526–15.267ms; outputs exact. This
motivates measurement of native phases before any persistent-context design.
Separate host observer preserves the winding program/PTX and input/output ABI.
It adds steady-clock boundaries for input read, CUDA initialization, OptiX context/
stream, vertex upload/GAS, module/pipeline, SBT/ray upload, launch/sync, readback/
write and teardown. One added stream synchronization completes GAS for attribution;
this may alter scheduling and is measurement overhead, not an optimized backend.
Phase durations are emitted as JSON on stdout; no new implicit report path.

Fixed gates: two full outputs repeat exactly and match the frozen CPU mask/EDT;
all phases finite/nonnegative; same two-leg whole-speed criterion and idle/fault
guards retained. Timings not bit-identical. Fresh timestamped single-flock wrapper
records actual sample/lock intervals; no performance attribution before results.

Measured native phase table before persistent-context design; eight samples, two complete legs:

| phase | minimum ms | maximum ms |
|---|---|---|
| input_read | 0.282731 | 0.618623 |
| cuda_init | 133.637243 | 212.904865 |
| optix_context_stream | 61.113501 | 62.520345 |
| vertices_gas | 0.360600 | 0.503462 |
| module_pipeline | 12.474591 | 12.648881 |
| sbt_ray_upload | 0.091445 | 0.109543 |
| launch_sync | 0.052439 | 0.063420 |
| readback_write | 0.168561 | 0.220930 |
| teardown | 6.867611 | 7.396214 |
| process wall minus measured phases | 87.654203 | 88.179775 |

Six/seven gates PASS; all8 whole-speed comparisons FAIL. Complete winding/EDT
arrays repeat exactly and match frozen CPU arrays. PTX SHA unchanged; native
observer is separate from the stable host. Positive finite phase durations are
measurements, not bit-identical timing claims. The launch includes stream sync;
GAS includes vertex upload and allocation. Residual process time above includes
unmeasured work before/after instrumented scope and cannot be assigned to a
specific driver or loader operation. CUDA initialization and OptiX context setup
dominate measured phases; no persistent-context speedup has yet been measured.

Actual timing lock:2026-09-13T00:57:11.550008+00:00 through
2026-09-13T00:57:14.946740+00:00. Full UTC/monotonic intervals retained, GPU idle
prechecks and journal guards PASS. Coordination overlap audit requested; no claim
of complete absence of background activity. Prior timestamped run separately
received confirmation of no competing coordinated heavy work in its actual
interval; that does not remove arbitrary desktop activity or certify the earlier
untimestamped run.

Evidence: field artifacts/field_rt_native_phase_v1.json, matching arrays and
field_rt_native_phase_events.jsonl. A JSON copy is in kernel reports. Next design
may retain context/pipeline/GAS across repeated queries, but must report setup
cost separately and include required transfers/EDT for whole-call comparisons.

| module | status | evidence |
|---|---|---|
| `kernel_gen/rt_winding_phase_probe/host.cpp` | OWN-GATE-FAIL | Build0, eight processes0, exact frozen outputs; all8 whole-speed FAIL; phase table retained. |
| `kernel_gen/rt_winding_phase_probe/build.sh` | OWN-GATE-FAIL | Build0, eight processes0, exact frozen outputs; all8 whole-speed FAIL; phase table retained. |

## Reused-context batch v1: design after native phase table

The preceding phase table measures initialization134–213ms, context61–63ms and
pipeline12.5ms versus launch/sync0.052–0.063ms. Separate batch host therefore
retains context, pipeline and GAS for exactly64 identical queries. Every query
reuploads all ray origins, launches unchanged PTX, synchronizes and copies all
winding counts to the host; every array is compared exactly to the first. Mesh
and rays remain fixed for this mechanism experiment. This is not a public mutable
query API or an independent set of64 geometries. Original host remains frozen.

Before execution: two complete batches per fixed fixture must yield exact full
arrays and match CPU mask/EDT, every query must repeat exactly, phase values
finite/nonnegative, idle/fault guards pass. Whole batch time includes one startup,
geometry preparation/I/O, all64 ray upload/launch/readbacks and64 CPU EDT/surface
reconstructions. Divide by64 and compare with the mean of64 full frozen CPU
selector calls; require faster in both legs/all cases. Setup remains included,
not discarded. Fixed64 is not tuned after results. No general speed claim.

Measured fixed64 context-reuse table; all times per query except where named:

| case | leg | CPU mean ms | native total/64 ms | ratio | native upload/launch/readback mean ms |
|---|---|---|---|---|---|
| sphere | 1 | 9.016938 | 14.733381 | 1.633967 | 0.102874 |
| sphere_fine | 1 | 9.634500 | 11.548295 | 1.198640 | 0.110216 |
| sphere_overlap | 1 | 12.319462 | 14.922455 | 1.211291 | 0.135080 |
| subdivided_rotated_box | 1 | 8.257692 | 11.813627 | 1.430621 | 0.103378 |
| sphere | 2 | 7.853162 | 11.592415 | 1.476146 | 0.102517 |
| sphere_fine | 2 | 8.395122 | 11.382494 | 1.355846 | 0.107672 |
| sphere_overlap | 2 | 11.308501 | 15.075160 | 1.333082 | 0.137713 |
| subdivided_rotated_box | 2 | 9.639023 | 11.687982 | 1.212569 | 0.103525 |

Six/seven gates PASS; all8 total-speed comparisons FAIL. All64 native result
arrays compare exactly within each process; full CPU64-call distance results,
64 native-side EDT reconstructions and the two complete native runs are exact.
The final full arrays match frozen reference hashes. No output or PTX change.

Per-query native upload/launch/readback mean0.102517–0.137713ms demonstrates
context reuse on this fixed repeated input, but it excludes one-time setup and
CPU EDT. Whole total/64 includes those required costs and remains slower by
1.198640–1.633967 in this run. Fixed64 and the failed gate are retained. These
figures are neither a general API certificate nor evidence of faster one-off
queries. The host exports only the final array after internally checking all64;
it is not an interactive API that can accept changed queries between launches.

The phase JSON contains query_mean/min/max summaries nested within the batch
phase; do not sum those summary fields as disjoint phases. native_process_ms is
whole64-query child process time, while native_adapter_total_ms is whole batch/64.
Existing compiler cache/no warmup retained. UTC/monotonic actual lock interval
2026-09-13T01:12:45.715432+00:00–2026-09-13T01:12:57.854180+00:00;
idle/fault guards pass, overlap audit requested and not yet received. No complete
background-idle claim. Native phase table from the preceding run separately
received confirmation of no coordinated heavy work in its actual interval.

Evidence: field artifacts/field_rt_batch_v1.json, matching full arrays and
field_rt_batch_events.jsonl; kernel reports/rt_winding_batch_v1.json mirrors the
metrics. Next: a callable ownership/lifecycle seam with changed-query correctness,
then representative query amortization and EDT mechanism work. Do not infer an
optimal batch size or loosen64 from this failed experiment. Stable single-shot
API and every prior negative remain unchanged.

| module | status | evidence |
|---|---|---|
| `kernel_gen/rt_winding_batch_v1/host.cpp` | OWN-GATE-FAIL | Build0, fixed64 full-query repeats exact, all8 total-speed FAIL; query mean0.102517–0.137713ms excludes setup/EDT. |
| `kernel_gen/rt_winding_batch_v1/build.sh` | OWN-GATE-FAIL | Build0, fixed64 full-query repeats exact, all8 total-speed FAIL; query mean0.102517–0.137713ms excludes setup/EDT. |

## Persistent changed-query API: gates before execution

After the CPU changed-query table (field b6fbb00, phase differences985–1904), add
an independent C ABI create/query/destroy handle owning OptiX context/stream/GAS/
pipeline and fixed-capacity buffers. Immutable local float32 triangle corners;
query supplies new finite ray origins and receives int32 winding counts. Same
thread and unchanged CUDA context required; concurrent use/context switching are
outside this version. No explicit output/report writes, external PTX read only;
SDK compiler caches remain external runtime behavior. Stable executable unchanged.

Fixed correctness gates: sequence0,1,2,0 on each of four meshes with two freshly
created handles, full counts/EDT identical across repeats and exact CPU mask/EDT;
short-query prefix exact with untouched output suffix; zero/over-capacity/nonfinite
queries rejected before output mutation; null destroy idempotent and live destroy
nulls handle; no observed current-boot fault. Build and query failures retained.
No performance or production-lifetime/concurrency certificate from this probe.

Initial API result: five/seven gates PASS, exact CPU occupancy/EDT FAIL. Full
repeats, prefix/suffix, invalid-output preservation and handle cleanup pass.

| mesh | phase0 mask errors | phase1 mask errors | phase2 mask errors |
|---|---|---|---|
| sphere | 25020 | 24294 | 24658 |
| sphere_fine | 25008 | 24286 | 24638 |
| sphere_overlap | 33289 | 33248 | 32869 |
| subdivided_rotated_box | 24558 | 25871 | 24772 |

Both complete runs have identical errors/hashes. Maximum mask errors33289,
maximum EDT differences65142. Evidence: field artifacts/field_rt_api_v1.json
and full arrays. Native process normal, no observed fault. API not certified for
CPU agreement. Next measure host-memory layout before any implementation change.

| module | status | evidence |
|---|---|---|




Contiguous caller result: all seven fixed gates PASS. Eight freshly created
handles,32 full changed-query calls in phase order0/1/2/0,32 short prefix calls,
24 invalid zero/overflow/NaN calls and16 destroy calls (including null repeats).
Every full mask/EDT equals its frozen CPU reference, complete winding arrays
repeat across handles. Short output suffixes and invalid-query sentinels remain
untouched. No current-boot fault observed. No timing claim.

| quantity | initial pointer caller | separate contiguous caller |
|---|---|---|
| maximum mask differences |33289|0|
| maximum EDT differences |65142|0|
| fixed gates passed |5/7|7/7|
| native library SHA |7e019e51887ae23961a288d3bcdafcd4a15cd90ce57fcc7abe1728499362f524|same|

Evidence: field artifacts/field_rt_api_contiguous_v1.json, full arrays and
field_rt_api_contiguous_events.jsonl; mirrored metrics in kernel reports.
Previous failure remains frozen. This certifies changed-query correctness on
these fixtures with fixed capacity and same-thread handles. It does not certify
thread/context switching, device loss, allocation-failure cleanup, arbitrary
pointer validity or production concurrency. Destroy frees owned OptiX/buffer
resources; CUDA runtime primary context lifetime is managed by the runtime.

C ABI buffers MUST be C-contiguous float32 triangle[M,3,3]/ray[R,3] and int32
output[R]; the raw ABI cannot inspect NumPy strides or validate pointer length.
The caller retains arrays through each synchronous call. This requirement is
satisfied explicitly by the separate verified caller. Library performs no explicit
report writes; external compiler caches still apply. Stable single-shot host/PTX
and fixed64 speed negatives remain unchanged. Next library wrapper should enforce
shape/dtype/contiguity and same-thread lifetime before public caller adoption;
performance requires a separate timestamped changed-query measurement.

| module | status | evidence |
|---|---|---|
| `kernel_gen/rt_winding_api_v1/api.h` | SYNTHETIC-ONLY | Separate contiguous caller seven/seven gates PASS,32 full queries exact; earlier pointer-layout negative retained. |
| `kernel_gen/rt_winding_api_v1/api.cpp` | SYNTHETIC-ONLY | Separate contiguous caller seven/seven gates PASS,32 full queries exact; earlier pointer-layout negative retained. |
| `kernel_gen/rt_winding_api_v1/build.sh` | SYNTHETIC-ONLY | Separate contiguous caller seven/seven gates PASS,32 full queries exact; earlier pointer-layout negative retained. |

### Earlier module attempts

Earlier attempt: `kernel_gen/optix_sample_probe.py`; CUDA-ONLY; Preregistered external sample runtime proof; no accepted image runs yet..

Earlier attempt: `kernel_gen/optix_sample_probe.py`; VERIFIED-FRESH; SDK9.1 fails ABI initialization,0 images, first exit1; no second attempt and no observed current-boot fault..

Earlier attempt: `kernel_gen/rt_winding_api_v1/api.h`; SYNTHETIC-ONLY; Build0; initial caller five/seven gates PASS, CPU agreement fails33289 mask samples; not promoted..

Earlier attempt: `kernel_gen/rt_winding_api_v1/api.cpp`; SYNTHETIC-ONLY; Build0; initial caller five/seven gates PASS, CPU agreement fails33289 mask samples; not promoted..

Earlier attempt: `kernel_gen/rt_winding_api_v1/build.sh`; SYNTHETIC-ONLY; Build0; initial caller five/seven gates PASS, CPU agreement fails33289 mask samples; not promoted..

SDK selection: configure OPTIX_ROOT to the externally installed compatible SDK9.0.
SDK9.1 produced unsupported ABI initialization with zero images; that failed
attempt is retained above. SDK9.0 produced two identical images and4/4 passing
gates. No SDK is vendored. Machine-specific installation location belongs in the
private integration handoff, not public source or build defaults.

## C2 whole-selftest adjoints: remaining mechanism and gates

The frozen bounded runtime sweep reports8/8 identical normalized selftest texts,
zero runtime exceptions and7/8 physical passes. Flow control retains exit1/1
and zero printed wake recovery. Normalized text alone does not prove full-array
identity. The existing runtime fixed-order scatter reduction and the derived
440-record tomography bound are retained as measured mechanisms.

Before changing the failed control driver, measure its zero-control loss and
full gradient, the original6000-times-gradient update, and the wake response
to constant controls0,1,8. Re-run the original three finite-difference probes
with unchanged epsilon0.1 and relative threshold0.05 (at least2/3).
Observer gates: two entire dictionaries and gradient hashes identical; all
values finite, positive target deficit, nonzero gradient, original FD gate.
This observer does not claim a successful control optimization or full-suite
physical pass. All GPU work is on isolated Modal workers.

Measured flow scale, Modal NVIDIA L4/Warp1.17.0, two entire outputs identical:

| quantity | measured value |
|---|---|
| zero-control probe | 0.00237007876858115 |
| control1 probe | 0.00242696513887495 |
| control8 probe | 0.00282614436000586 |
| target (unchanged1.6 multiplier) | 0.00379212602972984 |
| control8 recovery | 0.320710572626332 |
| zero-control maximum gradient | 1.92014368849414e-07 |
| original maximum control update | 0.00115208618808538 |
| FD relative errors, steps5/25/50 | 0.863645871083/0.253779738317/0.0172608903433 |

| module | status | evidence |
|---|---|---|
| `certified_kernels/innovation_flow_scale_probe.py` | OWN-GATE-FAIL | L4 and H100 four/five gates pass; identical baseline values, original FD only1/3 meets5%, needs2/3; control8 recovery0.320710573. |

### C2 separate precision and scaled-control driver

The two small derivatives are approximately2e-9 while their finite-difference
perturbations propagate through float32 populations; the large derivative passes.
This motivates testing float64 populations/control/loss arithmetic beside the
frozen float32 solver. Keep the original lattice, direction/weight values,
initial populations,60 steps,jet/probe masks,target multiplier,FD epsilon0.1,
5% FD threshold (at least2/3),control bounds[0,8],maximum coordinate step0.5,
50 optimization iterations and recovery>0.3. No prior physical tolerance changes.

The new driver scales each nonzero gradient coordinate to a bounded sign step,
with up to8 halvings to accept only nonincreasing reported loss. Full-gradient
fixed ordering uses the already measured runtime RUN_TO_RUN reduction; no
claim of a new reduction algorithm. Own gates additionally require finite
arrays,bounded controls and nonincreasing accepted loss. The original module
and its failed selftest remain frozen and unpromoted.

### C2 full-array fixed-order suite, gates before execution

The separate suite uses the exact eight-slot frozen module list, selecting
`lbm/differentiable_flow_control_fixed64` explicitly in the failed flow-control
slot and retaining the other seven reference modules. The original flow module
is not changed or relabelled as passing. Each worker selects RUN_TO_RUN before
module import, with the measured440-record capacity only for tomography.

The observer records every numeric array returned by Warp numpy readback, plus
every primal and gradient array present in Tape.gradients after each complete
backward pass. Metadata includes ordinal,role,shape,dtype,byte count and SHA256
of every contiguous array; memory addresses are excluded. It changes no values.
There is no inference of full-array identity from rounded printed numbers.

Fixed gates: eight selected selftests exit0 twice with no runtime exceptions;
all observed array-byte hash sequences and normalized full-selftest texts match;
every observed numeric value is finite; all eight produce observed arrays; all
six tape-using cases produce backward/gradient records; all ten frozen int64
sites return exactly0 difference. Physics thresholds inside every module stay
unchanged. This is a correctness-only heavy run; record the actual cloud GPU.
The original float32 flow failure, default tomography overflow and previous
text-only suite failures remain in their own rows.

First H100 suite attempt completed0 selftest slots: enabling fixed ordering for
the ancillary frozen float sensitivity diagnostic overflowed its default record
capacity. This is retained as an initial failed attempt, not a determinism result.
The int64 site control now uses its original runtime mode, including ordinary
float diagnostics; only int64 deltas are gated there, exactly as preregistered.
The eight adjoint workers still require RUN_TO_RUN and tomography capacity440.
No integer, array, FD, recovery or returncode gate is changed.

C2 final suite, Modal NVIDIA H100 80GB HBM3, driver580.95.05, Warp1.17.0,
2026-09-13:5/5 suite gates PASS, all eight selected selftests exit0 twice and
all normalized texts and observed array-byte hash sequences match exactly.
The separate flow variant is selected explicitly; the frozen flow module
retains its own failure. Seven other source modules are unchanged.

| suite slot | observed arrays per leg | backward passes per leg | two runs / own physics |
|---|---|---|---|
| `lbm/differentiable_flow_control_fixed64` | 28269 | 51 | exact / PASS |
| `lbm/differentiable_lbm_probe` | 402 | 1 | exact / PASS |
| `lbm/differentiable_fsi_chain` | 84608 | 84 | exact / PASS |
| `lbm/lbm3d_immersed_boundary` | 2 | 0 | exact / PASS |
| `wave_fdtd/diff_wave_3d` | 910 | 1 | exact / PASS |
| `wave_fdtd/diff_wave_substrate` | 1384 | 1 | exact / PASS |
| `wave_fdtd/xray_tomography_sigma` | 5622 | 401 | exact / PASS |
| `wave_fdtd/xray_3d_dda` | 183 | 0 | exact / PASS |

There are121380 array observations and6400352992 observed bytes per leg.
This includes complete numeric readbacks and all tape primal/gradient arrays
after every backward pass, not an assertion about unobserved temporary memory.
All ten frozen int64-site numeric deltas are0; their float diagnostics remain
not gated and keep the original runtime mode. No throughput measurement.

Flow variant: original FD relative errors at steps5/25/50 are
1.45674145833e-6,2.44885264992e-5,1.87891588381e-7 (all below0.05).
Wake recovery0.425832791820 exceeds the unchanged0.3 gate; all6/6 own gates
pass, including original bounds[0,8],maximum step0.5 and accepted loss monotonicity.
The original float32 FD/recovery failures remain in their own rows.

| module | status | evidence |
|---|---|---|
| `lbm/differentiable_flow_control_fixed64.py` | VERIFIED-FRESH | H100 6/6 own gates; FD max2.44885265e-5, wake recovery0.425832792; full observed arrays identical in two suite workers. |
| `certified_kernels/fixed_order_selftests_v1.py` | VERIFIED-FRESH | H100 5/5 suite gates;8/8 selected selftests exit0 twice,121380 observed arrays per leg exact;10 int64 numeric deltas0. |

Reproduce: `PYTHONPATH=src python src/kernel_engine/certified_kernels/fixed_order_selftests_v1.py`.
Compact report: `reports/fixed_order_selftests_v1_h100.json`; complete per-array
records: `reports/fixed_order_selftests_v1_h100.json.gz`, losslessly compressed
and checksummed by the compact report. The compression preserves all records.
Flow details: `reports/differentiable_flow_control_fixed64_h100.json`.

Same-H100 reference control: the frozen float32 flow probe reproduces every L4
value and gradient byte, including failed FD0.863645871083/0.253779738317.
Evidence: `reports/innovation_flow_scale_probe_h100.json`. The precision
comparison therefore also holds on the same GPU as the successful suite.

## C3 LBM export dispatch mechanism, gates before execution

The frozen native export matches every output byte at512 and1024, but the
512 native/Warp bandwidth ratios1.890411/1.924847 exceed the fixed[0.9,1.1]
band. At1024 the ratios1.002739/1.002741 pass. No band is widened.

Before a new export variant, measure the same frozen Warp stream with direct
Python dispatch versus a captured30-kernel graph on the same buffers. Both
sizes retain seed20260912,independent CPU gather oracle,10 warm kernel launches
and30 measured kernel launches. Record event/wall time and graph capture cost
separately; capture is excluded from repeated-dispatch bandwidth on both paths.
Two isolated workers must match all output bytes/hashes and both direct/graph
paths must match the CPU oracle. Every event/wall measurement must be finite
and positive. No speed-gain gate or DRAM peak-bandwidth claim in this observer.

### Vulkan ray-query mechanism before backend design

An isolated cloud loader probe completed twice with identical logs. Its measured
table has1 software device and0 hardware devices: the hardware ICD could not
load because libXext.so.6 was absent. The shader compiler is present (15.1.0).
This is a missing container dependency, not a successful hardware backend.

Before execution of a separate feature probe with that dependency installed:
two full JSON feature tables must be identical, and at least one non-CPU device
must advertise rayQuery, accelerationStructure, bufferDeviceAddress and shaderFloat64.
Record float32 denormal preservation/flush properties. No winding, portability
across vendors, subnormal-equivalence or throughput claim follows from discovery.

| Module | Evidence status | Measured result |
|---|---|---|
| kernel_gen/vulkan_winding_v1/features.cpp | SYNTHETIC-ONLY | 2/2 fresh hardware inventory gates; exact repeat, hardware ray-query/AS/address/float64 available; reports/vulkan_features_hardware.json. Earlier cloud1/2 hardware0 retained. |

| Container | Hardware devices | Software devices | Required features on hardware | Full repeat |
|---|---:|---:|---|---|
| Initial loader dependency missing | 0 | 1 | unavailable | exact |
| Added missing loader library | 0 | 1 | unavailable | exact |

Software-only ray-query support does not pass the hardware gate. A separate
read-only loader/dependency audit will determine whether another library is
missing or the service cannot expose this API; no host device permissions or
sandbox settings will be changed.

### C3 dispatch measurement and matched graph candidate

| Side | Direct event ms, two legs | Graph event ms, two legs | Capture ms, two legs |
|---|---|---|---|
| 512 | 0.0431104 / 0.0439285 | 0.0346112 / 0.0365568 | 1.617265 / 1.595427 |
| 1024 | 0.342699 / 0.343791 | 0.351642 / 0.352563 | 1.721973 / 1.675797 |

All output bytes match the independent CPU oracle and repeat exactly. Graph
submission explains only part of the small-grid difference. Measure native
and Warp with the same captured thirty-kernel protocol before interpreting
remaining code-generation differences. No artificial delay or tolerance change.

| module | status | evidence |
|---|---|---|
| `kernel_gen/innovation_lbm_dispatch_probe.py` | VERIFIED-FRESH | 3/3 observer gates; two L4 workers, direct/graph/CPU bytes exact at both sizes; reports/innovation_lbm_dispatch_probe_l4.json. |

Preregistered candidate gates: unchanged generated forward body; independent
C11 caller built with gcc and CUDA shim with nvcc; exact CPU/Warp/native full
outputs at both sizes; two independent runs byte-identical; finite positive
event/wall times; native/Warp payload bandwidth ratio in unchanged[0.9,1.1]
for each size and each leg. Both paths capture thirty identical kernel launches,
check two replays, warm with ten kernel launches, and time one graph replay.
Capture/instantiation and allocation are outside repeated throughput timing.
Native uses an explicit nonblocking stream required for capture. Existing
export sources and direct-launch baseline remain frozen. This measures
repeated graph dispatch, with no claim about cold-start cost or peak DRAM rate.

Loader follow-up: all direct dynamic-library dependencies resolve, but selecting
only the hardware ICD returns ERROR_INCOMPATIBLE_DRIVER: its proc-address entry
cannot provide vkCreateInstance. Hardware devices remain0. No device-node,
permission, driver-module or sandbox changes are attempted.

### Vulkan winding candidate gates before design/execution

The frozen OptiX seam reads two uint32 counts, then float32 triangle corners and
ray origins, and writes int32 signed winding. It sums sign of the projected
oriented triangle area for every +z hit with t>0, ignoring each hit to continue
traversal. Its five-fixture saved field comparison has0 occupancy/EDT differences.
The Vulkan candidate will preserve that seam in a new directory, use nonopaque
ray-query candidates without confirming a closest hit, and compute orientation
from float64-converted coordinates. Normal execution requires a hardware device;
an explicit --allow-software switch permits a diagnostic CPU implementation.

Seven fixed gates: all child invocations succeed; two full winding outputs exact;
counts match independent double ray-triangle sums; occupancy matches frozen
five-fixture references; corrected float32 EDT matches those references; all
five valid fixtures complete; selected implementation is hardware. The last gate
must remain false for software-only execution, even if correctness passes.
No performance or cross-vendor claim without fresh hardware runs. Compilation
alone is not acceptance. Report float32 subnormal behavior separately; no global
CUDA/Vulkan numeric-equivalence claim follows from these normal-coordinate cases.

### C3 matched graph export result

| Side | Warp event ms, two legs | Native event ms, two legs | Native/Warp payload bandwidth ratio |
|---|---|---|---|
| 512 | 0.0216405 / 0.0217088 | 0.0214016 / 0.0216064 | 1.01116427 / 1.00473937 |
| 1024 | 0.3400704 / 0.3385344 | 0.3382955 / 0.3387051 | 1.00524674 / 0.99949609 |

Modal L4, Warp1.17.0, driver580.95.05: all7/7 gates pass. Every full
output matches the CPU gather oracle, Warp and independent C processes, and
both runs repeat exactly. The generated body was checked against the runtime
source; the C11 host and generated CUDA source remain unchanged. The native
executable runs without Python. Reproduce with
`PYTHONPATH=src python src/kernel_engine/kernel_gen/innovation_lbm_stream_graph_export.py`.
Evidence: `reports/innovation_lbm_stream_graph_export_l4.json`.

Both candidates execute two correctness graph replays before the ten warm
kernel launches. This additional shared preconditioning differs from the
earlier observer, so its timing difference cannot be attributed solely to
host dispatch. These are warm repeated-graph results; the frozen direct-call
comparison still fails its upper bandwidth bound. Useful population bytes
are a read/write floor; cached small-grid traffic is not a DRAM peak fraction.

| module | status | evidence |
|---|---|---|
| `kernel_gen/innovation_lbm_stream_graph_export.py` | VERIFIED-FRESH | 7/7 integration gates, exact full outputs twice, bandwidth ratios0.99949609..1.01116427 inside fixed[0.9,1.1]. |
| `kernel_gen/stream_graph_export_v1/stream_shim.cu` | VERIFIED-FRESH | Same7/7 integration gates through existing C ABI, two graph replays exact; frozen generated body. |
| `kernel_gen/stream_graph_export_v1/build.sh` | VERIFIED-FRESH | gcc C11 caller and nvcc CUDA build execute independently in both passing integration legs. |

| Module | Evidence status | Measured result |
|---|---|---|
| kernel_gen/vulkan_winding_v1/host.cpp | SYNTHETIC-ONLY | 7/7 hardware gates, reports/vulkan_winding_hardware.json; five fixtures/two full runs exact,0 count/occupancy/EDT differences. |
| kernel_gen/vulkan_winding_v1/winding.comp | SYNTHETIC-ONLY | Same7/7 full hardware gates; nonopaque ray-query counts exact; normal-coordinate synthetic scope. |
| kernel_gen/vulkan_winding_v1/probe.py | SYNTHETIC-ONLY | 102162 hardware voxel results across10 runs, full arrays exact/repeated; all30 saved arrays match prior software run too. |

| Fixture | Voxels per run | Signed-count differences | Occupancy differences | EDT differences | Full repeat |
|---|---:|---:|---:|---:|---|
| box | 9261 | 0 | 0 | 0 | exact |
| offset | 10648 | 0 | 0 | 0 | exact |
| reversed | 9261 | 0 | 0 | 0 | exact |
| overlap | 12650 | 0 | 0 | 0 | exact |
| nested | 9261 | 0 | 0 | 0 | exact |

Full arrays: reports/vulkan_winding_software_arrays.npz. The OptiX baseline and
field reference remain unchanged. A follow-up enables the standard validation
layer with the same inputs and adds an observer requiring0 validation errors;
all seven original gates remain unchanged, including the hardware failure.

Validation-layer follow-up: reports/vulkan_winding_validation.json,0 reported
validation errors across10 repeated children. All output hashes match the first
software run; compiled host and shader hashes also match. Hardware gate still
fails. This is standard API validation, not GPU-assisted race instrumentation.

Build with a Vulkan development package and glslang-tools installed:

```sh
c++ -std=c++17 -O2 src/kernel_engine/kernel_gen/vulkan_winding_v1/host.cpp -lvulkan -o winding
 glslangValidator -V --target-env vulkan1.2 src/kernel_engine/kernel_gen/vulkan_winding_v1/winding.comp -o winding.spv
```

Set VULKAN_WINDING_BINARY and VULKAN_WINDING_SHADER to the resulting files and
FIELD_REFERENCE_SOURCE to the frozen field-engine mesh-to-SDF module; run the
probe.py beside the host. Its optional --allow-software flag is diagnostic only.
Normal host invocation rejects software-only availability. Files are written only
to the explicit output argument (host) or reports directory (probe main).

### Vulkan hardware correctness closure

The original seven gates ran unchanged on actual hardware after the cloud ICD
limitation. A bounded externally coordinated window built the frozen host/shader,
read features twice and ran all five fixtures twice. No driver/settings changes.
Kernel journal guards before/after every native child observed no current-boot
fault. This is correctness, not throughput; background idleness was not certified
for timing. Earlier cloud hardware absence and software6/7 remain historical
negative evidence, not a hardware-backend algorithm failure.

| Fixture | Voxels per leg | Signed count differences | Occupancy differences | EDT differences |
|---|---:|---:|---:|---:|
| Box | 9261 | 0 | 0 | 0 |
| Offset | 10648 | 0 | 0 | 0 |
| Reversed | 9261 | 0 | 0 | 0 |
| Overlap | 12650 | 0 | 0 | 0 |
| Nested | 9261 | 0 | 0 | 0 |

All30 full saved arrays also match the prior software Vulkan run. The report's
binary_sha256 names the external journal-guard launcher; native_binary_sha256
names the compiled host. No launcher paths or private input are shipped.
Float32 denormal-preserve/flush-zero feature flags are both0 on the measured
hardware; subnormal behavior and cross-vendor portability remain unverified.
Hardware throughput and direct same-input OptiX comparison remain separate gates;
this does not establish a CUDA-to-Vulkan bandwidth or speed claim.

### Recurrence stability boundary: mechanism before design

Frozen evaluator, scalar gated delta recurrence, T128, q=k=v=1, beta0.15,
initial state0.25, float32. One CPU exploration (not a certificate) measured:

| Gate factor | Chunk | Transition eigenvalue | max output error | Error / original roundoff budget |
|---|---|---|---|---|
| 0.98 | 16 | 0.833000064 | 5.36441803e-7 | 1.25250047 FAIL |
| 0.98 | 32 | 0.833000064 | 5.36441803e-7 | 0.885651578 |
| 0.98 | 64 | 0.833000064 | 5.36441803e-7 | 0.626250236 |
| 0.98 | 128 | 0.833000064 | 5.36441803e-7 | 0.442825789 |
| 1.20 | 16 | 1.02000010 | 0.000122070313 | 2.83650515 FAIL |
| 1.20 | 32 | 1.02000010 | 0.00100708008 | 16.5471242 FAIL |
| 1.20 | 64 | 1.02000010 | 0.155220032 | 1803.39679 FAIL |
| 1.20 | 128 | 1.02000010 | 437.319336 | 3592751.72 FAIL |

At chunk128 the unstable sequential endpoint is90.2519073 and the chunked
endpoint is-347.067444. The cumulative-gate change of variables introduces
large cancelling intermediates even though the exact transition eigenvalue
is only1.02. Stability alone also does not certify a roundoff budget: retain
the stable chunk16 failure. Do not edit the frozen source or its bound.

Preregister a separate conservative classifier requiring supplied finite
transition arrays and non-expansive operator norms at every step, followed
by the actual output/final-state error gates for any accepted evaluation.
Unknown numeric transitions are refused; spectral radius alone is not enough
for non-normal transitions. This is a sufficient structural restriction for
non-expansion, not a theorem that the floating-point chunk algorithm is stable.
Own observer gates: all eight table cells and complete arrays repeat exactly
in two independent CPU workers; unstable chunk128 error exceeds1000 times
chunk16 and its original budget; scalar eigenvalue>1; old syntax-only rule
accepts while the new rule refuses; stable chunk32 retains original output
and state bounds; stable chunk16 remains a reported numerical refusal.

## Recurrence stability boundary: gates before execution

The frozen syntax classifier and WY evaluator remain unchanged. A separate
numerical policy consumes actual finite transition matrices: absent evidence,
nonlinear sources, or any spectral norm above one refuse a chunk recommendation.
This is a conservative non-expansion screen, not a proof of WY conditioning.
Eigenvalues alone do not bound transient amplification for nonnormal operators.
Fixed experiment: seed19, d4, beta0.5, gains0.99/1.01/1.1/1.5,
T32/64/128, chunk min(T,64), frozen fp32 output/state roundoff budgets.
Acceptance: gain1.1 at T64 exceeds the output budget; its transition spectral
radius exceeds one and policy refuses; gain0.99 passes the original output/state
budgets and screen; absent/nonfinite/nonnormal inputs refuse; full arrays repeat
byte-identically across two observations. Report every case, including expansive
cases still inside the error budget. CPU experiment only, no hardware claim.

| Module | Status | Evidence |
|---|---|---|
| `certified_kernels/chunkable_stability_v1.py` | VERIFIED-FRESH | CPU9/9 gates,12 cases twice with exact full-array hashes; gain1.1/T64 output error7.379055e-5 vs budget1.392420e-5 (5.299448x), policy refuses. |

At gain0.99 all three lengths stay within output/state budgets. Gain1.01
is conservatively refused despite passing sampled error budgets; gain1.5/T128
reaches114.851579 times the output budget. A nonnormal radius0.9 matrix is
also refused by its operator norm. These are fixed synthetic observations,
not a claim that all eigenvalues above one cause immediate failure or that a
non-expansive transition guarantees a well-conditioned WY representation.
Evidence: `reports/chunkable_stability_v1.json`;11 focused tests pass.


Fresh CPU measurement:7/7 own gates, two independent workers and all40 full
arrays per leg byte-identical. All eight mechanism rows reproduced; unstable
chunk128 error437.3193359375 exceeds its bound by3592751.7245891755. The new
numerical guard refuses every expansive case before evaluating chunks; the
stable chunk16 case is separately refused by its unchanged measured bound.
Three targeted tests pass, including a non-normal matrix whose spectral radius
is below1 but operator norm exceeds1. No GPU, speed or universal stability claim.

| Module | Status | Evidence |
|---|---|---|
| `certified_kernels/chunkable_roundoff_guard_v2.py` | VERIFIED-FRESH | 7/7 own gates, full two-process repeats; unstable error437.3193359375 refused, stable numerical failure retained. |

## Foreign CUDA certification: gates before execution

Pinned NVIDIA cuda-samples vecAdd, extracted verbatim with original license.
The separate harness compiles using nvcc, without Warp or source rewriting.
G1: full-byte fp32 CPU oracle. G2: two independent allocation/launch/readback
legs have exact full bytes. G3: zero-add/cancellation controls and32 output
canaries for sizes0/1/255/256/257/65539/16777219, including nonmultiple blocks,
signed zero/subnormals. Do not weaken exactness if any case fails.
Run on Modal L4 and A10G. Cross-SKU gate: identical canonical certificates,
including full input/output hashes and pinned source/harness/runner identities.
Compiler/device/timing metadata is separate so the numerical certificate is
reproducible; timing observations are never required to repeat exactly.
Report useful12N bytes per vector-add against measured same-worker D2D
copy bandwidth (8N read+write bytes), both legs. No universal-kernel claim or
minimum throughput fraction. Public source URL/commit/license in provenance.

| Module | Status | Evidence |
|---|---|---|
| `kernel_gen/foreign_cuda_certify_v1.py` | VERIFIED-FRESH | Modal L4/A10,3/3 gates on21 cases/SKU, independent allocation/launch/readback legs exact; includes subnormal/signed-zero/null/bounds controls. |
| `kernel_gen/foreign_cuda_v1/vectorAdd.cu` | VERIFIED-FRESH | Same exact21-case gates on both SKUs; upstream kernel retained verbatim, source identity in provenance.json. |
| `kernel_gen/foreign_cuda_v1/harness.cu` | VERIFIED-FRESH | Built by nvcc on both SKUs; every output/canary checked, two event/copy timing legs recorded. |
| `kernel_gen/foreign_cuda_cross_sku_v1.py` | VERIFIED-FRESH | 5/5 audit gates; canonical certificate hash2916c10c8bc65883b483bade2dd7b5c7b04d2f60f34a5e5ae398f506ab83d303 on both SKUs;6 evidence/corruption tests pass. |

At N16777219, add-case useful bandwidth: L4 236.071/241.207GB/s,
A10 496.193/494.990GB/s. Fractions of measured same-worker D2D copy bandwidth:
L4 1.020954/1.043304; A10 1.041820/1.038356. Copy throughput is a reference
measurement, not a theoretical upper bound; fractions above1 are possible.
Driver580.95.05 on both, CUDA nvcc details in each measurement.json.
Scope is the declared finite fixtures and launch contract, not arbitrary CUDA
or an exhaustive proof of bounds safety. Upstream sample:
https://github.com/NVIDIA/cuda-samples/blob/5443602d89ed99aede2e4b7bf329daddeadb320e/cpp/0_Introduction/vectorAdd/vectorAdd.cu
Reproduce each GPU leg with
`python src/kernel_engine/kernel_gen/foreign_cuda_certify_v1.py`, then compare
saved report directories with `foreign_cuda_cross_sku_v1.py DIR_A DIR_B`.
Evidence: `reports/foreign_cuda_v1/{l4,a10g}/` and
`reports/foreign_cuda_cross_sku_v1.json`. Timing files are intentionally outside
the canonical numerical certificate. No local hardware execution.

## Kernel cloud matrix: gates before execution

Merged item16, E-owned kernel portion only: sequential fresh Modal L4/A10G/H100
captures with Warp1.17.0; local RTX5070 remains unmeasured. Run all eight frozen
fixed-order selftests twice, preserving the fixed64 flow variant and bounded
runtime adjoints. Compare every recorded array, including tape primals/gradients.
Capture actual int64 site outputs and input-array hashes, not just zero within-SKU
deltas. Reuse frozen graph LBM stream worker and C host at512/1024 with the actual
GPU compile architecture; preserve exact CPU/Warp/native oracle and repeat gates.
Original export10% timing criterion is separately reported and may fail; no
performance certification is inferred from numerical determinism.

Matrix requires full eight-slot/ten-site/two-export coverage, all own numerical
gates, matching source identities and strict full-record equality across all
three SKUs. Equal full-byte hashes imply observed max absolute delta0; differing
hashes must name the first differing record, with numeric delta explicitly
unmeasured until raw-array replay. No tolerance, source arithmetic or runtime
mode changes; missing rows/devices cannot silently pass. Produce compressed
complete capsules, compact summaries and a reusable local report auditor.

| Module | Status | Evidence |
|---|---|---|
| `certified_kernels/kernel_matrix_capture_v1.py` | VERIFIED-FRESH | Modal L4/A10/H100 each5/5 numerical capture gates;121380 full-array observations/6400352992 observed bytes per selftest leg, two independent legs each. |
| `certified_kernels/kernel_matrix_report_v1.py` | VERIFIED-FRESH | 5/5 matrix gates;20/20 rows exact for all three pairwise SKU comparisons;8 corruption/coverage controls pass. |
| `scripts/kernel_matrix_modal_v1.py` | VERIFIED-FRESH | Sequential transport collector controls2/2, including incomplete-report rejection and numeric-failure exit; invokes the hardware-verified capture/audit commands. Scheduler not installed. |

Fresh cloud runs,2026-09-13, Warp1.17.0/driver580.95.05:
L4(sm89), A10(sm86), H10080GBHBM3(sm90). Every one of the eight selftest record
sequences is byte-hash-identical across both independent legs on all three
SKUs. All ten reduction sites have identical actual int64 output arrays and
input-array hashes; this is stronger than comparing only within-SKU zero deltas.
Both native/Warp stream sizes512/1024 match each other and their frozen CPU
oracles exactly across all three GPUs. The20-row matrix therefore reports
observed max absolute delta0 on all60 pairwise row comparisons. These are
observed readback/tape arrays, not unobserved temporary memory or every possible
input. Float-atomic diagnostic controls are retained in capsules and are not
included in the fixed-point claim.

The original graph-export10% bandwidth criterion also passed in each of these
captures; its measured per-leg ratios remain in the capsules. It was not used
to excuse any numerical mismatch. Full capsules and checked summaries:
`reports/kernel_matrix_v1/{l4,a10g,h100}/`. Matrix:
`reports/kernel_matrix_v1/matrix.md` and `matrix.json`; runner source hashes:
`reports/kernel_matrix_v1/runner_sources.json`. Independent direct capsule
comparison confirmed all three pairs of selftest records and raw int64 legs.

Reproduction and one-command sequential runner: `docs/KERNEL_MATRIX.md`.
The new auditor/runner were added after all three capture uploads to preserve
one source snapshot across hardware. Their own source hashes are retained
separately; subsequent complete runs will include the added auditor in each
new, mutually identical capsule manifest.
Local RTX5070 remains unmeasured by this work. Other repository families and an
installed nightly schedule remain outside this E-owned cloud certificate; merged
item16 is not declared complete for the whole fleet.

## RTX 5070 matrix extension: VERIFIED-FRESH

Local capture is split into individual selftest legs, integer sites and native export. Each part requires an external exclusive GPU lock and a bounded process-group timeout. Original cloud sources must match exactly; observer hashes are recorded separately. Numerical gates and strict byte equality are unchanged. No local timing claim is planned.

Fresh RTX 5070 observations pass5/5 matrix gates against the retained L4/A10/H100
capsules:20 rows, six device pairs each,120 exact comparisons with observed delta0.
All16 independently executed selftest legs match the cloud records,121380 array
observations/6400352992 bytes per leg. All10 actual int64 outputs and both native
stream sizes also match, including original numerical and repeat gates.

The local driver was580.178.04 versus cloud580.95.05; Warp1.17.0 and all originally
captured source hashes match. Native compilation used CUDA12.9 for sm120. New
observer scripts live outside the frozen source snapshot and have separate hashes.
Eighteen exclusive local windows each used a150-second process-group limit; longest
observed window50.623seconds. Journal/compute-context guards passed before and after
all windows. Background graphics remained present; no local performance claim.

Evidence: `reports/kernel_matrix_local_v1/` contains split parts, assembled capsule,
checksum, four-device matrix, observer hashes and sanitized window receipts.
CPU reproduction: `python scripts/kernel_matrix_local_report_v1.py`.
GPU reproduction requires externally guarded exclusive windows, each invoking
`scripts/kernel_matrix_local_v1.py slot --slot SLOT --leg 0` (then leg1),
followed by `sites` and `exports`; do not run the part collector outside that guard.
The original three-device capsule and report remain unchanged. Six new acceptance
and corruption controls plus the ten original audit/transport tests pass16/16.
These results cover the declared finite fixtures and observed arrays, not universal
cross-device determinism, other repositories, or an installed nightly scheduler.

## Signed kernel observation certificates: VERIFIED-FRESH

Preregistered gates: exactly20 certificates for the existing matrix rows; all four device identities, original source and runtime checks; Ed25519 signature verification against an independently supplied trusted public key; tamper/refusal controls; fresh Modal re-execution of one certified row with exact input/output hashes and original numerical gates. Selftest input identity refers to the pinned deterministic fixture definition, not retrospectively observed input arrays. Existing finite-observation scope remains unchanged.

Twenty signed row certificates pass full inventory/signature/source/capsule checks.
Two independent complete issuer runs produce byte-identical bundles. Eight focused
signing/tamper controls pass. The new ModalL4 replay passes7/7 gates: trusted signature
and retained evidence, original source identity, runtime identity, original numerical
gates, independent repeat, recorded input identity and certified output identity.
This fresh replay covers export:512 only; other rows retain their prior four-device
measurements. Signature trust and the selftest fixture-definition input limitation
are explicit in `docs/KERNEL_CERTIFICATES.md`. That document attaches one certificate
to each of the20 VERIFIED-FRESH matrix rows; they are not20 distinct modules.

## Floating-point provenance catalogue: VERIFIED-FRESH

`reports/kernel_float_provenance_v1/catalogue.json`:3/3 observation gates;20rows/120pair comparisons, zero differing records. FMA/transcendental/reduction-order categories are empty because no operation difference was observed within this certified kernel fixture scope. Float-atomic diagnostic controls and motion/photon families are excluded. No universal operation-portability claim. The catalogue refuses invented classification: a future mismatch remains unresolved until frozen inputs and intermediate operations are replayed. Two negative/coverage controls pass. Reproduce on CPU with `python scripts/kernel_float_provenance_v1.py`.

## Complete signed-certificate replay: VERIFIED-FRESH

Fresh ModalL4 capture must execute both independent legs of all8selftests,10int64 sites and2native sizes. Four gates require complete20-row inventory, original source identity, all original numerical/repeat capture gates and signed input/output/runtime equality for every row. Driver changes are recorded; no performance acceptance or all-repository coverage claim.

Fresh ModalL4/Warp1.17.0/driver580.95.05 complete replay passes4/4 main gates and
all80 per-row checks across20rows. Sixteen independent selftest legs preserve
121380array observations/6400352992observed bytes perleg;10actualint64 sites and
both native/Warp export sizes also reproduce the signed input/output identities.
All original numerical and within-device repeat gates pass. Child runtime317.1s
includes certificate audit and capture; it is not a kernel performance measure.

Ten focused controls cover altered source/input/output, device runtime, missing
coverage, failed gates and stale-success removal. The final runner adds the latter
failure-path guard after the Modal upload; exact executed source is retained in
`reports/kernel_certificate_replay_v2/executed_runner_source.txt`. Its successful
numerical path is independently re-audited by the final verifier on CPU. No private
signing key is uploaded and no local GPU is used. Existing signed certificates and
numerical baselines remain unchanged. Reproduction and public-key trust requirements
are in `docs/KERNEL_CERTIFICATES.md`. This is complete coverage of the20certificate
rows, not all VERIFIED-FRESH entries or all repositories in queued item27.

## Verification entrypoint readiness: OWN-GATE-FAIL

| Module | Status | Decisive observation |
|---|---|---|
| `scripts/verification_inventory_v1.py` | OWN-GATE-FAIL | 288 VERIFIED-FRESH declarations resolve without duplicate aliases; 3 explicit recipes, 285 still unmapped. Inventory performs no numerical execution. |
| `scripts/verify_declared_v1.py` | OWN-GATE-FAIL | Full repository coverage remains incomplete. Selected CPU scope: 2/2 recipes pass twice with exact numerical report hashes; seven runner/inventory controls pass. Strict make verify refuses incomplete coverage before execution. |

Commands, evidence schema and CPU/Modal scope: `docs/VERIFICATION.md`.
