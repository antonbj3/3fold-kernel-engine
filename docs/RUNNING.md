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
without a device, or its CPU fallback exceeds the time budget); last verified in the platform.
SYNTHETIC-ONLY = the method was demonstrated on private input, and it ships with a synthetic input and a test.

| module | status |
| --- | --- |
| src/kernel_engine/_vendor/goal_oriented_culling.py | VERIFIED-FRESH |
| src/kernel_engine/_vendor/lastfalt_v1_fem.py | VERIFIED-FRESH |
| src/kernel_engine/_vendor/render_match_scaffold.py | SYNTHETIC-ONLY |
| src/kernel_engine/_vendor/report_sigma.py | VERIFIED-FRESH |
| src/kernel_engine/_vendor/uq.py | VERIFIED-FRESH |
| src/kernel_engine/allocation/d_fpga_bitwidth_waterfilling.py | VERIFIED-FRESH |
| src/kernel_engine/allocation/d_goal_derived_representation.py | VERIFIED-FRESH |
| src/kernel_engine/allocation/d_pair_rep_waterfill_goalderived.py | VERIFIED-FRESH |
| src/kernel_engine/allocation/d_poxel_waterfilling_unification.py | VERIFIED-FRESH |
| src/kernel_engine/allocation/d_self_tuning_sensitivity_kernel.py | VERIFIED-FRESH |
| src/kernel_engine/allocation/fpga_bitwidth_waterfilling_is_precision_floor_allocation_marginal_modes_need_more_bits.py | VERIFIED-FRESH |
| src/kernel_engine/allocation/l_phase1_quant_x_cert_waterfill.py | VERIFIED-FRESH |
| src/kernel_engine/allocation/l_waterfill_deploygap.py | VERIFIED-FRESH |
| src/kernel_engine/allocation/l_waterfill_sufficiency.py | VERIFIED-FRESH |
| src/kernel_engine/allocation/price_vector_waterfilling_multicommodity_capacity_allocation.py | VERIFIED-FRESH |
| src/kernel_engine/allocation/probe_fair_vs_waterfill_knobs.py | CUDA-ONLY |
| src/kernel_engine/allocation/reservation_waterfill_survival.py | VERIFIED-FRESH |
| src/kernel_engine/allocation/sigmin_price_vector_waterfilling.py | VERIFIED-FRESH |
| src/kernel_engine/allocation/sigmin_waterfill_bstar_stop.py | VERIFIED-FRESH |
| src/kernel_engine/amr_poisson/amr_octree_fv.py | VERIFIED-FRESH |
| src/kernel_engine/amr_poisson/amr_sigma_scaling.py | VERIFIED-FRESH |
| src/kernel_engine/amr_poisson/lbm_poisson.py | VERIFIED-FRESH |
| src/kernel_engine/amr_poisson/lightning_dbm.py | CUDA-ONLY |
| src/kernel_engine/amr_poisson/poisson_dispatch.py | VERIFIED-FRESH |
| src/kernel_engine/certified_kernels/d_1c_iv_end_to_end_real_cuda_cert.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/d_avbd_gpu_certified_roofline.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/d_coupled_knob_ordering_advantage_real_cuda.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/d_cuda_transcendental_parity.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/apriori_requirement_cert_on_real_kernelbench.py | VERIFIED-FRESH |
| src/kernel_engine/certified_kernels/kernelbench_addressing_census_provenance_gate_absent.py | VERIFIED-FRESH |
| src/kernel_engine/certified_kernels/d_ensemble_coexecution_cert_contended_roofline.py | VERIFIED-FRESH |
| src/kernel_engine/certified_kernels/d_r2_r4_dissociation_real_cuda.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/d_roofline_scene_eye.py | VERIFIED-FRESH |
| src/kernel_engine/certified_kernels/d_wave92_B_friction_gate_roofline.py | VERIFIED-FRESH |
| src/kernel_engine/certified_kernels/diag_fp16_bandwidth.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/fem_sass_roofline.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/l_phase0_c2_roofline.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/probe_cuda_machinery_port_first_node.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/probe_rung3_warp_scheduler.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/probe_tensorcore_precision_rung.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/u_compute_twin_roofline_ceiling_v1.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/u_h5_thermal_roofline_twin_state.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/u_v870_roofline_ceiling_long_window_sustained_throttle_recheck.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/u_v874_roofline_soft_knee_sweep_tunable_arithmetic_intensity.py | CUDA-ONLY |
| src/kernel_engine/certified_loop/d_1c_v_autonomy_loop_end_to_end_real_cuda.py | CUDA-ONLY |
| src/kernel_engine/certified_loop/d_cuda_scene_eyes_determinism_real_gpu.py | VERIFIED-FRESH |
| src/kernel_engine/certified_loop/d_lbm_soa_certified_roofline_close.py | CUDA-ONLY |
| src/kernel_engine/certified_loop/d_privatized_reduction_close_abstain.py | VERIFIED-FRESH |
| src/kernel_engine/certified_loop/d_vulkan_port_certified_kernel.py | CUDA-ONLY |
| src/kernel_engine/euler_hllc/axisym_ns_solver.py | VERIFIED-FRESH |
| src/kernel_engine/euler_hllc/detonation_cellular.py | VERIFIED-FRESH |
| src/kernel_engine/euler_hllc/detonation_znd.py | VERIFIED-FRESH |
| src/kernel_engine/euler_hllc/gas_flow_engine.py | VERIFIED-FRESH |
| src/kernel_engine/euler_hllc/goc_baseline_honesty.py | VERIFIED-FRESH |
| src/kernel_engine/euler_hllc/jeans_instability.py | VERIFIED-FRESH |
| src/kernel_engine/euler_hllc/pa_adjoint_vs_cheap_goal.py | VERIFIED-FRESH |
| src/kernel_engine/euler_hllc/pa_adjoint_vs_cheap_proxies.py | VERIFIED-FRESH |
| src/kernel_engine/euler_hllc/phenomenon_registry.py | VERIFIED-FRESH |
| src/kernel_engine/euler_hllc/sigma_solver_routing.py | VERIFIED-FRESH |
| src/kernel_engine/euler_hllc/substrate_compose.py | CUDA-ONLY |
| src/kernel_engine/euler_hllc/traffic_flow_lwr.py | VERIFIED-FRESH |
| src/kernel_engine/fem/buckling_euler_column.py | VERIFIED-FRESH |
| src/kernel_engine/fem/cad_kirsch_mesh.py | VERIFIED-FRESH |
| src/kernel_engine/fem/cad_to_femmesh.py | VERIFIED-FRESH |
| src/kernel_engine/fem/cad_to_tetmesh.py | VERIFIED-FRESH |
| src/kernel_engine/fem/euler_buckling.py | VERIFIED-FRESH |
| src/kernel_engine/fem/fatigue_life.py | VERIFIED-FRESH |
| src/kernel_engine/fem/fem3d_elasticity.py | VERIFIED-FRESH |
| src/kernel_engine/fem/fem3d_modal.py | VERIFIED-FRESH |
| src/kernel_engine/fem/fem3d_orthotropic.py | VERIFIED-FRESH |
| src/kernel_engine/fem/fem3d_thermoelastic.py | VERIFIED-FRESH |
| src/kernel_engine/fem/fracture_lefm.py | VERIFIED-FRESH |
| src/kernel_engine/fem/spectral_fatigue.py | VERIFIED-FRESH |
| src/kernel_engine/fem/viscoelasticity_creep.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_acoustic_design.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_acoustic_modal.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_adjoint_arbitrary.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_cad_elasticity.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_compliant_inverter.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_compute_router.py | CUDA-ONLY |
| src/kernel_engine/fem/warpfem_coupled_adjoint.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_design_gradient.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_design_objective.py | CUDA-ONLY |
| src/kernel_engine/fem/warpfem_elasticity_validate.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_em_magnetostatics.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_field_surrogate.py | CUDA-ONLY |
| src/kernel_engine/fem/warpfem_field_surrogate_fallbevis_v1.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_kirsch.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_mms_3d.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_mms_cad.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_mms_cad3d.py | CUDA-ONLY |
| src/kernel_engine/fem/warpfem_mms_elasticity.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_modal.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_navier_stokes.py | CUDA-ONLY |
| src/kernel_engine/fem/warpfem_stokes_poiseuille.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_stress3d.py | CUDA-ONLY |
| src/kernel_engine/fem/warpfem_surrogate.py | CUDA-ONLY |
| src/kernel_engine/fem/warpfem_thermoelastic.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_topopt.py | CUDA-ONLY |
| src/kernel_engine/fem/warpfem_transient_heat_energy.py | VERIFIED-FRESH |
| src/kernel_engine/fem/warpfem_transient_structural.py | VERIFIED-FRESH |
| src/kernel_engine/kernel_gen/d_crossbackend_kernel_port_verify.py | CUDA-ONLY |
| src/kernel_engine/kernel_gen/d_l1_kernel_certvec_compose.py | VERIFIED-FRESH |
| src/kernel_engine/kernel_variants/kernelvarv_v1_f2_matvec.py | CUDA-ONLY |
| src/kernel_engine/kernel_variants/kernelvarv_v1_f4_csg.py | CUDA-ONLY |
| src/kernel_engine/kernel_variants/kernelvarv_v2_f4_closing.py | SYNTHETIC-ONLY |
| src/kernel_engine/lbm/acoustic_streaming.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/cloud_morphology.py | CUDA-ONLY |
| src/kernel_engine/lbm/d_certvector_on_real_lbm.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/d_integration_stitch_lbm_contact_scenario_cert.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/d_per_variable_precision_cert_lbm.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/d_static_deployment_gate_real_lbm.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/differentiable_lbm_probe.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/double_diffusive_lbm.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/excitable_media_fhn.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/gpu_lbm_luftflode_v1.py | CUDA-ONLY |
| src/kernel_engine/lbm/gpu_lbm_utilization_cell.py | CUDA-ONLY |
| src/kernel_engine/lbm/hartmann_mhd_lbm.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/lbm3d_gpu.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/lbm3d_immersed_boundary.py | CUDA-ONLY |
| src/kernel_engine/lbm/lbm3d_poiseuille.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/lbm_aero_v0.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/lbm_compressible_boundary.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/lbm_fsi_bouzidi.py | CUDA-ONLY |
| src/kernel_engine/lbm/lbm_fsi_gpu.py | CUDA-ONLY |
| src/kernel_engine/lbm/lbm_fsi_viv.py | CUDA-ONLY |
| src/kernel_engine/lbm/lbm_gpu_fast.py | CUDA-ONLY |
| src/kernel_engine/lbm/lbm_gpu_fp16.py | CUDA-ONLY |
| src/kernel_engine/lbm/lbm_gpu_fp16_half2.py | CUDA-ONLY |
| src/kernel_engine/lbm/lbm_lattice.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/lbm_mach_ceiling.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/lbm_mrt_stability.py | CUDA-ONLY |
| src/kernel_engine/lbm/lbm_voxel_aero.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/lbm_voxel_aero_gpu.py | CUDA-ONLY |
| src/kernel_engine/lbm/magnetic_induction_lattice.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/moist_convection_lbm.py | CUDA-ONLY |
| src/kernel_engine/lbm/probe_lbm_fp16_steady_dither.py | CUDA-ONLY |
| src/kernel_engine/lbm/rayleigh_taylor_lbm.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/reaction_diffusion_turing.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/rte_lbm.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/thermofluid_lbm_rayleigh.py | CUDA-ONLY |
| src/kernel_engine/reductions/d_1c_real_reduction_roofline_torch.py | CUDA-ONLY |
| src/kernel_engine/reductions/probe_kernel_determinism_cert.py | VERIFIED-FRESH |
| src/kernel_engine/reductions/u_v54_gather_law_deterministic_reduction_vs_atomic_int64.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/biot_transient_conduction.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/bohm_sheath_criterion.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/d_nonnormal_thermo_price_of_amplification_probe.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/d_thermo_computing_equals_fusion.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/d_thermo_datahole_law.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/d_thermo_voi_law_multiworld.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/debye_specific_heat.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/diffusion_induced_stress.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/explosion_combustion.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/fizeau_drag.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/fourier_conduction.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/heat_pipe_capillary_limit.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/heat_pipe_sigma_budget.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/heat_pump_cop.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/heat_pump_sigma_budget.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/hopf_reaction_diffusion.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/induction_heating_skin.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/ising_thermo.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/joule_heating_thermal.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/kapitza_acoustic_mismatch.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/p8_combustion_cert_pod_a90.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/p8_combustion_cert_watertight.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/p8_combustion_state_certification.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/p8_delft_conditional_variance.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/p8_delft_flamelet_render_match.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/p8_delft_species_flamelet.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/p8_ecn_combustion_energy.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/p8_flame_mixture_fraction_render_match.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/p8_sandia_extinction_flamelet_breakdown.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/photoelasticity_isochromatics.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/photon_diffusion_escape.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/saffman_delbruck_diffusion.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/sommerfeld_electron_heat.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/soret_thermodiffusion.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/thermocouple_seebeck.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/thermoelectric_seebeck.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/tidal_heating.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/wheatstone_bridge.py | VERIFIED-FRESH |
| src/kernel_engine/thermo/wkb_quantization.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/2jet_sdf_curvature_decouples_v6.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/JxA_contact_param_over_determination_material_physics_leg_lifts_the_motion_regime_null.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/certified_generative_support_placement_worstcase_sigmamin_robust_dfc.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/contact_cert_chi_from_sdf_error.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/contact_dof_cert_identify_or_abstain.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/contact_orientation_gauge_emergence.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/contact_penalty_conditioning_omega_max_side_accuracy_cost_tradeoff_completes_sim_conditioning.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/engine_contact_manifold_rank_twist_selfstress_2point_reduction.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/engine_manifold_reduction_needs_3points_not_2_wrench_span.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/engine_which_3_points_Eoptimal_sigmamin_manifold_not_max_area.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/friction_mu_is_a_sigma_min_null_under_stick_lifted_only_by_slip_events_contact_sim_ready.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/mma_capstone_validity_band_sweep.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/mma_topopt_capstone_stress_constrained_dfc_positive.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/restitution_null_lifted_by_impact_friction_and_restitution_are_decorrelated_contact_nulls.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/so_arm100_contact_sim_cost_from_tropical_sdf_backend_hertz_stiffness_sets_stable_dt_stability_verified.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/stress_constrained_compliance_topopt_dfc_positive.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/stress_objective_topopt_design_for_certifiability_positive.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/topopt_stress_frontier_forced_number_not_honest_label.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/topopt_stress_vs_compliance_design_for_certifiability.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/tropical_sdf_SIMT_tax_cost_model_compute_bound_vs_voxel_bandwidth_bound_crossover_Kstar.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/tropical_sdf_backend_computes_full_v6_contact_geometry_ladder_2jet_curvature_anisotropy_on_so_arm100.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/tropical_sdf_backend_min_of_capsules_beats_voxel_3x_bytes_matched_penetration_so_arm100.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/tropical_sdf_so_arm100_geometry_backend_v2_config_driven_posing_plus_contact_normal_gradient.py | VERIFIED-FRESH |
| src/kernel_engine/tropical_sdf/tropical_sdf_swept_volume_continuous_collision_detection_min_over_trajectory.py | VERIFIED-FRESH |
| src/kernel_engine/warp_gpu/gpu_fracture_determinism_sigma.py | VERIFIED-FRESH |
| src/kernel_engine/warp_gpu/mujoco_warp_bench.py | CUDA-ONLY |
| src/kernel_engine/warp_gpu/rigid2d_primal_vbd_gpu.py | CUDA-ONLY |
| src/kernel_engine/warp_gpu/warp_bodybody_colored_gpu.py | CUDA-ONLY |
| src/kernel_engine/warp_gpu/warp_bodybody_jacobi_gpu.py | CUDA-ONLY |
| src/kernel_engine/warp_gpu/warp_bodybody_scale_gpu.py | CUDA-ONLY |
| src/kernel_engine/warp_gpu/warp_cylinder_roll_gpu.py | CUDA-ONLY |
| src/kernel_engine/warp_gpu/warp_granular_friction.py | CUDA-ONLY |
| src/kernel_engine/warp_gpu/warp_granular_gpu.py | CUDA-ONLY |
| src/kernel_engine/warp_gpu/warp_mesh_roll_gpu.py | CUDA-ONLY |
| src/kernel_engine/warp_gpu/warp_rigid_ramp_gpu.py | CUDA-ONLY |
| src/kernel_engine/warp_gpu/warp_rigid_ramp_gpu_v2.py | CUDA-ONLY |
| src/kernel_engine/warp_gpu/warp_rl_env_gpu.py | CUDA-ONLY |
| src/kernel_engine/wave_fdtd/acoustic_emission.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/acoustic_sigma_renderer.py | SYNTHETIC-ONLY |
| src/kernel_engine/wave_fdtd/acoustic_fdtd.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/diag_acoustic_mode.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/diag_acoustic_seed.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/diag_acoustic_spectrum.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/diff_wave_3d.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/diff_wave_substrate.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/elastodynamics_kache.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/goc_wave_transient.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/optics_coating.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/optics_fdtd.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/persona_design_acoustic.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/persona_design_acoustic_cavity_shape.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/persona_design_acoustic_inverse.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/persona_design_acoustic_shape_adjoint.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/persona_design_lbm_acoustic.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/s1_acoustic_metamaterial_bandgap.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/thermoacoustic_rijke_dde.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/vibroacoustic.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/wave_fdtd_3d.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/wave_fdtd_kache.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/wave_fdtd_verify.py | VERIFIED-FRESH |
| src/kernel_engine/certified_kernels/aa_micro_bench.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/d_energy_exponent_substrate_invariant.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/gpu_duty_torch_v1.py | VERIFIED-FRESH |
| src/kernel_engine/certified_kernels/module_const_launch_tune.py | VERIFIED-FRESH |
| src/kernel_engine/certified_kernels/morton_3d_stencil.py | VERIFIED-FRESH |
| src/kernel_engine/certified_kernels/morton_sparse_gather.py | VERIFIED-FRESH |
| src/kernel_engine/certified_kernels/probe_compute_ladder_descent.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/probe_exclusive_sweep_block.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/probe_l2_inband_endgame.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/probe_sync_density_victim_model.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/probe_sync_penalty_vs_priority.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/probe_timeslice_dma_fartail.py | CUDA-ONLY |
| src/kernel_engine/certified_kernels/ser_fracture_compaction.py | CUDA-ONLY |
| src/kernel_engine/euler_hllc/lubrication_reynolds_bearing.py | VERIFIED-FRESH |
| src/kernel_engine/fem/fsi_added_mass.py | VERIFIED-FRESH |
| src/kernel_engine/fem/fsi_pipe_flutter.py | VERIFIED-FRESH |
| src/kernel_engine/fem/neuber_notch_plasticity.py | VERIFIED-FRESH |
| src/kernel_engine/fem/plasticity_3d_j2.py | VERIFIED-FRESH |
| src/kernel_engine/fem/plasticity_return_mapping.py | VERIFIED-FRESH |
| src/kernel_engine/fem/thermal_buckling.py | VERIFIED-FRESH |
| src/kernel_engine/fem/viscoelastic_preload_relaxation.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/coupled_design_aero_struct.py | CUDA-ONLY |
| src/kernel_engine/lbm/differentiable_flow_control.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/differentiable_fsi_chain.py | VERIFIED-FRESH |
| src/kernel_engine/lbm/g18_cfd_nilss_prereq_wake_chaos.py | CUDA-ONLY |
| src/kernel_engine/lbm/g19_forced_2d_wake_nilss_prereq.py | CUDA-ONLY |
| src/kernel_engine/lbm/g20_3d_wake_chaos_nilss_prereq.py | CUDA-ONLY |
| src/kernel_engine/lbm/g21_3d_wake_chaos_highRe.py | CUDA-ONLY |
| src/kernel_engine/lbm/probe_kam_resonance_dither_strides.py | CUDA-ONLY |
| src/kernel_engine/reductions/d_1c_iv_best_in_class_float4.py | CUDA-ONLY |
| src/kernel_engine/reductions/det_accumulation_probe.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/coupled_multiphysics_calibration.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/goc_wave_verify.py | VERIFIED-FRESH |
| src/kernel_engine/wave_fdtd/sigma_guided_fwi.py | CUDA-ONLY |
| src/kernel_engine/wave_fdtd/twin_calibration_multisource.py | VERIFIED-FRESH |

278 modules: 198 VERIFIED-FRESH, 77 CUDA-ONLY, 3 SYNTHETIC-ONLY.


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
  computed a-priori; histogram worst case 256x its uniform typical.
- `kernelbench_addressing_census_provenance_gate_absent`: 270 kernels (100/100/50/20), input-keyed write count 0 on all
  three paths; the injected bincount control routes to `distributional` and abstains.

`acoustic_sigma_renderer` stays SYNTHETIC-ONLY: the archive copy holds only the three TRAIN bearing codes, so the module's
own data-availability gate would refuse the real run with

    RuntimeError: DEGENERATE CALIB SET -- missing bearing .mat files for ['K002', 'KA03', 'KI03'] (CALIB collected 0
    samples). Re-fetch data/paderborn_kat_severity/ before trusting this script's output.

It ships with the stand-in signal generator instead; no tolerance in it was changed.
