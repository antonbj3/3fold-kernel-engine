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

## Jargon table, additions

| internal | public |
| --- | --- |
| kache (the one-thread-per-voxel/particle thesis) | one thread per voxel / per particle |
| reality-grind | real-world gate |
| fallbevis | planted-fault test |
| fantom / fantom-claim | false positive / overclaim |
| leveransyta | delivery surface |
| flaskhals | bottleneck |
| kolv | piston |
| FAS I / FAS J | the combustion stage / the gas-flow stage |
| waveNNN (a numbered predecessor cell) | cell NNN |
| verdikt / grind / ankare | verdict / gate / anchor |

## Files copied in the second extraction pass (destination -> source path relative to the platform repository root)

| in this repository | source |
| --- | --- |
| src/kernel_engine/_vendor/goal_oriented_culling.py | scripts/physics_exp/goal_oriented_culling.py |
| src/kernel_engine/_vendor/report_sigma.py | scripts/physics_exp/probes/report_sigma.py |
| src/kernel_engine/_vendor/uq.py | src/cad2simready/uq.py |
| src/kernel_engine/allocation/d_pair_rep_waterfill_goalderived.py | scripts/physics_exp/d_pair_rep_waterfill_goalderived.py |
| src/kernel_engine/allocation/fpga_bitwidth_waterfilling_is_precision_floor_allocation_marginal_modes_need_more_bits.py | scripts/physics_exp/fpga_bitwidth_waterfilling_is_precision_floor_allocation_marginal_modes_need_more_bits.py |
| src/kernel_engine/allocation/l_phase1_quant_x_cert_waterfill.py | scripts/physics_exp/l_phase1_quant_x_cert_waterfill.py |
| src/kernel_engine/allocation/l_waterfill_deploygap.py | scripts/modal/l_waterfill_deploygap.py |
| src/kernel_engine/allocation/l_waterfill_sufficiency.py | scripts/modal/l_waterfill_sufficiency.py |
| src/kernel_engine/allocation/price_vector_waterfilling_multicommodity_capacity_allocation.py | scripts/physics_exp/price_vector_waterfilling_multicommodity_capacity_allocation.py |
| src/kernel_engine/allocation/probe_fair_vs_waterfill_knobs.py | scripts/physics_exp/probes/probe_fair_vs_waterfill_knobs.py |
| src/kernel_engine/allocation/reservation_waterfill_survival.py | scripts/physics_exp/reservation_waterfill_survival.py |
| src/kernel_engine/allocation/sigmin_price_vector_waterfilling.py | scripts/physics_exp/sigmin_price_vector_waterfilling.py |
| src/kernel_engine/allocation/sigmin_waterfill_bstar_stop.py | scripts/physics_exp/sigmin_waterfill_bstar_stop.py |
| src/kernel_engine/amr_poisson/amr_octree_fv.py | scripts/physics_exp/amr_octree_fv.py |
| src/kernel_engine/amr_poisson/amr_sigma_scaling.py | scripts/physics_exp/amr_sigma_scaling.py |
| src/kernel_engine/amr_poisson/lbm_poisson.py | scripts/physics_exp/lbm_poisson.py |
| src/kernel_engine/amr_poisson/lightning_dbm.py | scripts/physics_exp/lightning_dbm.py |
| src/kernel_engine/amr_poisson/poisson_dispatch.py | scripts/physics_exp/poisson_dispatch.py |
| src/kernel_engine/certified_kernels/d_1c_iv_end_to_end_real_cuda_cert.py | scripts/physics_exp/d_1c_iv_end_to_end_real_cuda_cert.py |
| src/kernel_engine/certified_kernels/d_avbd_gpu_certified_roofline.py | scripts/physics_exp/d_avbd_gpu_certified_roofline.py |
| src/kernel_engine/certified_kernels/d_coupled_knob_ordering_advantage_real_cuda.py | scripts/physics_exp/d_coupled_knob_ordering_advantage_real_cuda.py |
| src/kernel_engine/certified_kernels/d_cuda_transcendental_parity.py | scripts/physics_exp/d_cuda_transcendental_parity.py |
| src/kernel_engine/certified_kernels/d_ensemble_coexecution_cert_contended_roofline.py | scripts/physics_exp/d_ensemble_coexecution_cert_contended_roofline.py |
| src/kernel_engine/certified_kernels/d_r2_r4_dissociation_real_cuda.py | scripts/physics_exp/d_r2_r4_dissociation_real_cuda.py |
| src/kernel_engine/certified_kernels/d_roofline_scene_eye.py | scripts/physics_exp/d_roofline_scene_eye.py |
| src/kernel_engine/certified_kernels/d_wave92_B_friction_gate_roofline.py | scripts/physics_exp/d_wave92_B_friction_gate_roofline.py |
| src/kernel_engine/certified_kernels/diag_fp16_bandwidth.py | scripts/physics_exp/diag_fp16_bandwidth.py |
| src/kernel_engine/certified_kernels/fem_sass_roofline.py | scripts/physics_exp/fem_sass_roofline.py |
| src/kernel_engine/certified_kernels/l_phase0_c2_roofline.py | scripts/physics_exp/l_phase0_c2_roofline.py |
| src/kernel_engine/certified_kernels/probe_cuda_machinery_port_first_node.py | scripts/physics_exp/probes/probe_cuda_machinery_port_first_node.py |
| src/kernel_engine/certified_kernels/probe_rung3_warp_scheduler.py | scripts/physics_exp/probes/probe_rung3_warp_scheduler.py |
| src/kernel_engine/certified_kernels/probe_tensorcore_precision_rung.py | scripts/physics_exp/probes/probe_tensorcore_precision_rung.py |
| src/kernel_engine/certified_kernels/u_compute_twin_roofline_ceiling_v1.py | scripts/physics_exp/u_compute_twin_roofline_ceiling_v1.py |
| src/kernel_engine/certified_kernels/u_h5_thermal_roofline_twin_state.py | scripts/physics_exp/u_h5_thermal_roofline_twin_state.py |
| src/kernel_engine/certified_kernels/u_v870_roofline_ceiling_long_window_sustained_throttle_recheck.py | scripts/physics_exp/u_v870_roofline_ceiling_long_window_sustained_throttle_recheck.py |
| src/kernel_engine/certified_kernels/u_v874_roofline_soft_knee_sweep_tunable_arithmetic_intensity.py | scripts/physics_exp/u_v874_roofline_soft_knee_sweep_tunable_arithmetic_intensity.py |
| src/kernel_engine/euler_hllc/axisym_ns_solver.py | scripts/physics_exp/axisym_ns_solver.py |
| src/kernel_engine/euler_hllc/detonation_cellular.py | scripts/physics_exp/detonation_cellular.py |
| src/kernel_engine/euler_hllc/detonation_znd.py | scripts/physics_exp/detonation_znd.py |
| src/kernel_engine/euler_hllc/gas_flow_engine.py | scripts/physics_exp/gas_flow_engine.py |
| src/kernel_engine/euler_hllc/goc_baseline_honesty.py | scripts/physics_exp/goc_baseline_honesty.py |
| src/kernel_engine/euler_hllc/jeans_instability.py | scripts/physics_exp/jeans_instability.py |
| src/kernel_engine/euler_hllc/pa_adjoint_vs_cheap_goal.py | scripts/physics_exp/pa_adjoint_vs_cheap_goal.py |
| src/kernel_engine/euler_hllc/pa_adjoint_vs_cheap_proxies.py | scripts/physics_exp/pa_adjoint_vs_cheap_proxies.py |
| src/kernel_engine/euler_hllc/phenomenon_registry.py | scripts/physics_exp/phenomenon_registry.py |
| src/kernel_engine/euler_hllc/sigma_solver_routing.py | scripts/physics_exp/sigma_solver_routing.py |
| src/kernel_engine/euler_hllc/substrate_compose.py | scripts/physics_exp/substrate_compose.py |
| src/kernel_engine/euler_hllc/traffic_flow_lwr.py | scripts/physics_exp/traffic_flow_lwr.py |
| src/kernel_engine/fem/buckling_euler_column.py | scripts/buckling_euler_column.py |
| src/kernel_engine/fem/cad_kirsch_mesh.py | scripts/cad_kirsch_mesh.py |
| src/kernel_engine/fem/cad_to_femmesh.py | scripts/cad_to_femmesh.py |
| src/kernel_engine/fem/cad_to_tetmesh.py | scripts/cad_to_tetmesh.py |
| src/kernel_engine/fem/euler_buckling.py | scripts/physics_exp/euler_buckling.py |
| src/kernel_engine/fem/fatigue_life.py | scripts/fatigue_life.py |
| src/kernel_engine/fem/fem3d_elasticity.py | scripts/fem3d_elasticity.py |
| src/kernel_engine/fem/fem3d_modal.py | scripts/fem3d_modal.py |
| src/kernel_engine/fem/fem3d_orthotropic.py | scripts/fem3d_orthotropic.py |
| src/kernel_engine/fem/fem3d_thermoelastic.py | scripts/fem3d_thermoelastic.py |
| src/kernel_engine/fem/fracture_lefm.py | scripts/fracture_lefm.py |
| src/kernel_engine/fem/spectral_fatigue.py | scripts/spectral_fatigue.py |
| src/kernel_engine/fem/viscoelasticity_creep.py | scripts/viscoelasticity_creep.py |
| src/kernel_engine/fem/warpfem_acoustic_design.py | scripts/warpfem_acoustic_design.py |
| src/kernel_engine/fem/warpfem_acoustic_modal.py | scripts/warpfem_acoustic_modal.py |
| src/kernel_engine/fem/warpfem_adjoint_arbitrary.py | scripts/warpfem_adjoint_arbitrary.py |
| src/kernel_engine/fem/warpfem_cad_elasticity.py | scripts/warpfem_cad_elasticity.py |
| src/kernel_engine/fem/warpfem_compliant_inverter.py | scripts/warpfem_compliant_inverter.py |
| src/kernel_engine/fem/warpfem_compute_router.py | scripts/warpfem_compute_router.py |
| src/kernel_engine/fem/warpfem_coupled_adjoint.py | scripts/warpfem_coupled_adjoint.py |
| src/kernel_engine/fem/warpfem_design_gradient.py | scripts/warpfem_design_gradient.py |
| src/kernel_engine/fem/warpfem_design_objective.py | scripts/warpfem_design_objective.py |
| src/kernel_engine/fem/warpfem_elasticity_validate.py | scripts/warpfem_elasticity_validate.py |
| src/kernel_engine/fem/warpfem_em_magnetostatics.py | scripts/warpfem_em_magnetostatics.py |
| src/kernel_engine/fem/warpfem_field_surrogate.py | scripts/warpfem_field_surrogate.py |
| src/kernel_engine/fem/warpfem_field_surrogate_fallbevis_v1.py | scripts/gemini_verify/warpfem_field_surrogate_fallbevis_v1.py |
| src/kernel_engine/fem/warpfem_kirsch.py | scripts/warpfem_kirsch.py |
| src/kernel_engine/fem/warpfem_mms_3d.py | scripts/warpfem_mms_3d.py |
| src/kernel_engine/fem/warpfem_mms_cad.py | scripts/warpfem_mms_cad.py |
| src/kernel_engine/fem/warpfem_mms_cad3d.py | scripts/warpfem_mms_cad3d.py |
| src/kernel_engine/fem/warpfem_mms_elasticity.py | scripts/warpfem_mms_elasticity.py |
| src/kernel_engine/fem/warpfem_modal.py | scripts/warpfem_modal.py |
| src/kernel_engine/fem/warpfem_navier_stokes.py | scripts/warpfem_navier_stokes.py |
| src/kernel_engine/fem/warpfem_stokes_poiseuille.py | scripts/warpfem_stokes_poiseuille.py |
| src/kernel_engine/fem/warpfem_stress3d.py | scripts/warpfem_stress3d.py |
| src/kernel_engine/fem/warpfem_surrogate.py | scripts/warpfem_surrogate.py |
| src/kernel_engine/fem/warpfem_thermoelastic.py | scripts/warpfem_thermoelastic.py |
| src/kernel_engine/fem/warpfem_topopt.py | scripts/warpfem_topopt.py |
| src/kernel_engine/fem/warpfem_transient_heat_energy.py | scripts/warpfem_transient_heat_energy.py |
| src/kernel_engine/fem/warpfem_transient_structural.py | scripts/warpfem_transient_structural.py |
| src/kernel_engine/kernel_gen/d_crossbackend_kernel_port_verify.py | scripts/physics_exp/d_crossbackend_kernel_port_verify.py |
| src/kernel_engine/kernel_gen/d_l1_kernel_certvec_compose.py | scripts/physics_exp/d_l1_kernel_certvec_compose.py |
| src/kernel_engine/lbm/acoustic_streaming.py | scripts/physics_exp/acoustic_streaming.py |
| src/kernel_engine/lbm/cloud_morphology.py | scripts/physics_exp/cloud_morphology.py |
| src/kernel_engine/lbm/d_certvector_on_real_lbm.py | scripts/physics_exp/d_certvector_on_real_lbm.py |
| src/kernel_engine/lbm/d_integration_stitch_lbm_contact_scenario_cert.py | scripts/physics_exp/d_integration_stitch_lbm_contact_scenario_cert.py |
| src/kernel_engine/lbm/d_per_variable_precision_cert_lbm.py | scripts/physics_exp/d_per_variable_precision_cert_lbm.py |
| src/kernel_engine/lbm/d_static_deployment_gate_real_lbm.py | scripts/physics_exp/d_static_deployment_gate_real_lbm.py |
| src/kernel_engine/lbm/differentiable_lbm_probe.py | scripts/physics_exp/differentiable_lbm_probe.py |
| src/kernel_engine/lbm/double_diffusive_lbm.py | scripts/physics_exp/double_diffusive_lbm.py |
| src/kernel_engine/lbm/excitable_media_fhn.py | scripts/physics_exp/excitable_media_fhn.py |
| src/kernel_engine/lbm/gpu_lbm_utilization_cell.py | scripts/physics_exp/gpu_lbm_utilization_cell.py |
| src/kernel_engine/lbm/hartmann_mhd_lbm.py | scripts/physics_exp/hartmann_mhd_lbm.py |
| src/kernel_engine/lbm/lbm3d_immersed_boundary.py | scripts/physics_exp/lbm3d_immersed_boundary.py |
| src/kernel_engine/lbm/lbm_aero_v0.py | scripts/physics_exp/lbm_aero_v0.py |
| src/kernel_engine/lbm/lbm_compressible_boundary.py | scripts/physics_exp/lbm_compressible_boundary.py |
| src/kernel_engine/lbm/lbm_fsi_bouzidi.py | scripts/physics_exp/lbm_fsi_bouzidi.py |
| src/kernel_engine/lbm/lbm_fsi_gpu.py | scripts/physics_exp/lbm_fsi_gpu.py |
| src/kernel_engine/lbm/lbm_fsi_viv.py | scripts/physics_exp/lbm_fsi_viv.py |
| src/kernel_engine/lbm/lbm_gpu_fp16.py | scripts/physics_exp/lbm_gpu_fp16.py |
| src/kernel_engine/lbm/lbm_gpu_fp16_half2.py | scripts/physics_exp/lbm_gpu_fp16_half2.py |
| src/kernel_engine/lbm/lbm_mach_ceiling.py | scripts/physics_exp/lbm_mach_ceiling.py |
| src/kernel_engine/lbm/lbm_mrt_stability.py | scripts/physics_exp/lbm_mrt_stability.py |
| src/kernel_engine/lbm/lbm_voxel_aero.py | scripts/physics_exp/lbm_voxel_aero.py |
| src/kernel_engine/lbm/lbm_voxel_aero_gpu.py | scripts/physics_exp/lbm_voxel_aero_gpu.py |
| src/kernel_engine/lbm/magnetic_induction_lattice.py | scripts/physics_exp/magnetic_induction_lattice.py |
| src/kernel_engine/lbm/moist_convection_lbm.py | scripts/physics_exp/moist_convection_lbm.py |
| src/kernel_engine/lbm/probe_lbm_fp16_steady_dither.py | scripts/physics_exp/probes/probe_lbm_fp16_steady_dither.py |
| src/kernel_engine/lbm/rayleigh_taylor_lbm.py | scripts/physics_exp/rayleigh_taylor_lbm.py |
| src/kernel_engine/lbm/reaction_diffusion_turing.py | scripts/physics_exp/reaction_diffusion_turing.py |
| src/kernel_engine/lbm/rte_lbm.py | scripts/physics_exp/rte_lbm.py |
| src/kernel_engine/lbm/thermofluid_lbm_rayleigh.py | scripts/physics_exp/thermofluid_lbm_rayleigh.py |
| src/kernel_engine/reductions/d_1c_real_reduction_roofline_torch.py | scripts/physics_exp/d_1c_real_reduction_roofline_torch.py |
| src/kernel_engine/reductions/probe_kernel_determinism_cert.py | scripts/physics_exp/probes/probe_kernel_determinism_cert.py |
| src/kernel_engine/reductions/u_v54_gather_law_deterministic_reduction_vs_atomic_int64.py | scripts/physics_exp/u_v54_gather_law_deterministic_reduction_vs_atomic_int64.py |
| src/kernel_engine/thermo/biot_transient_conduction.py | scripts/physics_exp/biot_transient_conduction.py |
| src/kernel_engine/thermo/bohm_sheath_criterion.py | scripts/physics_exp/bohm_sheath_criterion.py |
| src/kernel_engine/thermo/d_nonnormal_thermo_price_of_amplification_probe.py | scripts/physics_exp/d_nonnormal_thermo_price_of_amplification_probe.py |
| src/kernel_engine/thermo/d_thermo_computing_equals_fusion.py | scripts/physics_exp/d_thermo_computing_equals_fusion.py |
| src/kernel_engine/thermo/d_thermo_datahole_law.py | scripts/physics_exp/d_thermo_datahole_law.py |
| src/kernel_engine/thermo/d_thermo_voi_law_multiworld.py | scripts/physics_exp/d_thermo_voi_law_multiworld.py |
| src/kernel_engine/thermo/debye_specific_heat.py | scripts/physics_exp/debye_specific_heat.py |
| src/kernel_engine/thermo/diffusion_induced_stress.py | scripts/physics_exp/diffusion_induced_stress.py |
| src/kernel_engine/thermo/explosion_combustion.py | scripts/physics_exp/explosion_combustion.py |
| src/kernel_engine/thermo/fizeau_drag.py | scripts/physics_exp/fizeau_drag.py |
| src/kernel_engine/thermo/fourier_conduction.py | scripts/physics_exp/fourier_conduction.py |
| src/kernel_engine/thermo/heat_pipe_capillary_limit.py | scripts/physics_exp/heat_pipe_capillary_limit.py |
| src/kernel_engine/thermo/heat_pipe_sigma_budget.py | scripts/physics_exp/heat_pipe_sigma_budget.py |
| src/kernel_engine/thermo/heat_pump_cop.py | scripts/physics_exp/heat_pump_cop.py |
| src/kernel_engine/thermo/heat_pump_sigma_budget.py | scripts/physics_exp/heat_pump_sigma_budget.py |
| src/kernel_engine/thermo/hopf_reaction_diffusion.py | scripts/physics_exp/hopf_reaction_diffusion.py |
| src/kernel_engine/thermo/induction_heating_skin.py | scripts/physics_exp/induction_heating_skin.py |
| src/kernel_engine/thermo/ising_thermo.py | scripts/physics_exp/ising_thermo.py |
| src/kernel_engine/thermo/joule_heating_thermal.py | scripts/physics_exp/joule_heating_thermal.py |
| src/kernel_engine/thermo/kapitza_acoustic_mismatch.py | scripts/physics_exp/kapitza_acoustic_mismatch.py |
| src/kernel_engine/thermo/photoelasticity_isochromatics.py | scripts/physics_exp/photoelasticity_isochromatics.py |
| src/kernel_engine/thermo/photon_diffusion_escape.py | scripts/physics_exp/photon_diffusion_escape.py |
| src/kernel_engine/thermo/saffman_delbruck_diffusion.py | scripts/physics_exp/saffman_delbruck_diffusion.py |
| src/kernel_engine/thermo/sommerfeld_electron_heat.py | scripts/physics_exp/sommerfeld_electron_heat.py |
| src/kernel_engine/thermo/soret_thermodiffusion.py | scripts/physics_exp/soret_thermodiffusion.py |
| src/kernel_engine/thermo/thermocouple_seebeck.py | scripts/physics_exp/thermocouple_seebeck.py |
| src/kernel_engine/thermo/thermoelectric_seebeck.py | scripts/physics_exp/thermoelectric_seebeck.py |
| src/kernel_engine/thermo/tidal_heating.py | scripts/physics_exp/tidal_heating.py |
| src/kernel_engine/thermo/wheatstone_bridge.py | scripts/physics_exp/wheatstone_bridge.py |
| src/kernel_engine/thermo/wkb_quantization.py | scripts/physics_exp/wkb_quantization.py |
| src/kernel_engine/tropical_sdf/2jet_sdf_curvature_decouples_v6.py | scripts/wave195_2jet_sdf_curvature_decouples_v6.py |
| src/kernel_engine/tropical_sdf/JxA_contact_param_over_determination_material_physics_leg_lifts_the_motion_regime_null.py | scripts/wave663_JxA_contact_param_over_determination_material_physics_leg_lifts_the_motion_regime_null.py |
| src/kernel_engine/tropical_sdf/certified_generative_support_placement_worstcase_sigmamin_robust_dfc.py | scripts/wave302_certified_generative_support_placement_worstcase_sigmamin_robust_dfc.py |
| src/kernel_engine/tropical_sdf/contact_cert_chi_from_sdf_error.py | scripts/wave199_contact_cert_chi_from_sdf_error.py |
| src/kernel_engine/tropical_sdf/contact_dof_cert_identify_or_abstain.py | scripts/wave198_contact_dof_cert_identify_or_abstain.py |
| src/kernel_engine/tropical_sdf/contact_orientation_gauge_emergence.py | scripts/wave197_contact_orientation_gauge_emergence.py |
| src/kernel_engine/tropical_sdf/contact_penalty_conditioning_omega_max_side_accuracy_cost_tradeoff_completes_sim_conditioning.py | scripts/wave488_contact_penalty_conditioning_omega_max_side_accuracy_cost_tradeoff_completes_sim_conditioning.py |
| src/kernel_engine/tropical_sdf/engine_contact_manifold_rank_twist_selfstress_2point_reduction.py | scripts/wave288_engine_contact_manifold_rank_twist_selfstress_2point_reduction.py |
| src/kernel_engine/tropical_sdf/engine_manifold_reduction_needs_3points_not_2_wrench_span.py | scripts/wave289_engine_manifold_reduction_needs_3points_not_2_wrench_span.py |
| src/kernel_engine/tropical_sdf/engine_which_3_points_Eoptimal_sigmamin_manifold_not_max_area.py | scripts/wave290_engine_which_3_points_Eoptimal_sigmamin_manifold_not_max_area.py |
| src/kernel_engine/tropical_sdf/friction_mu_is_a_sigma_min_null_under_stick_lifted_only_by_slip_events_contact_sim_ready.py | scripts/wave661_friction_mu_is_a_sigma_min_null_under_stick_lifted_only_by_slip_events_contact_sim_ready.py |
| src/kernel_engine/tropical_sdf/mma_capstone_validity_band_sweep.py | scripts/wave246_mma_capstone_validity_band_sweep.py |
| src/kernel_engine/tropical_sdf/mma_topopt_capstone_stress_constrained_dfc_positive.py | scripts/wave245_mma_topopt_capstone_stress_constrained_dfc_positive.py |
| src/kernel_engine/tropical_sdf/restitution_null_lifted_by_impact_friction_and_restitution_are_decorrelated_contact_nulls.py | scripts/wave662_restitution_null_lifted_by_impact_friction_and_restitution_are_decorrelated_contact_nulls.py |
| src/kernel_engine/tropical_sdf/so_arm100_contact_sim_cost_from_tropical_sdf_backend_hertz_stiffness_sets_stable_dt_stability_verified.py | scripts/wave530_so_arm100_contact_sim_cost_from_tropical_sdf_backend_hertz_stiffness_sets_stable_dt_stability_verified.py |
| src/kernel_engine/tropical_sdf/stress_constrained_compliance_topopt_dfc_positive.py | scripts/wave237_stress_constrained_compliance_topopt_dfc_positive.py |
| src/kernel_engine/tropical_sdf/stress_objective_topopt_design_for_certifiability_positive.py | scripts/wave236_stress_objective_topopt_design_for_certifiability_positive.py |
| src/kernel_engine/tropical_sdf/topopt_stress_frontier_forced_number_not_honest_label.py | scripts/wave250_topopt_stress_frontier_forced_number_not_honest_label.py |
| src/kernel_engine/tropical_sdf/topopt_stress_vs_compliance_design_for_certifiability.py | scripts/wave235_topopt_stress_vs_compliance_design_for_certifiability.py |
| src/kernel_engine/tropical_sdf/tropical_sdf_SIMT_tax_cost_model_compute_bound_vs_voxel_bandwidth_bound_crossover_Kstar.py | scripts/wave505_tropical_sdf_SIMT_tax_cost_model_compute_bound_vs_voxel_bandwidth_bound_crossover_Kstar.py |
| src/kernel_engine/tropical_sdf/tropical_sdf_backend_computes_full_v6_contact_geometry_ladder_2jet_curvature_anisotropy_on_so_arm100.py | scripts/wave528_tropical_sdf_backend_computes_full_v6_contact_geometry_ladder_2jet_curvature_anisotropy_on_so_arm100.py |
| src/kernel_engine/tropical_sdf/tropical_sdf_backend_min_of_capsules_beats_voxel_3x_bytes_matched_penetration_so_arm100.py | scripts/wave496_tropical_sdf_backend_min_of_capsules_beats_voxel_3x_bytes_matched_penetration_so_arm100.py |
| src/kernel_engine/tropical_sdf/tropical_sdf_so_arm100_geometry_backend_v2_config_driven_posing_plus_contact_normal_gradient.py | scripts/wave527_tropical_sdf_so_arm100_geometry_backend_v2_config_driven_posing_plus_contact_normal_gradient.py |
| src/kernel_engine/tropical_sdf/tropical_sdf_swept_volume_continuous_collision_detection_min_over_trajectory.py | scripts/wave540_tropical_sdf_swept_volume_continuous_collision_detection_min_over_trajectory.py |
| src/kernel_engine/warp_gpu/gpu_fracture_determinism_sigma.py | scripts/physics_exp/gpu_fracture_determinism_sigma.py |
| src/kernel_engine/warp_gpu/mujoco_warp_bench.py | scripts/physics_exp/mujoco_warp_bench.py |
| src/kernel_engine/warp_gpu/rigid2d_primal_vbd_gpu.py | scripts/physics_exp/rigid2d_primal_vbd_gpu.py |
| src/kernel_engine/warp_gpu/warp_bodybody_colored_gpu.py | scripts/physics_exp/warp_bodybody_colored_gpu.py |
| src/kernel_engine/warp_gpu/warp_bodybody_jacobi_gpu.py | scripts/physics_exp/warp_bodybody_jacobi_gpu.py |
| src/kernel_engine/warp_gpu/warp_bodybody_scale_gpu.py | scripts/physics_exp/warp_bodybody_scale_gpu.py |
| src/kernel_engine/warp_gpu/warp_cylinder_roll_gpu.py | scripts/physics_exp/warp_cylinder_roll_gpu.py |
| src/kernel_engine/warp_gpu/warp_granular_friction.py | scripts/physics_exp/warp_granular_friction.py |
| src/kernel_engine/warp_gpu/warp_granular_gpu.py | scripts/physics_exp/warp_granular_gpu.py |
| src/kernel_engine/warp_gpu/warp_mesh_roll_gpu.py | scripts/physics_exp/warp_mesh_roll_gpu.py |
| src/kernel_engine/warp_gpu/warp_rigid_ramp_gpu.py | scripts/physics_exp/warp_rigid_ramp_gpu.py |
| src/kernel_engine/warp_gpu/warp_rigid_ramp_gpu_v2.py | scripts/physics_exp/warp_rigid_ramp_gpu_v2.py |
| src/kernel_engine/warp_gpu/warp_rl_env_gpu.py | scripts/physics_exp/warp_rl_env_gpu.py |
| src/kernel_engine/wave_fdtd/acoustic_emission.py | scripts/acoustic_emission.py |
| src/kernel_engine/wave_fdtd/acoustic_fdtd.py | scripts/physics_exp/acoustic_fdtd.py |
| src/kernel_engine/wave_fdtd/diag_acoustic_mode.py | scripts/physics_exp/diag_acoustic_mode.py |
| src/kernel_engine/wave_fdtd/diag_acoustic_seed.py | scripts/physics_exp/diag_acoustic_seed.py |
| src/kernel_engine/wave_fdtd/diag_acoustic_spectrum.py | scripts/physics_exp/diag_acoustic_spectrum.py |
| src/kernel_engine/wave_fdtd/diff_wave_3d.py | scripts/physics_exp/diff_wave_3d.py |
| src/kernel_engine/wave_fdtd/diff_wave_substrate.py | scripts/physics_exp/diff_wave_substrate.py |
| src/kernel_engine/wave_fdtd/elastodynamics_kache.py | scripts/physics_exp/elastodynamics_kache.py |
| src/kernel_engine/wave_fdtd/goc_wave_transient.py | scripts/physics_exp/goc_wave_transient.py |
| src/kernel_engine/wave_fdtd/optics_coating.py | scripts/physics_exp/optics_coating.py |
| src/kernel_engine/wave_fdtd/optics_fdtd.py | scripts/physics_exp/optics_fdtd.py |
| src/kernel_engine/wave_fdtd/persona_design_acoustic.py | scripts/physics_exp/persona_design_acoustic.py |
| src/kernel_engine/wave_fdtd/persona_design_acoustic_cavity_shape.py | scripts/physics_exp/persona_design_acoustic_cavity_shape.py |
| src/kernel_engine/wave_fdtd/persona_design_acoustic_inverse.py | scripts/physics_exp/persona_design_acoustic_inverse.py |
| src/kernel_engine/wave_fdtd/persona_design_acoustic_shape_adjoint.py | scripts/physics_exp/persona_design_acoustic_shape_adjoint.py |
| src/kernel_engine/wave_fdtd/persona_design_lbm_acoustic.py | scripts/physics_exp/persona_design_lbm_acoustic.py |
| src/kernel_engine/wave_fdtd/s1_acoustic_metamaterial_bandgap.py | scripts/physics_exp/s1_acoustic_metamaterial_bandgap.py |
| src/kernel_engine/wave_fdtd/thermoacoustic_rijke_dde.py | scripts/physics_exp/thermoacoustic_rijke_dde.py |
| src/kernel_engine/wave_fdtd/vibroacoustic.py | scripts/physics_exp/vibroacoustic.py |
| src/kernel_engine/wave_fdtd/wave_fdtd_3d.py | scripts/physics_exp/wave_fdtd_3d.py |
| src/kernel_engine/wave_fdtd/wave_fdtd_kache.py | scripts/physics_exp/wave_fdtd_kache.py |
| src/kernel_engine/wave_fdtd/wave_fdtd_verify.py | scripts/physics_exp/wave_fdtd_verify.py |

212 files.

The `scripts/wave*` cells are copied into `src/kernel_engine/tropical_sdf/` with the `wave<NNN>_` filename
prefix removed; the source column above carries the original file name, so the mapping stays exact.

## Deliberate code changes in the second pass

4. Sibling imports: the platform keeps these modules in one flat directory, this repository splits them into
   packages by solver class. Files that import a sibling module by bare name gained one block at the top of
   the import section that puts the relevant package directories on `sys.path` when the file is run as a
   script. No import name changed.
5. `warpfem_compute_router.py`: the conformal-UQ helper it loads by path now resolves to
   `src/kernel_engine/_vendor/uq.py` instead of the platform's `src/cad2simready/uq.py`.
6. `l_waterfill_deploygap.py` and `l_waterfill_sufficiency.py` wrote their evidence JSON to a platform
   directory; they now write to `artifacts/` next to themselves (created on demand).
7. Absolute paths (`/home/...`, worktree roots, scratch directories) were rewritten to repository-relative
   paths or to `python3`; dates, worktree/session identifiers and personal names were removed from prose.
8. Five tropical-SDF cells load a sibling cell by file name; since the `wave<NNN>_` prefix was dropped, those
   loads now resolve the renamed file next to them.
9. Seventeen modules wrote evidence JSON to a platform `reports/` path; they now write to `artifacts/` next
   to themselves. `artifacts/`, `data/` and `reports/` are gitignored.
10. `reductions/probe_kernel_determinism_cert.py` and `lbm/d_static_deployment_gate_real_lbm.py` reach a
    module that now lives in a different package; both gained the directory of that package on `sys.path`.
11. `fem/warpfem_field_surrogate_fallbevis_v1.py`: the source script it checks for and the side-file
    directory it writes are now resolved next to the file.
12. `allocation/d_goal_derived_representation.py` (from the first pass): one string field held a platform
    path; it now holds the bare file name.

## Swedish-prose sweep 2026-09-12
| File | Text changed |
| --- | --- |
| `src/kernel_engine/_vendor/uq.py` | one docstring line |
| `src/kernel_engine/fem/warpfem_transient_structural.py` | one progress print string |
| `src/kernel_engine/fem/spectral_fatigue.py` | one inline comment |
| `src/kernel_engine/fem/warpfem_design_objective.py` | two-line comment on the projected-gradient step |
| `src/kernel_engine/fem/warpfem_stokes_poiseuille.py` | module docstring (analytic gate line) |
| `src/kernel_engine/fem/warpfem_modal.py` | module docstring (analytic gate line) |
| `src/kernel_engine/fem/warpfem_thermoelastic.py` | one tolerance comment, one solver print string |
| `src/kernel_engine/kernel_variants/kernelvarv_v1_f4_csg.py` | three-line correctness-gate comment |
| `src/kernel_engine/lbm/lbm_voxel_aero.py` | two result/progress print strings |
| `src/kernel_engine/lbm/lbm_voxel_aero_gpu.py` | four print strings, two inline comments |
| `src/kernel_engine/lbm/lbm_gpu_fp16.py` | module docstring title, two print strings |
| `src/kernel_engine/lbm/lbm_gpu_fp16_half2.py` | module docstring title, one print string |
| `src/kernel_engine/lbm/gpu_lbm_luftflode_v1.py` | four boundary-condition/forcing comment blocks, one env-var comment, one ValueError message, one progress print string (identifiers `steg`/`andel`/`namn` and the history keys left unchanged) |
| `src/kernel_engine/warp_gpu/warp_rigid_ramp_gpu_v2.py` | one inline comment |
| `examples/kernelvarv_v2.json` | the prose `winner` value (variant names unchanged) |

## Files copied in the dataset follow-up (destination -> source path relative to the platform repository root)

Twelve modules that the second pass had deferred for a missing measured dataset. The datasets are now in
`data/` (git-ignored, see docs/RUNNING.md "Real datasets").

| in this repository | source |
| --- | --- |
| src/kernel_engine/certified_kernels/apriori_requirement_cert_on_real_kernelbench.py | scripts/physics_exp/apriori_requirement_cert_on_real_kernelbench.py |
| src/kernel_engine/certified_kernels/kernelbench_addressing_census_provenance_gate_absent.py | scripts/physics_exp/kernelbench_addressing_census_provenance_gate_absent.py |
| src/kernel_engine/thermo/p8_combustion_cert_pod_a90.py | scripts/physics_exp/p8_combustion_cert_pod_a90.py |
| src/kernel_engine/thermo/p8_combustion_cert_watertight.py | scripts/physics_exp/p8_combustion_cert_watertight.py |
| src/kernel_engine/thermo/p8_combustion_state_certification.py | scripts/physics_exp/p8_combustion_state_certification.py |
| src/kernel_engine/thermo/p8_delft_conditional_variance.py | scripts/physics_exp/p8_delft_conditional_variance.py (dependency: `cond_stats`) |
| src/kernel_engine/thermo/p8_delft_flamelet_render_match.py | scripts/physics_exp/p8_delft_flamelet_render_match.py (dependency: `dng_streams`) |
| src/kernel_engine/thermo/p8_delft_species_flamelet.py | scripts/physics_exp/p8_delft_species_flamelet.py (dependency: `load_scatter`) |
| src/kernel_engine/thermo/p8_ecn_combustion_energy.py | scripts/physics_exp/p8_ecn_combustion_energy.py |
| src/kernel_engine/thermo/p8_flame_mixture_fraction_render_match.py | scripts/physics_exp/p8_flame_mixture_fraction_render_match.py (dependency: `burke_schumann`) |
| src/kernel_engine/thermo/p8_sandia_extinction_flamelet_breakdown.py | scripts/physics_exp/p8_sandia_extinction_flamelet_breakdown.py (dependency: `load_yall`) |
| src/kernel_engine/wave_fdtd/acoustic_sigma_renderer.py | scripts/acoustic_sigma_renderer.py |

## Deliberate code changes in the dataset follow-up

| file | change |
| --- | --- |
| all twelve | absolute platform dataset paths replaced by `data/<slug>/...` resolved from the repository root via `__file__` |
| all twelve | a stand-in input path: when the slug directory is absent the module prints `SYNTHETIC INPUT` and runs the same pipeline on generated input (flamelet manifold with turbulent noise and an extinguished branch; a constant-volume pressure trace; KernelBench-form single-op modules; bearing vibration as broadband noise plus a defect-frequency impulse train) |
| `certified_kernels/apriori_requirement_cert_on_real_kernelbench.py` | module docstring rewritten to behaviour + I/O + gates; session/adjudication prose removed from the prints and from the `claim`/`honest_scope`/`provenance` JSON fields; evidence written to `artifacts/` instead of `reports/`; corpus iteration moved into `iter_level1()` |
| `certified_kernels/kernelbench_addressing_census_provenance_gate_absent.py` | same three edits; `iter_corpus()` falls back to the stand-in corpus; JSON key `cell` -> `module`, gate key `G3_grounds_D_residual` -> `G3_input_keyed_not_testable_on_this_corpus` |
| `wave_fdtd/acoustic_sigma_renderer.py` | Swedish docstring, comments, printed strings and verdict text translated; the hard-coded per-bearing counts in the closing verdict replaced by the computed values; the fail-loud data-availability check kept for the real-data branch |
| the nine `thermo/p8_*.py` | run-command line in the docstring reduced to `python3 <file>`; cross-cell references and bracketed doc links removed |


## Files copied in the GPU-index sweep (destination -> source path relative to the platform repository root)

Thirty-seven modules selected from the index of GPU-importing platform scripts that were not yet in any
staging repository: fluid/CFD, FEM, thermal, wave, allocation and kernel-certification rows only.

| in this repository | source |
| --- | --- |
| src/kernel_engine/certified_kernels/aa_micro_bench.py | scripts/physics_exp/aa_micro_bench.py |
| src/kernel_engine/certified_kernels/d_energy_exponent_substrate_invariant.py | scripts/physics_exp/d_energy_exponent_substrate_invariant.py |
| src/kernel_engine/certified_kernels/gpu_duty_torch_v1.py | scripts/phys/gpu_duty_torch_v1.py |
| src/kernel_engine/certified_kernels/module_const_launch_tune.py | scripts/physics_exp/module_const_launch_tune.py |
| src/kernel_engine/certified_kernels/morton_3d_stencil.py | scripts/physics_exp/morton_3d_stencil.py |
| src/kernel_engine/certified_kernels/morton_sparse_gather.py | scripts/physics_exp/morton_sparse_gather.py |
| src/kernel_engine/certified_kernels/probe_compute_ladder_descent.py | scripts/physics_exp/probes/probe_compute_ladder_descent.py |
| src/kernel_engine/certified_kernels/probe_exclusive_sweep_block.py | scripts/physics_exp/probes/probe_exclusive_sweep_block.py |
| src/kernel_engine/certified_kernels/probe_l2_inband_endgame.py | scripts/physics_exp/probes/probe_l2_inband_endgame.py |
| src/kernel_engine/certified_kernels/probe_sync_density_victim_model.py | scripts/physics_exp/probes/probe_sync_density_victim_model.py |
| src/kernel_engine/certified_kernels/probe_sync_penalty_vs_priority.py | scripts/physics_exp/probes/probe_sync_penalty_vs_priority.py |
| src/kernel_engine/certified_kernels/probe_timeslice_dma_fartail.py | scripts/physics_exp/probes/probe_timeslice_dma_fartail.py |
| src/kernel_engine/certified_kernels/ser_fracture_compaction.py | scripts/physics_exp/ser_fracture_compaction.py |
| src/kernel_engine/euler_hllc/lubrication_reynolds_bearing.py | scripts/lubrication_reynolds_bearing.py |
| src/kernel_engine/fem/fsi_added_mass.py | scripts/fsi_added_mass.py |
| src/kernel_engine/fem/fsi_pipe_flutter.py | scripts/fsi_pipe_flutter.py |
| src/kernel_engine/fem/neuber_notch_plasticity.py | scripts/neuber_notch_plasticity.py |
| src/kernel_engine/fem/plasticity_3d_j2.py | scripts/plasticity_3d_j2.py |
| src/kernel_engine/fem/plasticity_return_mapping.py | scripts/plasticity_return_mapping.py |
| src/kernel_engine/fem/thermal_buckling.py | scripts/thermal_buckling.py |
| src/kernel_engine/fem/viscoelastic_preload_relaxation.py | scripts/viscoelastic_preload_relaxation.py |
| src/kernel_engine/lbm/coupled_design_aero_struct.py | scripts/physics_exp/coupled_design_aero_struct.py |
| src/kernel_engine/lbm/differentiable_flow_control.py | scripts/physics_exp/differentiable_flow_control.py |
| src/kernel_engine/lbm/differentiable_fsi_chain.py | scripts/physics_exp/differentiable_fsi_chain.py |
| src/kernel_engine/lbm/g18_cfd_nilss_prereq_wake_chaos.py | scripts/physics_exp/g18_cfd_nilss_prereq_wake_chaos.py |
| src/kernel_engine/lbm/g19_forced_2d_wake_nilss_prereq.py | scripts/physics_exp/g19_forced_2d_wake_nilss_prereq.py |
| src/kernel_engine/lbm/g20_3d_wake_chaos_nilss_prereq.py | scripts/physics_exp/g20_3d_wake_chaos_nilss_prereq.py |
| src/kernel_engine/lbm/g21_3d_wake_chaos_highRe.py | scripts/physics_exp/g21_3d_wake_chaos_highRe.py |
| src/kernel_engine/lbm/probe_kam_resonance_dither_strides.py | scripts/physics_exp/probes/probe_kam_resonance_dither_strides.py |
| src/kernel_engine/reductions/d_1c_iv_best_in_class_float4.py | scripts/physics_exp/d_1c_iv_best_in_class_float4.py |
| src/kernel_engine/reductions/det_accumulation_probe.py | scripts/physics_exp/det_accumulation_probe.py |
| src/kernel_engine/wave_fdtd/coupled_multiphysics_calibration.py | scripts/physics_exp/coupled_multiphysics_calibration.py |
| src/kernel_engine/wave_fdtd/goc_wave_verify.py | scripts/physics_exp/goc_wave_verify.py |
| src/kernel_engine/wave_fdtd/sigma_guided_fwi.py | scripts/physics_exp/sigma_guided_fwi.py |
| src/kernel_engine/wave_fdtd/twin_calibration_multisource.py | scripts/physics_exp/twin_calibration_multisource.py (dependency of coupled_multiphysics_calibration) |

## Deliberate code changes in the GPU-index sweep

| file | change |
| --- | --- |
| all thirty-seven | Swedish docstrings, comments, printed strings and verdict text translated to English; module docstrings trimmed to behaviour, I/O and gates; session, worktree and attribution references removed from prose (identifiers, JSON keys and CLI flags unchanged) |
| all run-command lines | `.venv-newton/bin/python <platform path>` reduced to `python3 <file name>` |
| `certified_kernels/probe_timeslice_dma_fartail.py` | evidence path `reports/probes/...` -> `artifacts/` next to the file; the scratch `RATE_DIR` for the disturber rate files -> `artifacts/idma/`, created on demand |
| `certified_kernels/probe_exclusive_sweep_block.py` | `REP` -> `artifacts/` next to the file; `RATE_DIR` -> `artifacts/isweep/` |
| `certified_kernels/probe_sync_density_victim_model.py` | `RATE_DIR` -> `artifacts/isync/` |
| `certified_kernels/probe_l2_inband_endgame.py` | `RATE_DIR` -> `artifacts/` |
| `certified_kernels/aa_micro_bench.py` | sibling-package bootstrap added for `lbm_gpu_fp16_half2` |
| `certified_kernels/probe_thermal_rung.py` | copied, then removed: it loads `probe_compute_twin_v1.py` (796 lines) by path, which is not in this repository |
| `certified_kernels/gpu_duty_torch_v1.py` | the module is a helper class with no gate of its own, so it is covered by a dedicated pytest case instead of being run as a script |
- 2026-09-12: `inverse_design_wave.py` and `twin_calibration_wave.py` were staged here by the GPU-index sweep and removed again: both already ship in `3fold-physics/src/physics_engine/wave_optics/`, and a module lives in one repository only.

## CUDA verification run 2026-09-12 (code edits)

| file | change |
| --- | --- |
| `kernel_variants/kernelvarv_v1_f2_matvec.py`, `kernel_variants/kernelvarv_v1_f4_csg.py`, `kernel_variants/kernelvarv_v2_f4_closing.py` | `OUT_DIR` retargeted from `<src>/reports/probes/kernelvarv_v1_sidofiler/` to `artifacts/` next to the file; the now-unused `ROOT` constant dropped; the stray `src/reports/` directory deleted |
| `certified_kernels/u_v874_roofline_soft_knee_sweep_tunable_arithmetic_intensity.py` | evidence path `evidence/u_v874_roofline_soft_knee_results.json` (relative to the caller's working directory) -> `artifacts/` next to the file, created on demand; `import os` added |
| `certified_kernels/u_v870_roofline_ceiling_long_window_sustained_throttle_recheck.py` | same change for `evidence/u_v870_roofline_long_window_results.json` |
| `certified_loop/d_lbm_soa_certified_roofline_close.py` | `os.makedirs` added for the `artifacts/` directory it writes its evidence JSON into |
| `lbm/gpu_lbm_utilization_cell.py` | `REPO` pointed at the repository root instead of `src/`, so `reports/probes/` resolves to the shipped directory |
| `tests/test_pass3_selftests.py` | `run()` skips a CUDA case when the module's own GPU-idle guard aborted it (markers only, no guard weakened); `lbm/gpu_lbm_utilization_cell.py` moved to a dedicated case that skips while its predecessor artefact is absent; `certified_kernels/d_coupled_knob_ordering_advantage_real_cuda.py`, `lbm/lbm_fsi_viv.py` and `lbm/lbm_gpu_fp16.py` moved to dedicated cases that accept their designed non-zero exit and assert the verdict line, as `amr_octree_fv` already did |
| `tests/test_selftests.py` | `kernelvarv_v1_f2_matvec` and `kernelvarv_v1_f4_csg` likewise accept their designed exit 1 and assert the JSON verdict |
| `certified_kernels/l_phase0_c2_roofline.py` (prose) | the `RUNTIME["requested"]` string lost its worktree and author references: now "Hong-Kung roofline cell, run under the GPU lock" |
| `certified_kernels/u_h5_thermal_roofline_twin_state.py` | evidence JSON `src/evidence/` -> `artifacts/` next to the file |
| `certified_kernels/l_phase0_c2_roofline.py`, `certified_kernels/probe_sync_density_victim_model.py` | evidence JSON written next to the module -> `artifacts/`, created on demand, so a run leaves nothing outside the ignored output directories |
| `tests/test_pass3_selftests.py` (run budgets) | `lbm/g21_3d_wake_chaos_highRe.py` is run with its own `--validate` short self-test (the full 288x176x224 sweep exceeds 1500 s); `lbm/g20_3d_wake_chaos_nilss_prereq.py` gets a 1200 s subprocess budget (measured 678 s) |
| `requirements.txt` | `mujoco` and `mujoco-warp` added as optional extras (for `warp_gpu/mujoco_warp_bench.py`); the optional `wgpu` comment now names both modules that need it |
- 2026-09-12: `wave_fdtd/sigma_guided_fwi.py`, `wave_fdtd/twin_calibration_multisource.py` and
  `wave_fdtd/coupled_multiphysics_calibration.py` moved to `3fold-physics/src/physics_engine/wave_optics/`:
  they import `twin_calibration_wave`, which ships there, and the chain
  `coupled_multiphysics_calibration -> twin_calibration_multisource -> sigma_guided_fwi` is wave calibration,
  not a kernel cell. Their rows moved to that repository's RUNNING.md, RENAMES.md and tests.

## Prose sweep 2026-09-12 (placeholders, platform wording)

Comments, docstrings and printed narrative only. No identifier, filename, JSON key or CLI flag changed.
Prose uses of "the platform" replaced with plain wording; internal milestone tags and ledger references dropped.

Files:

- `src/kernel_engine/amr_poisson/poisson_dispatch.py`
- `src/kernel_engine/amr_poisson/amr_sigma_scaling.py`
- `src/kernel_engine/certified_kernels/d_1c_iv_end_to_end_real_cuda_cert.py`
- `src/kernel_engine/certified_kernels/d_avbd_gpu_certified_roofline.py`
- `src/kernel_engine/certified_kernels/d_ensemble_coexecution_cert_contended_roofline.py`
- `src/kernel_engine/certified_kernels/u_v870_roofline_ceiling_long_window_sustained_throttle_recheck.py`
- `src/kernel_engine/certified_kernels/u_v874_roofline_soft_knee_sweep_tunable_arithmetic_intensity.py`
- `src/kernel_engine/certified_loop/d_lbm_soa_certified_roofline_close.py`
- `src/kernel_engine/euler_hllc/substrate_compose.py`
- `src/kernel_engine/kernel_gen/d_l1_kernel_certvec_compose.py`
- `src/kernel_engine/lbm/d_certvector_on_real_lbm.py`
- `src/kernel_engine/lbm/d_integration_stitch_lbm_contact_scenario_cert.py`
- `src/kernel_engine/lbm/d_static_deployment_gate_real_lbm.py`
- `src/kernel_engine/lbm/g18_cfd_nilss_prereq_wake_chaos.py`
- `src/kernel_engine/lbm/g19_forced_2d_wake_nilss_prereq.py`
- `src/kernel_engine/lbm/g20_3d_wake_chaos_nilss_prereq.py`
- `src/kernel_engine/lbm/g21_3d_wake_chaos_highRe.py`
- `src/kernel_engine/lbm/lbm_aero_v0.py`
- `src/kernel_engine/lbm/lbm_mach_ceiling.py`
- `src/kernel_engine/lbm/thermofluid_lbm_rayleigh.py`
- `src/kernel_engine/reductions/d_1c_iv_best_in_class_float4.py`
- `src/kernel_engine/thermo/d_thermo_computing_equals_fusion.py`
- `src/kernel_engine/thermo/thermoelectric_seebeck.py`
- `src/kernel_engine/tropical_sdf/JxA_contact_param_over_determination_material_physics_leg_lifts_the_motion_regime_null.py`
- `src/kernel_engine/warp_gpu/gpu_fracture_determinism_sigma.py`
- `src/kernel_engine/wave_fdtd/persona_design_acoustic_cavity_shape.py`
- `src/kernel_engine/wave_fdtd/s1_acoustic_metamaterial_bandgap.py`
- `docs/RUNNING.md`

## Chunkable-recurrence rule (new code, no platform source)

`src/kernel_engine/certified_kernels/chunkable_recurrence_rule.py` and
`tests/test_chunkable_recurrence_rule.py` were written for this repository; they are not copies of any
platform file. The module adds a fifth a-priori requirement to the requirement cert: the structure of a
task's state transition (diagonal, diagonal-plus-low-rank / delta rule with or without a diagonal gate, or an
explicitly associative scan -> `chunkable`; nonlinear state map -> `sequential`), the suggested chunk size and
the sequential and chunked byte floors. The chunkwise (WY) form of the rank-1 delta-rule update is cited in
the module docstring (arXiv 2510.26692). Synthetic inputs only.

| file | change |
| --- | --- |
| `certified_kernels/chunkable_recurrence_rule.py` | new module: `detect_recurrence`, `recurrence_requirement`, `suggest_chunk_size`, `byte_floors`, the step-by-step and chunkwise gated delta-rule reference implementations, and a selftest with three gates |
| `certified_kernels/apriori_requirement_cert_on_real_kernelbench.py` | imports `recurrence_requirement` (not re-implemented) and reports it as `[R5]` per file and as JSON key `R5_chunkable_recurrence`; new CLI flag `--no-chunkable-rule` reproduces the four-requirement cert exactly; `main()` takes an optional `argv`; gates G1-G3 and their numbers unchanged |
| `certified_kernels/kernelbench_addressing_census_provenance_gate_absent.py` | same import; the census reports per level how many kernels change class from `sequential` to `chunkable` (printed `[R5]` block, JSON key `R5_chunkable_recurrence`); gates G1-G3 and their numbers unchanged |
| `tests/test_chunkable_recurrence_rule.py` | new pytest wrapper: chunked-vs-sequential equivalence, the transition classes, the byte floors, and cert-gate parity with and without `--no-chunkable-rule` |
| `tests/test_pass3_selftests.py` | `certified_kernels/chunkable_recurrence_rule.py` added to `CPU_MODULES` |

## Warp -> CUDA C++ export prototype

The Warp code generator's output for `reduce_int64_atomic` (from
`reductions/u_v54_gather_law_deterministic_reduction_vs_atomic_int64.py`) compiled offline with `nvcc`
into a host program that links no Warp runtime, plus the verification that it stays bit-identical and
keeps its bandwidth fraction. The generated `.cu` and the header closure are NVIDIA Warp output /
Warp device headers, Apache-2.0; see `kernel_gen/export/THIRD_PARTY`.

| file | change |
| --- | --- |
| `kernel_gen/export/reduce_int64_atomic_generated.cu` | Warp-generated CUDA C++ for the module, taken from the kernel cache unmodified except that `#define WP_NO_CRT` is commented out (offline `nvcc` has the real CRT headers, NVRTC does not); comment added at that line |
| `kernel_gen/export/warp_native/` | the 35 Warp device headers the generated file includes (transitive closure from `nvcc -M`), copied byte-for-byte, Apache-2.0 |
| `kernel_gen/export/export_shim.cu` | new: `extern "C" void wp_export_launch_reduce_int64(const int32_t*, const int64_t*, int64_t*, int n, int nb, int block_dim, cudaStream_t)` - builds `wp::launch_bounds_t<1>` and the three `wp::array_t<>` arguments and launches the generated kernel with Warp's own configuration (block 256, grid-stride grid) |
| `kernel_gen/export/host_int64_reduce.cu` | new: CUDA-runtime-only host program; reads the two `.npy` inputs, launches through the shim, prints the slot sum, an FNV-1a hash of the output bytes and CUDA-event timing over 20 repetitions, writes `out_nvcc.bin` |
| `kernel_gen/export/warp_reference_run.py` | new: Warp-side driver; writes `in_tgt.npy`, `in_qval.npy`, `out_warp.npy` and prints the same sum/hash/timing line for the Warp-launched original |
| `kernel_gen/export/build.sh` | new: the exact `nvcc` command (`-O3 -std=c++17 -arch=sm_120a -DNDEBUG -DWP_ENABLE_CUDA=1 -diag-suppress 177,550 -Iwarp_native`) |
| `kernel_gen/export/EXPORT.md` | new: export steps, the shim and Warp's launch ABI, the measured table, and what a foreign C engine needs to call the kernel |
| `kernel_gen/export/THIRD_PARTY` | new: licence note for the copied Warp headers and generated source |
| `tests/test_kernel_gen_export.py` | new pytest wrapper: builds with `nvcc`, runs both paths on the same input, asserts element-wise equality; skips without `nvcc` or a CUDA device |

## Next batch of the GPU-index sweep (2026-09-12)

Source worktrees: `CADtoSIMReady-G` and `bodytwin` (a platform worktree checkout, not body-twin subject
matter). 15 modules plus one vendored dependency; `wave###_` prefixes were absent in this batch.

| file | platform source | edits |
| --- | --- | --- |
| `_vendor/_emit.py` | `evidence/_emit.py` | vendored dependency of the three wake cells; docstring trimmed; `_DIR` retargeted from the directory of the file to `src/kernel_engine/artifacts/` and the directory is created on write |
| `certified_kernels/probe_dma_single_stream_gap.py` | `bodytwin/scripts/physics_exp/probes/probe_dma_single_stream_gap.py` | docstring header line rewritten (session/agent reference removed, tenancy line reworded); the report JSON now goes through a new `_out()` helper into `artifacts/` instead of `reports/probes/` |
| `certified_kernels/probe_dma_split_shape_decisive.py` | `bodytwin/scripts/physics_exp/probes/probe_dma_split_shape_decisive.py` | same: header line rewritten, the two instruments named by what they do; `_out()` helper, `artifacts/` |
| `certified_kernels/probe_l2_arbitration_endgame.py` | `bodytwin/scripts/physics_exp/probes/probe_l2_arbitration_endgame.py` | header caveat kept verbatim in substance, its date, round numbers and agent reference removed; the reference to a scratchpad calibration script dropped; `REP` retargeted to `artifacts/`; no gate or tolerance changed |
| `fem/fno_3d.py` | `bodytwin/scripts/fno_3d.py` | Swedish docstring, comments and print/verdict strings translated; the roadmap and cloud-training paragraphs dropped; gates, weights and tolerances unchanged |
| `fem/fno_tpu_compatible.py` | `bodytwin/scripts/fno_tpu_compatible.py` | Swedish docstring, comments and print/verdict strings translated; the accelerator-mapping narrative reduced to "matmul-only accelerators"; numbers and gates unchanged |
| `lbm/g39_cfd_force_nilss_bed.py` | `CADtoSIMReady-G/scripts/physics_exp/g39_cfd_force_nilss_bed.py` | `G39_CACHE` default and the evidence path retargeted from a scratchpad path and `evidence/` to `artifacts/`; invocation line and the venv name in the provenance string replaced |
| `lbm/g42_wake_benettin_dim.py` | `CADtoSIMReady-G/scripts/physics_exp/g42_wake_benettin_dim.py` | `SCRATCH` retargeted to `artifacts/`; `_emit` import path changed from `evidence/` to `_vendor/`; invocation line replaced |
| `lbm/g61_modeA_Rec_cylinder_vs_ellipse.py` | `CADtoSIMReady-G/scripts/physics_exp/g61_modeA_Rec_cylinder_vs_ellipse.py` | `RESDIR` and the `_emit` import path retargeted to `artifacts/g61_runs` and `_vendor/`; invocation line replaced |
| `reductions/d_1c_v_atomic_saturation_check.py` | `bodytwin/scripts/physics_exp/d_1c_v_atomic_saturation_check.py` | copied unchanged (already English, no platform paths) |
| `warp_gpu/substep_value_probe.py` | `bodytwin/scripts/physics_exp/substep_value_probe.py` | Swedish docstring, comments and all print strings translated; the reference to an external agent report reworded as "the reported XPBD substepping result"; every number, gate and threshold unchanged |
| `warp_gpu/warmstart_massratio.py` | `bodytwin/scripts/physics_exp/warmstart_massratio.py` | Swedish docstring, comments and print strings translated; the `sys.path` insert made relative to the file; the unrelated `/tmp/rnea_parity_franka.npz` precondition guard removed (a stale robot-spec file gate, not used by the physics) |
| `warp_gpu/warmstart_value_probe.py` | `bodytwin/scripts/physics_exp/warmstart_value_probe.py` | Swedish docstring, comments and print strings translated; the instrument guard and its thresholds unchanged |
| `wave_fdtd/xray_3d_dda.py` | `bodytwin/scripts/physics_exp/xray_3d_dda.py` | first docstring line rewritten (attribution removed); invocation line replaced |
| `wave_fdtd/xray_tomography_sigma.py` | `bodytwin/scripts/physics_exp/xray_tomography_sigma.py` | first two docstring lines rewritten (attribution removed); invocation line replaced |
| `docs/RUNNING.md` | - | 16 Status rows added; the count line updated to 292 module rows |
| `tests/test_pass3_selftests.py` | - | 2 modules added to `CPU_MODULES`, 12 to `CUDA_MODULES`; `ARGS` entries for the three wake cells (`--validate`, `--validate`, `--smoke`) and 1800 s `TIMEOUTS` for two of them |


## Deterministic accumulation sweep (2026-09-12)

No new platform source: these are edits to already-shipped files plus one new cell and its test. Every edit
adds an int64 fixed-point accumulation path behind a module-level `DETERMINISTIC_ACCUMULATION = True` switch
(float path selectable by setting it to `False`); no identifier was renamed, no gate or tolerance changed.

| file | platform source | edits |
| --- | --- | --- |
| `certified_kernels/determinism_sweep_int64.py` | - | new cell: launches every touched accumulation kernel twice (float twin and int64 twin) and asserts the int64 result array is bit-identical, runs each touched selftest twice as a subprocess and compares normalised stdout hashes, writes `reports/determinism_sweep_int64_result.json`, exits non-zero on any failure |
| `lbm/differentiable_flow_control.py` | unchanged | added `DETERMINISTIC_ACCUMULATION`, `ACC_SCALE = 2**48` with an overflow assertion, kernels `track_loss_i64` and `probe_ux_i64`, helpers `probe_ux_value` and `track_loss_value`; `rollout` now also returns the final distribution field so the loss value can be recomputed order-invariantly; the float kernels stay for the tape |
| `lbm/differentiable_fsi_chain.py` | unchanged | added `DETERMINISTIC_ACCUMULATION`, `ACC_SCALE = 2**48` with an overflow assertion and kernel `drag_force_i64`; `forward` uses the int64 seam value whenever it is not recording a tape |
| `lbm/differentiable_lbm_probe.py` | unchanged | added `DETERMINISTIC_ACCUMULATION`, `ACC_SCALE = 2**49` with an overflow assertion, kernel `objective_i64` and helper `objective_value`; the inner `forward` now also returns the final distribution field |
| `lbm/lbm3d_immersed_boundary.py` | unchanged | added `DETERMINISTIC_ACCUMULATION`, `ACC_SCALE = 2**55` with an overflow assertion and kernels `spread_i64`, `zero4i`, `dequant4`; the force-spreading step runs zero/spread/dequantise in int64 fixed point |
| `wave_fdtd/diff_wave_3d.py` | unchanged | added `DETERMINISTIC_ACCUMULATION`, `ACC_SCALE = 2**45` with an overflow assertion, kernel `energy_i64` and helper `energy_value`; `forward` now returns `(loss, p)` so the reported value can be recomputed order-invariantly |
| `wave_fdtd/diff_wave_substrate.py` | unchanged | added `DETERMINISTIC_ACCUMULATION`, `ACC_SCALE = 2**48` with an overflow assertion, kernel `focus_loss_i64` and helper `focus_loss_value`; the finite-difference checks and `J0` use the int64 value |
| `wave_fdtd/xray_3d_dda.py` | unchanged | added `DETERMINISTIC_ACCUMULATION`, `ACC_SCALE = 2**35` with an overflow assertion, kernels `backproject_i64`, `dequant3` and helper `backproject_value`; the sensitivity map and every SIRT update go through it |
| `wave_fdtd/xray_tomography_sigma.py` | unchanged | added `DETERMINISTIC_ACCUMULATION`, `CN_SCALE = 2**43` and `LOSS_SCALE = 2**35` with overflow assertions, kernels `sensitivity_i64`, `dequant2`, `sq_resid_i64` and helpers `sensitivity_map`, `forward_loss_value` |
| `docs/RUNNING.md` | - | 1 Status row added (`determinism_sweep_int64.py`), 8 Status notes extended with the determinism measurement, count line updated to 293 module rows, new section "Deterministic accumulation" with the per-site and per-selftest tables and the list of sites deliberately left on the float path |
| `tests/test_determinism_sweep_int64.py` | - | new pytest wrapper for the sweep cell; skips without a CUDA device |

| `certified_kernels/innovation_adjoint_baseline.py` | Add separate frozen-site/eight-selftest observer retaining returncodes, exact stdout hashes and aligned numeric deltas. | Measure all adjoint-carrying cases before design without changing the existing sweep or exemptions. |
| `wave_fdtd/diff_wave_3d_gather.py` | Add separate one-writer gather adjoint with frozen forward kernels, explicit history and whole-gradient/FD gates. | Address the measured wave3D backward nondeterminism without modifying its reference. |
| `certified_kernels/innovation_runtime_adjoint.py` | Add separate process harness selecting runtime RUN_TO_RUN before frozen-module imports, with exact text and strict error gates. | Test available segmented adjoint machinery against the measured baseline without editing reference modules. |
| `kernel_gen/innovation_lbm_stream_baseline.py` | Add frozen D2Q9 stream measurement with independent CPU index oracle, two isolated workers and event/wall timings. | Establish the memory mechanism before a C-host export design. |
| `certified_kernels/innovation_tomography_record_bound.py` | Add separate bounded runtime wrapper for the frozen110-sample tomography loop, deriving440 records and retaining original selftest/error gates. | Diagnose measured default scatter overflow without changing the comparator or tolerances. |
| `certified_kernels/innovation_runtime_bounded_sweep.py` | Add composed eight-case observer using the measured default wrapper plus the separate bounded tomography wrapper. | Verify full-suite text repeatability while retaining the independent all-physics-pass gate and its known negative. |
| `kernel_gen/innovation_lbm_stream_export.py` | Add isolated Warp/C-host comparison with raw fixtures, frozen CPU oracle, generated-body check, strict idle guards and two event/wall timing legs. | Test the registered exact-output and10% bandwidth gates without editing the measured stream baseline. |
| `kernel_gen/stream_export_v1/stream_generated.cu` | Export only the frozen Warp1.17 stream forward function; remove WP_NO_CRT and source comments, retain generated operations. | Standalone CUDA translation unit beside the prior reduction export. |
| `kernel_gen/stream_export_v1/stream_api.h` | Add plain C declaration taking host buffers and returning timing/output. | Keep generated types behind a C-callable seam. |
| `kernel_gen/stream_export_v1/stream_shim.cu` | Add native memory/array descriptors, fixed launch, repeated-output check and CUDA-event/wall timing. | Reproduce the frozen launch without linking Warp runtime. |
| `kernel_gen/stream_export_v1/host_stream.c` | Add gcc-compiled C11 raw-input/output caller. | Demonstrate an actual C host rather than only a C++ declaration. |
| `kernel_gen/stream_export_v1/build.sh` | Add separate gcc/nvcc build using existing licensed headers and explicit architecture. | Keep the prior reduction export and build frozen. |
| `kernel_gen/optix_sample_probe.py` | Add a two-run headless external SDK sample probe with exact image bytes, nontrivial pixel gates and an optional current-boot fault guard. | Establish the installed RT toolchain before designing a shared backend; no SDK source or binary is vendored. |
| `kernel_gen/rt_winding_v1/params.h` | Add native launch-parameter seam for triangle corners, ray origins and signed winding output. | Share a minimal host/device ABI beside existing exports. |
| `kernel_gen/rt_winding_v1/program.cu` | Add built-in triangle any-hit signed winding, one invocation per primitive and strict positive ray parameter. | Preserve measured winding semantics instead of parity. |
| `kernel_gen/rt_winding_v1/host.cpp` | Add bounded single-shot raw-fixture caller, GAS build and independent output arrays. | Execute the new backend without modifying Warp comparators. |
| `kernel_gen/rt_winding_v1/build.sh` | Add external SDK/toolkit build with no vendored SDK files. | Reuse the measured compatible toolchain. |
| `kernel_gen/rt_winding_phase_probe/host.cpp` | Add separate native phase timestamps and GAS-completion boundary beside the frozen host, reusing unchanged launch parameters/PTX. | Attribute measured lifecycle cost before persistent API design. |
| `kernel_gen/rt_winding_phase_probe/build.sh` | Add external SDK/toolkit CPU build for observer only. | Preserve the verified native executable and device program. |
| `kernel_gen/rt_winding_batch_v1/host.cpp` | Add separate fixed64-query context/pipeline/GAS reuse with full upload/readback and exact within-batch comparisons. | Test amortization suggested by the measured phase table without modifying the stable host/PTX. |
| `kernel_gen/rt_winding_batch_v1/build.sh` | Add external-toolchain build for separate batch caller. | Preserve prior binaries and launch program. |
| `kernel_gen/rt_winding_api_v1/api.h` | Add explicit same-thread immutable-mesh C ABI with fixed capacity, status codes and caller-owned output. | Define mutable-query ownership beside the frozen executable. |
| `kernel_gen/rt_winding_api_v1/api.cpp` | Add separate persistent handle setup/query/cleanup, finite/count rejection before transfer, unchanged winding PTX. | Test changed ray origins against measured CPU references. |
| `kernel_gen/rt_winding_api_v1/build.sh` | Add external SDK/toolkit shared-library build. | Preserve all prior binaries and avoid SDK vendoring. |
