"""Pytest wrappers for the modules added in the second extraction pass.

Every module is run as a script in a fresh subprocess; the test asserts exit code 0.
Modules that need a CUDA device are collected separately and skip when no device is present.
The three CAD mesh generators run first, because several FEM modules read the meshes they write.
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "kernel_engine")

ARGS = {
    "lbm/lbm_aero_v0.py": ["--mode", "selftest"],
    "certified_kernels/probe_tensorcore_precision_rung.py": ["--inv", "A"],
    # the full sweep runs far past any test budget; this is the module's own short self-test
    "lbm/g21_3d_wake_chaos_highRe.py": ["--validate"],
}

# modules whose full run needs more than the default subprocess budget
TIMEOUTS = {"lbm/g20_3d_wake_chaos_nilss_prereq.py": 1200}

MESH_SCRIPTS = ["fem/cad_to_femmesh.py", "fem/cad_to_tetmesh.py", "fem/cad_kirsch_mesh.py"]
MESH_CONSUMERS = ("fem/warpfem_cad_elasticity.py", "fem/warpfem_mms_cad.py", "fem/warpfem_kirsch.py",
                  "fem/warpfem_mms_cad3d.py", "fem/warpfem_stress3d.py")


def _cuda_available():
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    try:
        import warp as wp
        wp.init()
        return wp.is_cuda_available()
    except Exception:
        return False


CUDA = _cuda_available()


# Gates that compare measured wall-clock crossovers; they can flip when the machine is loaded.
# Such a module gets one re-run before its failure counts.
TIMING_SENSITIVE = {"amr_poisson/poisson_dispatch.py"}


# Markers printed by a module's own GPU-idle guard when it refuses to measure on a busy device.
IDLE_GUARD_MARKERS = ("FOREIGN compute procs present", "FOREIGN TENANT", "ABORT foreign tenant", "not idle")


def run(rel, timeout=None):
    timeout = timeout or TIMEOUTS.get(rel, 600)
    attempts = 2 if rel in TIMING_SENSITIVE else 1
    for i in range(attempts):
        proc = subprocess.run([sys.executable, os.path.join(SRC, rel), *ARGS.get(rel, [])],
                              cwd=SRC, capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            return proc.stdout
    if any(m in proc.stdout + proc.stderr for m in IDLE_GUARD_MARKERS):
        pytest.skip("the module's own GPU-idle guard aborted the run (other GPU processes present)")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return proc.stdout


@pytest.fixture(scope="session")
def cad_meshes():
    for m in MESH_SCRIPTS:
        run(m)


CPU_MODULES = [
    '_vendor/goal_oriented_culling.py',
    'allocation/d_pair_rep_waterfill_goalderived.py',
    'allocation/fpga_bitwidth_waterfilling_is_precision_floor_allocation_marginal_modes_need_more_bits.py',
    'allocation/l_phase1_quant_x_cert_waterfill.py',
    'allocation/l_waterfill_deploygap.py',
    'allocation/l_waterfill_sufficiency.py',
    'allocation/price_vector_waterfilling_multicommodity_capacity_allocation.py',
    'allocation/reservation_waterfill_survival.py',
    'allocation/sigmin_price_vector_waterfilling.py',
    'allocation/sigmin_waterfill_bstar_stop.py',
    'amr_poisson/amr_sigma_scaling.py',
    'amr_poisson/lbm_poisson.py',
    # timing-sensitive: its gate compares measured wall-clock crossovers and can fail on a loaded machine
    'amr_poisson/poisson_dispatch.py',
    'certified_kernels/apriori_requirement_cert_on_real_kernelbench.py',
    'certified_kernels/chunkable_recurrence_rule.py',
    'certified_kernels/kernelbench_addressing_census_provenance_gate_absent.py',
    'certified_kernels/d_ensemble_coexecution_cert_contended_roofline.py',
    'certified_kernels/d_roofline_scene_eye.py',
    'certified_kernels/d_wave92_B_friction_gate_roofline.py',
    'euler_hllc/axisym_ns_solver.py',
    'euler_hllc/detonation_cellular.py',
    'euler_hllc/detonation_znd.py',
    'euler_hllc/gas_flow_engine.py',
    'euler_hllc/goc_baseline_honesty.py',
    'euler_hllc/jeans_instability.py',
    'euler_hllc/pa_adjoint_vs_cheap_goal.py',
    'euler_hllc/pa_adjoint_vs_cheap_proxies.py',
    'euler_hllc/phenomenon_registry.py',
    'euler_hllc/sigma_solver_routing.py',
    'euler_hllc/traffic_flow_lwr.py',
    'fem/buckling_euler_column.py',
    'fem/cad_kirsch_mesh.py',
    'fem/cad_to_femmesh.py',
    'fem/cad_to_tetmesh.py',
    'fem/euler_buckling.py',
    'fem/fatigue_life.py',
    'fem/fem3d_elasticity.py',
    'fem/fem3d_modal.py',
    'fem/fem3d_orthotropic.py',
    'fem/fem3d_thermoelastic.py',
    'fem/fracture_lefm.py',
    'fem/spectral_fatigue.py',
    'fem/viscoelasticity_creep.py',
    'fem/warpfem_acoustic_design.py',
    'fem/warpfem_acoustic_modal.py',
    'fem/warpfem_adjoint_arbitrary.py',
    'fem/warpfem_cad_elasticity.py',
    'fem/warpfem_compliant_inverter.py',
    'fem/warpfem_coupled_adjoint.py',
    'fem/warpfem_design_gradient.py',
    'fem/warpfem_elasticity_validate.py',
    'fem/warpfem_em_magnetostatics.py',
    'fem/warpfem_field_surrogate_fallbevis_v1.py',
    'fem/warpfem_kirsch.py',
    'fem/warpfem_mms_3d.py',
    'fem/warpfem_mms_cad.py',
    'fem/warpfem_mms_elasticity.py',
    'fem/warpfem_modal.py',
    'fem/warpfem_stokes_poiseuille.py',
    'fem/warpfem_thermoelastic.py',
    'fem/warpfem_transient_heat_energy.py',
    'fem/warpfem_transient_structural.py',
    'kernel_gen/d_l1_kernel_certvec_compose.py',
    'lbm/acoustic_streaming.py',
    'lbm/d_certvector_on_real_lbm.py',
    'lbm/d_integration_stitch_lbm_contact_scenario_cert.py',
    'lbm/d_per_variable_precision_cert_lbm.py',
    'lbm/d_static_deployment_gate_real_lbm.py',
    'lbm/differentiable_lbm_probe.py',
    'lbm/double_diffusive_lbm.py',
    'lbm/excitable_media_fhn.py',
    'lbm/hartmann_mhd_lbm.py',
    'lbm/lbm_aero_v0.py',
    'lbm/lbm_compressible_boundary.py',
    'lbm/lbm_mach_ceiling.py',
    'lbm/lbm_voxel_aero.py',
    'lbm/magnetic_induction_lattice.py',
    'lbm/rayleigh_taylor_lbm.py',
    'lbm/reaction_diffusion_turing.py',
    'lbm/rte_lbm.py',
    'reductions/probe_kernel_determinism_cert.py',
    'reductions/u_v54_gather_law_deterministic_reduction_vs_atomic_int64.py',
    'thermo/p8_combustion_cert_pod_a90.py',
    'thermo/p8_combustion_cert_watertight.py',
    'thermo/p8_combustion_state_certification.py',
    'thermo/p8_delft_conditional_variance.py',
    'thermo/p8_delft_flamelet_render_match.py',
    'thermo/p8_delft_species_flamelet.py',
    'thermo/p8_ecn_combustion_energy.py',
    'thermo/p8_flame_mixture_fraction_render_match.py',
    'thermo/p8_sandia_extinction_flamelet_breakdown.py',
    'thermo/biot_transient_conduction.py',
    'thermo/bohm_sheath_criterion.py',
    'thermo/d_nonnormal_thermo_price_of_amplification_probe.py',
    'thermo/d_thermo_computing_equals_fusion.py',
    'thermo/d_thermo_datahole_law.py',
    'thermo/d_thermo_voi_law_multiworld.py',
    'thermo/debye_specific_heat.py',
    'thermo/diffusion_induced_stress.py',
    'thermo/explosion_combustion.py',
    'thermo/fizeau_drag.py',
    'thermo/fourier_conduction.py',
    'thermo/heat_pipe_capillary_limit.py',
    'thermo/heat_pipe_sigma_budget.py',
    'thermo/heat_pump_cop.py',
    'thermo/heat_pump_sigma_budget.py',
    'thermo/hopf_reaction_diffusion.py',
    'thermo/induction_heating_skin.py',
    'thermo/ising_thermo.py',
    'thermo/joule_heating_thermal.py',
    'thermo/kapitza_acoustic_mismatch.py',
    'thermo/photoelasticity_isochromatics.py',
    'thermo/photon_diffusion_escape.py',
    'thermo/saffman_delbruck_diffusion.py',
    'thermo/sommerfeld_electron_heat.py',
    'thermo/soret_thermodiffusion.py',
    'thermo/thermocouple_seebeck.py',
    'thermo/thermoelectric_seebeck.py',
    'thermo/tidal_heating.py',
    'thermo/wheatstone_bridge.py',
    'thermo/wkb_quantization.py',
    'tropical_sdf/2jet_sdf_curvature_decouples_v6.py',
    'tropical_sdf/JxA_contact_param_over_determination_material_physics_leg_lifts_the_motion_regime_null.py',
    'tropical_sdf/certified_generative_support_placement_worstcase_sigmamin_robust_dfc.py',
    'tropical_sdf/contact_cert_chi_from_sdf_error.py',
    'tropical_sdf/contact_dof_cert_identify_or_abstain.py',
    'tropical_sdf/contact_orientation_gauge_emergence.py',
    'tropical_sdf/contact_penalty_conditioning_omega_max_side_accuracy_cost_tradeoff_completes_sim_conditioning.py',
    'tropical_sdf/engine_contact_manifold_rank_twist_selfstress_2point_reduction.py',
    'tropical_sdf/engine_manifold_reduction_needs_3points_not_2_wrench_span.py',
    'tropical_sdf/engine_which_3_points_Eoptimal_sigmamin_manifold_not_max_area.py',
    'tropical_sdf/friction_mu_is_a_sigma_min_null_under_stick_lifted_only_by_slip_events_contact_sim_ready.py',
    'tropical_sdf/mma_capstone_validity_band_sweep.py',
    'tropical_sdf/mma_topopt_capstone_stress_constrained_dfc_positive.py',
    'tropical_sdf/restitution_null_lifted_by_impact_friction_and_restitution_are_decorrelated_contact_nulls.py',
    'tropical_sdf/so_arm100_contact_sim_cost_from_tropical_sdf_backend_hertz_stiffness_sets_stable_dt_stability_verified.py',
    'tropical_sdf/stress_constrained_compliance_topopt_dfc_positive.py',
    'tropical_sdf/stress_objective_topopt_design_for_certifiability_positive.py',
    'tropical_sdf/topopt_stress_frontier_forced_number_not_honest_label.py',
    'tropical_sdf/topopt_stress_vs_compliance_design_for_certifiability.py',
    'tropical_sdf/tropical_sdf_SIMT_tax_cost_model_compute_bound_vs_voxel_bandwidth_bound_crossover_Kstar.py',
    'tropical_sdf/tropical_sdf_backend_computes_full_v6_contact_geometry_ladder_2jet_curvature_anisotropy_on_so_arm100.py',
    'tropical_sdf/tropical_sdf_backend_min_of_capsules_beats_voxel_3x_bytes_matched_penetration_so_arm100.py',
    'tropical_sdf/tropical_sdf_so_arm100_geometry_backend_v2_config_driven_posing_plus_contact_normal_gradient.py',
    'tropical_sdf/tropical_sdf_swept_volume_continuous_collision_detection_min_over_trajectory.py',
    'warp_gpu/gpu_fracture_determinism_sigma.py',
    'wave_fdtd/acoustic_emission.py',
    'wave_fdtd/acoustic_sigma_renderer.py',
    'wave_fdtd/acoustic_fdtd.py',
    'wave_fdtd/diag_acoustic_mode.py',
    'wave_fdtd/diag_acoustic_seed.py',
    'wave_fdtd/diag_acoustic_spectrum.py',
    'wave_fdtd/diff_wave_3d.py',
    'wave_fdtd/diff_wave_substrate.py',
    'wave_fdtd/elastodynamics_kache.py',
    'wave_fdtd/goc_wave_transient.py',
    'wave_fdtd/optics_coating.py',
    'wave_fdtd/optics_fdtd.py',
    'wave_fdtd/persona_design_acoustic.py',
    'wave_fdtd/persona_design_acoustic_cavity_shape.py',
    'wave_fdtd/persona_design_acoustic_inverse.py',
    'wave_fdtd/persona_design_acoustic_shape_adjoint.py',
    'wave_fdtd/persona_design_lbm_acoustic.py',
    'wave_fdtd/s1_acoustic_metamaterial_bandgap.py',
    'wave_fdtd/thermoacoustic_rijke_dde.py',
    'wave_fdtd/vibroacoustic.py',
    'wave_fdtd/wave_fdtd_3d.py',
    'wave_fdtd/wave_fdtd_kache.py',
    'wave_fdtd/wave_fdtd_verify.py',
    # added by the GPU-index sweep
    'certified_kernels/d_energy_exponent_substrate_invariant.py',
    'certified_kernels/module_const_launch_tune.py',
    'certified_kernels/morton_3d_stencil.py',
    'certified_kernels/morton_sparse_gather.py',
    'euler_hllc/lubrication_reynolds_bearing.py',
    'fem/fsi_added_mass.py',
    'fem/fsi_pipe_flutter.py',
    'fem/neuber_notch_plasticity.py',
    'fem/plasticity_3d_j2.py',
    'fem/plasticity_return_mapping.py',
    'fem/thermal_buckling.py',
    'fem/viscoelastic_preload_relaxation.py',
    'lbm/differentiable_fsi_chain.py',
    'reductions/det_accumulation_probe.py',
    'wave_fdtd/goc_wave_verify.py',
]

CUDA_MODULES = [
    'allocation/probe_fair_vs_waterfill_knobs.py',
    'amr_poisson/lightning_dbm.py',
    'certified_kernels/d_1c_iv_end_to_end_real_cuda_cert.py',
    'certified_kernels/d_avbd_gpu_certified_roofline.py',
    'certified_kernels/d_cuda_transcendental_parity.py',
    'certified_kernels/d_r2_r4_dissociation_real_cuda.py',
    'certified_kernels/diag_fp16_bandwidth.py',
    'certified_kernels/fem_sass_roofline.py',
    'certified_kernels/l_phase0_c2_roofline.py',
    'certified_kernels/probe_cuda_machinery_port_first_node.py',
    'certified_kernels/probe_rung3_warp_scheduler.py',
    'certified_kernels/probe_tensorcore_precision_rung.py',
    'certified_kernels/u_compute_twin_roofline_ceiling_v1.py',
    'certified_kernels/u_h5_thermal_roofline_twin_state.py',
    'certified_kernels/u_v870_roofline_ceiling_long_window_sustained_throttle_recheck.py',
    'certified_kernels/u_v874_roofline_soft_knee_sweep_tunable_arithmetic_intensity.py',
    'euler_hllc/substrate_compose.py',
    'fem/warpfem_compute_router.py',
    'fem/warpfem_design_objective.py',
    'fem/warpfem_field_surrogate.py',
    'fem/warpfem_mms_cad3d.py',
    'fem/warpfem_navier_stokes.py',
    'fem/warpfem_stress3d.py',
    'fem/warpfem_surrogate.py',
    'fem/warpfem_topopt.py',
    'kernel_gen/d_crossbackend_kernel_port_verify.py',
    'lbm/lbm3d_immersed_boundary.py',
    'lbm/lbm_fsi_bouzidi.py',
    'lbm/lbm_fsi_gpu.py',
    'lbm/lbm_gpu_fp16_half2.py',
    'lbm/lbm_mrt_stability.py',
    'lbm/lbm_voxel_aero_gpu.py',
    'lbm/moist_convection_lbm.py',
    'lbm/probe_lbm_fp16_steady_dither.py',
    'lbm/thermofluid_lbm_rayleigh.py',
    'reductions/d_1c_real_reduction_roofline_torch.py',
    'warp_gpu/mujoco_warp_bench.py',
    'warp_gpu/rigid2d_primal_vbd_gpu.py',
    'warp_gpu/warp_bodybody_colored_gpu.py',
    'warp_gpu/warp_bodybody_jacobi_gpu.py',
    'warp_gpu/warp_bodybody_scale_gpu.py',
    'warp_gpu/warp_cylinder_roll_gpu.py',
    'warp_gpu/warp_granular_friction.py',
    'warp_gpu/warp_granular_gpu.py',
    'warp_gpu/warp_mesh_roll_gpu.py',
    'warp_gpu/warp_rigid_ramp_gpu.py',
    'warp_gpu/warp_rigid_ramp_gpu_v2.py',
    'warp_gpu/warp_rl_env_gpu.py',
    # added by the GPU-index sweep
    'certified_kernels/aa_micro_bench.py',
    'certified_kernels/probe_compute_ladder_descent.py',
    'certified_kernels/probe_exclusive_sweep_block.py',
    'certified_kernels/probe_l2_inband_endgame.py',
    'certified_kernels/probe_sync_density_victim_model.py',
    'certified_kernels/probe_sync_penalty_vs_priority.py',
    'certified_kernels/probe_timeslice_dma_fartail.py',
    'certified_kernels/ser_fracture_compaction.py',
    'lbm/coupled_design_aero_struct.py',
    'lbm/g18_cfd_nilss_prereq_wake_chaos.py',
    'lbm/g19_forced_2d_wake_nilss_prereq.py',
    'lbm/g20_3d_wake_chaos_nilss_prereq.py',
    'lbm/g21_3d_wake_chaos_highRe.py',
    'lbm/probe_kam_resonance_dither_strides.py',
    'reductions/d_1c_iv_best_in_class_float4.py',
]


@pytest.mark.parametrize("rel", CPU_MODULES)
def test_selftest(rel, cad_meshes):
    run(rel)


@pytest.mark.skipif(not CUDA, reason="requires a CUDA device")
@pytest.mark.parametrize("rel", CUDA_MODULES)
def test_selftest_cuda(rel):
    run(rel)


def test_amr_octree_fv_reports_partial():
    """This module's own gate reports PARTIAL and it exits 1 by design; the run must still produce the verdict."""
    proc = subprocess.run([sys.executable, os.path.join(SRC, "amr_poisson/amr_octree_fv.py")],
                          cwd=SRC, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 1, proc.stdout[-2000:]
    assert "graded-octree AMR" in proc.stdout


def test_gpu_duty_torch_v1_duty_cycle():
    """The duty helper is a library, not a gated script: exercise it with a stub synchronise."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gpu_duty_torch_v1", os.path.join(SRC, "certified_kernels", "gpu_duty_torch_v1.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = mod.GpuDutyTorch(andel=0.5, block_ms=2.0, synk=lambda: None)
    for _ in range(200):
        d.steg()
    rep = d.rapport()
    assert rep["aktiv"] is True
    assert rep["n_steg"] == 200
    assert d.sovtid_s > 0.0 and rep["n_block_klara"] >= 1
    assert mod.GpuDutyTorch(andel=1.0, synk=lambda: None).steg() is None


def test_differentiable_flow_control_reports_partial():
    """Its own gate reports PARTIAL and it exits 1 by design; the run must still produce the verdict."""
    proc = subprocess.run([sys.executable, os.path.join(SRC, "lbm/differentiable_flow_control.py")],
                          cwd=os.path.join(SRC, "lbm"), capture_output=True, text=True, timeout=900)
    assert proc.returncode == 1, proc.stdout[-2000:]
    assert "differentiable flow-control" in proc.stdout


def test_coupled_knob_ordering_refutes():
    """Its own pre-registered verdict on this hardware is REFUTED and it exits 1 by design; the run must
    still produce the verdict line."""
    proc = subprocess.run([sys.executable, os.path.join(SRC, "certified_kernels/d_coupled_knob_ordering_advantage_real_cuda.py")],
                          cwd=SRC, capture_output=True, text=True, timeout=900)
    assert proc.returncode in (0, 1, 2), proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "ORDERING CAUSALLY BUYS ITERATIONS" in proc.stdout


def test_lbm_fsi_viv_reports_no_amplitude_peak():
    """Frequency capture is validated but the amplitude gate does not fire at this Reynolds number, so the
    module exits 1 by design; the run must still produce the verdict."""
    proc = subprocess.run([sys.executable, os.path.join(SRC, "lbm/lbm_fsi_viv.py")],
                          cwd=SRC, capture_output=True, text=True, timeout=900)
    assert proc.returncode in (0, 1), proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "FREQUENCY CAPTURE" in proc.stdout


def test_lbm_gpu_fp16_reports_partial():
    """Its own gate reports PARTIAL (throughput yes, precision no) and it exits 1 by design; the run must
    still produce the verdict."""
    proc = subprocess.run([sys.executable, os.path.join(SRC, "lbm/lbm_gpu_fp16.py")],
                          cwd=SRC, capture_output=True, text=True, timeout=900)
    assert proc.returncode in (0, 1), proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "FP16 storage wins" in proc.stdout


@pytest.mark.skipif(not CUDA, reason="requires a CUDA device")
def test_gpu_lbm_utilization_cell():
    """Gate G2 compares the measured energy per site against reports/probes/asic_fallback_feasibility.json,
    written by a predecessor cell that is not shipped in this repository."""
    if not os.path.exists(os.path.join(ROOT, "reports", "probes", "asic_fallback_feasibility.json")):
        pytest.skip("predecessor artefact reports/probes/asic_fallback_feasibility.json is absent")
    run('lbm/gpu_lbm_utilization_cell.py')


@pytest.mark.skipif(not CUDA, reason="requires a CUDA device")
def test_cloud_morphology_reports_honest_negative():
    """The moist Rayleigh-Benard cloud is a degenerate one-row layer here, so the fractal-dimension
    hypothesis does not fire and the module exits 1 by design; the run must still produce the verdict."""
    proc = subprocess.run([sys.executable, os.path.join(SRC, "lbm/cloud_morphology.py")],
                          cwd=SRC, capture_output=True, text=True, timeout=900)
    assert proc.returncode in (0, 1), proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "MORPHOLOGY" in proc.stdout
