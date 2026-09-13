# Kernel cloud determinism matrix

| Module | Own gates | L4 / A10 / H100 observed max delta |
|---|---|---|
| export:1024 | PASS | 0 |
| export:512 | PASS | 0 |
| int64:lbm/differentiable_flow_control:probe_ux | PASS | 0 |
| int64:lbm/differentiable_flow_control:track_loss | PASS | 0 |
| int64:lbm/differentiable_fsi_chain:drag_force | PASS | 0 |
| int64:lbm/differentiable_lbm_probe:objective | PASS | 0 |
| int64:lbm/lbm3d_immersed_boundary:spread | PASS | 0 |
| int64:wave_fdtd/diff_wave_3d:energy | PASS | 0 |
| int64:wave_fdtd/diff_wave_substrate:focus_loss | PASS | 0 |
| int64:wave_fdtd/xray_3d_dda:backproject | PASS | 0 |
| int64:wave_fdtd/xray_tomography_sigma:sensitivity | PASS | 0 |
| int64:wave_fdtd/xray_tomography_sigma:sq_resid | PASS | 0 |
| selftest:lbm/differentiable_flow_control | PASS | 0 |
| selftest:lbm/differentiable_fsi_chain | PASS | 0 |
| selftest:lbm/differentiable_lbm_probe | PASS | 0 |
| selftest:lbm/lbm3d_immersed_boundary | PASS | 0 |
| selftest:wave_fdtd/diff_wave_3d | PASS | 0 |
| selftest:wave_fdtd/diff_wave_substrate | PASS | 0 |
| selftest:wave_fdtd/xray_3d_dda | PASS | 0 |
| selftest:wave_fdtd/xray_tomography_sigma | PASS | 0 |

Local RTX 5070: unmeasured. Export performance criteria are reported separately in JSON.
