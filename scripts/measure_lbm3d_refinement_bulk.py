"""Independent bulk/shear refinement gate before turbulent validation."""
from measure_lbm3d_refinement_sgs import main

if __name__=="__main__":
    raise SystemExit(main(recursive=True,bulk_tau_fine=1.,output="reports/lbm3d_refinement_bulk_v1/report.json"))
