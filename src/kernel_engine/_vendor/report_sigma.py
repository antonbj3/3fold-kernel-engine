"""
SIGMA-DISCIPLINE helper: committed headline numbers should carry
{value, sigma} so the lattice-birth hook (per-tick chi-scan) can fire value-conflicts. Measured need:
hook sigma-coverage = 0.28% of 12,692 corpus numbers -> the value-conflict leg is near-vacuous without this.

USAGE in any probe report dict:
    from report_sigma import sig
    rep["lambda_hat"], rep["lambda_hat_sigma"] = sig(0.864, ci=(0.718, 1.015))   # from a 95% CI
    rep["floor_us"],   rep["floor_us_sigma"]   = sig(18.43, sd=0.9)              # from a measured sd
The sibling-key convention (<name>_sigma) is exactly what the hook's walker detects.
"""


def sig(value, sd=None, ci=None, se=None):
    """Return (value, sigma) with sigma derived from whichever uncertainty object the probe has."""
    if sd is not None:
        s = float(sd)
    elif se is not None:
        s = float(se)
    elif ci is not None:
        s = abs(ci[1] - ci[0]) / 3.92  # 95% CI width -> sigma
    else:
        raise ValueError("sigma-discipline: give sd=, se= or ci= — a bare headline number is hook-invisible")
    return float(value), s
