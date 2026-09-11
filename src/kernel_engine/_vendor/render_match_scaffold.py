"""Prediction-band comparison helper (vendored).

render_match(render_fn, param_band, best_params, benchmark, nulls, perturbations) evaluates render_fn
over a band of input parameters spanning the genuine modelling uncertainty, and reports whether the
measured benchmark value falls inside the resulting prediction band, together with the signed gap of
the best estimate. Null cases must collapse and perturbations must move the prediction the right way.

Benchmark(name, measured, sigma, source, units) is the measurement record.
Returns a RenderMatch with .ok and .report().
"""
import sys
from dataclasses import dataclass, field


@dataclass
class Benchmark:
    name: str
    measured: float
    sigma: float            # measurement σ in the same units (0 if only a point value is published)
    source: str
    units: str = ""


@dataclass
class RenderMatch:
    benchmark: Benchmark
    band_lo: float
    band_hi: float
    best: float
    gap_pct: float
    in_band: bool
    bounds: str             # 'upper' / 'lower' / 'central' — where the measurement sits in the band
    nulls_ok: bool
    perts_ok: bool
    notes: list = field(default_factory=list)

    @property
    def ok(self):
        return self.in_band and self.nulls_ok and self.perts_ok

    def report(self):
        b = self.benchmark
        lines = [
            f"RENDER→MATCH [{b.name}] — measured {b.measured:g}{(' ±'+format(b.sigma,'g')) if b.sigma else ''} {b.units}",
            f"  rendered band [{self.band_lo:g}, {self.band_hi:g}] (=σ), best estimate {self.best:g}",
            f"  → measured at the {self.bounds.upper()} edge; best estimate {'OVER' if self.gap_pct>=0 else 'UNDER'}-predicts by {self.gap_pct:+.0f}% (the reported Sim2Real gap)",
            f"  (1) measured ∈ rendered band (closure within σ)   {'✓' if self.in_band else 'FAIL'}",
            f"  (2) NULL cases collapse                           {'✓' if self.nulls_ok else 'FAIL'}",
            f"  (3) PERTURBATIONS move the prediction correctly    {'✓' if self.perts_ok else 'FAIL'}",
        ] + [f"  · {n}" for n in self.notes]
        lines.append(f"  ⇒ {'render→match CLOSES (within σ, gap reported)' if self.ok else 'OPEN — fix at source'}")
        return "\n".join(lines)


def render_match(render_fn, param_band, best_params, benchmark, nulls=None, perturbations=None, notes=None):
    """render_fn(params)->scalar prediction; param_band=list of param objects spanning the DERIVED uncertainty;
    best_params=the central estimate; benchmark=Benchmark; nulls/perturbations=list of (label, params, predicate).
    predicate(prediction, reference)->bool, reference = measured (nulls) or best (perturbations)."""
    preds = [render_fn(p) for p in param_band]
    lo, hi = min(preds), max(preds)
    best = render_fn(best_params)
    m = benchmark.measured
    in_band = lo <= m <= hi
    gap = (best - m) / m * 100.0
    frac = (m - lo) / (hi - lo) if hi > lo else 0.5
    bounds = 'lower' if frac < 0.33 else ('upper' if frac > 0.67 else 'central')
    nlist, plist = nulls or [], perturbations or []
    nulls_ok = all(pred(render_fn(p), m) for _, p, pred in nlist)
    perts_ok = all(pred(render_fn(p), best) for _, p, pred in plist)
    return RenderMatch(benchmark, lo, hi, best, gap, in_band, bounds, nulls_ok, perts_ok, list(notes or []))


# ----------------------------------------------------------------------------------------------------------------------
def _selftest():
    """Self-contained check: a linear render function, a measurement inside the band, a null case
    that collapses and a perturbation that moves the prediction upwards."""
    def rfn(p):
        return 100.0 * p["scale"]

    res = render_match(
        rfn, [{"scale": 0.85}, {"scale": 1.25}], {"scale": 1.05},
        Benchmark("linear reference", 100.0, 2.0, "analytic", "units"),
        nulls=[("collapsed scale", {"scale": 0.01}, lambda q, m: q < m / 5)],
        perturbations=[("larger scale", {"scale": 1.4}, lambda q, best: q > best)],
        notes=["null collapses; perturbation raises the prediction"])
    print(res.report())
    ok = res.ok and 0.0 < res.gap_pct < 10.0
    print(f"scaffold self-test: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest())
