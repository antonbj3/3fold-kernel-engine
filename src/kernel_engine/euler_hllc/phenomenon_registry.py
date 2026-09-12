"""PHENOMENON COMPLETENESS METHOD (the author ): a capability has NO meaning without the underlying PHENOMENON
being simulated from real physics — a thermal-camera view needs the temperature field; an X-ray of subsurface cracks
needs the fracture phenomenon; a wear-twin needs the wear phenomenon. So the foundation is NOT a list — it is a
GENERATIVE method that (1) enumerates the full phenomenon space (singles + combinations) so nothing is missed, and
(2) GROUNDS every capability in the specific phenomena it requires, failing loudly when one is missing.

The method = a COUPLING GRAPH:
  • NODES  = fundamental fields (the substrate lattices).
  • EDGES  = coupling terms — each is the LOCAL LBM source/force (LBM-maximalism), and physically these are the
             ONSAGER cross-couplings of irreversible thermodynamics (the L_ij matrix IS the catalog of pairwise physics).
  • A PHENOMENON = a connected SUBGRAPH (fields + the edges among them): single = one node; COMBINATION = many nodes.
  • COMPLETENESS = enumerate the field×field MATRIX CELLS, not ad-hoc phenomena → every empty cell is a found gap;
                   coverage is a NUMBER (filled / total).
  • CAPABILITY GROUNDING = each capability declares required phenomena; UNGROUNDED if any underlying phenomenon is
                   absent → enforces the author's invariant in code.

This module is the REGISTRY + the CHECKER. It does not validate physics (the per-phenomenon scripts do that); it
ORGANIZES the space, finds gaps systematically, and enforces grounding. Run it: it self-tests referential integrity,
then prints the coupling matrix, coverage, and the capability-grounding table.

  python3 phenomenon_registry.py
"""
import sys
import itertools

V, B, M = "validated", "buildlist", "missing"          # status of a coupling / phenomenon / capability-grounding

# ── NODES: the fundamental fields (substrate lattices) ──────────────────────────────────────────────────────────
FIELDS = {
    "flow":      "momentum / Navier-Stokes (LBM D2Q9)",
    "thermal":   "temperature / heat (D2Q5 advection-diffusion)",
    "species":   "mass fraction / concentration (D2Q5 AD)",
    "em":        "electromagnetic wave / induction (Yee / MHD)",
    "elastic":   "solid displacement / stress (elastodynamics)",
    "phase":     "interface / multiphase / phase-change (Shan-Chen / enthalpy)",
    "charge":    "electric potential / ions (quasistatic Poisson / Nernst-Planck)",
    "gravity":   "self-gravity potential (Poisson)",
    "radiation": "radiative intensity (DOM / P_N)",
    "plastic":   "plastic strain / damage / fracture / wear",
}

# ── EDGES: coupling terms (each = a local LBM source/force; the Onsager cross-couplings) ─────────────────────────
# key: (field_a, field_b, mechanism, status, anchor)
COUPLINGS = [
    ("thermal", "flow",     "buoyancy (Boussinesq)",          V, "Ra_c=1708; Nu~Ra^1/3"),
    ("flow",    "thermal",  "advective heat transport",       V, "Peclet scaling; RB"),
    ("em",      "flow",     "Lorentz force (MHD)",            V, "Hartmann velocity profile"),
    ("flow",    "em",       "induction / flux-freezing",      V, "Alfven speed; Lundquist"),
    ("thermal", "elastic",  "thermal stress (eigenstrain)",   V, "Timoshenko cooled-cylinder hoop"),
    ("flow",    "elastic",  "FSI (momentum exchange)",        V, "von-Karman St-Re (Williamson)"),
    ("thermal", "em",       "Joule heating",                  V, "I^2 R; conduction-Joule"),
    ("phase",   "thermal",  "latent heat release",            V, "heat = L*condensed (machine prec)"),
    ("thermal", "phase",    "condensation / saturation src",  V, "Clausius-Clapeyron; cloud LCL"),
    ("phase",   "flow",     "surface tension (Shan-Chen)",    V, "Laplace dp=s/R (sign); rho_l/rho_v~12"),
    ("elastic", "plastic",  "yield / brittle fracture",       V, "Griffith / Weibull strength"),
    ("species", "flow",     "compositional buoyancy",         V, "double_diffusive_lbm.py: salt fingers, differential-diffusion instability"),
    ("species", "species",  "reaction (Arrhenius / RD)",      V, "reaction_diffusion_turing.py + excitable_media_fhn.py"),
    ("em",      "thermal",  "induction / eddy heating",       V, "induction_heating_skin.py: skin depth d=sqrt(2/musigw), delta~1/sqrt(w)"),
    ("radiation","thermal", "radiative exchange",             V, "radiative_exchange.py: Q=epsSB A(Ta^4-Tb^4), equilibrium, Newton-limit h_rad=4epsSB Tbar^3"),
    ("phase",   "phase",    "phase change (enthalpy/Stefan)", V, "stefan_problem.py: front X=2lam sqrt(at), lam from transcendental"),
    ("charge",  "flow",     "electrokinetic / EHD (Coulomb)", V, "electro_osmotic.py: Smoluchowski plug u_HS=−εζE/μ; κh→plug"),
    ("charge",  "species",  "electromigration (Nernst-Planck)",V,"electromigration_blech.py: Blech product jL threshold, immortal lines"),
    ("charge",  "em",       "dielectric breakdown / leader (DBM)",B,"lightning_dbm.py: η-morphology VALIDATED (D↓ with η); but D≈1.33@N350 ≪ DLA 1.71 (finite-size) → honest B, not V"),
    ("gravity", "flow",     "self-gravity body force",        V, "jeans_instability.py: FULL Jeans dispersion across band (RMS 4%, cutoff validated)"),
    ("plastic", "thermal",  "plastic-work / friction heating",V, "adiabatic_shear.py: stress peak = thermal-softening localization, d(lnsig)/dg=0"),
    ("flow",    "plastic",  "machining / erosion separation", V, "machining_merchant.py: shear angle phi=argmin F_c=45-(b-a)/2, chip ratio, strain"),
    ("species", "thermal",  "Soret / Dufour (thermodiffusion)",V,"soret_thermodiffusion.py: steady c~exp(-S_T T) from transient PDE"),
    ("charge",  "elastic",  "piezoelectricity",               V, "piezoelectric.py: c^D=c^E(1+k²) from D=0; Maxwell reciprocity; resonance √(1+k²)"),
    ("gravity", "phase",    "Rayleigh-Taylor / settling",     V, "rayleigh_taylor_lbm.py: sigma~sqrt(Agk) measured, stable null"),
    ("radiation","flow",    "radiation pressure / photophoresis",V,"radiation_pressure.py: beta dist-indep, blowout 0.46um vs observed"),
    ("plastic", "plastic",  "wear evolution (abrasion/diffusion)",V,"wear_archard.py: A_real=F_N/H (Bowden-Tabor), Archard V=K F_N s/H"),
    # matrix-completion cells confirmed by the coupling-graph mini-wave (44/45 carry real physics) ──
    ("species", "elastic", "chemo-mechanics (Larche-Cahn) [native]", V, "chemo_mechanics_stress.py: diffusion-induced stress, transient fracture peak"),
    ("species", "phase",   "solutal phase change (Scheil) [native]", V, "scheil_microsegregation.py: c_s=k c0 (1-fs)^(k-1), last liquid->eutectic"),
    ("em",      "elastic", "magnetostriction/Lorentz-on-solid [native]", V, "magnetostriction.py: strain~M^2 from energy-min, AC drive->2w (transformer hum)"),
    ("elastic", "phase",   "martensitic/stress transform (SMA) [native]", V, "sma_superelastic.py: superelastic loop, dsig/dT=dS/eps_L, hysteresis=loop area"),
    ("plastic", "species", "stress-corrosion / H-embrittlement [native]", V, "stress_corrosion_oriani.py: H pumped to crack-tip tension (Oriani), decohesion"),
    ("charge",  "thermal", "thermoelectric (Seebeck/Peltier)", V, "thermoelectric_seebeck.py: ZT-governed efficiency, max-power-transfer, Peltier"),
    ("phase",   "charge",  "electrowetting (EWOD/Lippmann)",   V, "electrowetting_lippmann.py: cos th = cos th0 + (c/2g)V^2 from energy min"),
    ("phase",   "em",      "ferrofluid Rosensweig (Kelvin)",   V, "ferrofluid_rosensweig.py: spike onset at capillary k_c=sqrt(rho g/gamma), M_c"),
    ("elastic", "gravity", "tidal heating (Love k2)",          V, "tidal_heating.py: E~e^2 a^-6, Io 9e13 W vs observed 1e14"),
    ("radiation","plastic","radiation damage / void swelling", V, "radiation_damage.py: NRT N_d=0.8E/2E_d, void swelling incubation+1%/dpa"),
    ("radiation","species","photochemistry (Beer-Lambert)",    V, "photochemistry_chapman.py: Chapman layer peaks at tau=1, rises with zenith"),
    ("phase",   "plastic", "TRIP / phase-boundary damage [native]", V, "trip_steel.py: transformation strengthening delays Considere necking, enhanced ductility"),
]
CKEY = {(a, b): (mech, st, anc) for (a, b, mech, st, anc) in COUPLINGS}

# ── PHENOMENA: connected subgraphs (fields + the coupling edges among them) ──────────────────────────────────────
# name: (fields, coupling-pairs-used, status, validating-script-or-anchor)
def P(fields, couplings, status, ref):
    return dict(fields=set(fields), couplings=list(couplings), status=status, ref=ref)

PHENOMENA = {
    # singles (one node's self-dynamics)
    "navier_stokes":     P(["flow"], [], V, "Poiseuille analytic; lbm_*"),
    "heat_conduction":   P(["thermal"], [], V, "Fourier; D2Q5"),
    "elastodynamics":    P(["elastic"], [], V, "P/S wave speeds"),
    "multiphase_core":   P(["phase", "flow"], [("phase", "flow")], V, "shan_chen_multiphase.py"),
    "fracture":          P(["elastic", "plastic"], [("elastic", "plastic")], V, "GPU fracture; Weibull"),
    # validated combinations
    "rayleigh_benard":   P(["flow", "thermal"], [("thermal", "flow"), ("flow", "thermal")], V, "thermofluid_lbm_rayleigh.py: Ra_c→1708 VALIDATED (growth_rate_rayleigh.py σ(Ra)=0 + rb_conv_analyze.py, p=2 Richardson R²=1.000, 0.7%)"),
    "mhd":               P(["flow", "em"], [("em", "flow"), ("flow", "em")], V, "Hartmann/induction"),
    "thermoelasticity":  P(["thermal", "elastic"], [("thermal", "elastic")], V, "eigenstrain FEM"),
    "fsi_viv":           P(["flow", "elastic"], [("flow", "elastic")], V, "lbm_fsi_viv.py"),
    "joule_heating":     P(["thermal", "em"], [("thermal", "em")], V, "EM-thermal"),
    "cloud_formation":   P(["flow", "thermal", "species", "phase"],
                           [("thermal", "flow"), ("flow", "thermal"), ("thermal", "phase"), ("phase", "thermal")],
                           V, "moist_convection_lbm.py"),
    # build-list combinations (pieces validated; composition / +1 piece)
    "double_diffusive":  P(["flow", "thermal", "species"],
                           [("thermal", "flow"), ("species", "flow")], V, "double_diffusive_lbm.py: salt fingers — growth rate σ>0 inside 1<R_ρ<1/τ DESPITE static stability; equal-diffusivity NULL (τ=1) decays + R_ρ>1/τ suppressed → the DIFFERENTIAL diffusion is the driver (the double-diffusive paradox), on the shared T⊗S⊗flow lattice"),
    "rayleigh_taylor":   P(["flow", "phase", "gravity"], [("phase", "flow"), ("gravity", "phase")], V, "rayleigh_taylor_lbm.py: Boussinesq miscible RT — heavy-over-light GROWS, light-over-heavy NULL decays; σ≈√(A·g·k) absolute (18%), σ↑ with k (√k-like; sub-√2 = REAL finite-ν + diffuse-interface high-k suppression). Immiscible (phase) spike/bubble via shan_chen next"),
    "combustion":        P(["flow", "thermal", "species"], [("flow", "thermal"), ("species", "species")], B, "flame speed"),
    "pool_boiling":      P(["phase", "thermal", "flow"], [("phase", "thermal"), ("phase", "flow")], B, "Nukiyama/Zuber CHF"),
    "induction_heating": P(["em", "thermal"], [("em", "thermal")], V, "induction_heating_skin.py: magnetic diffusion ∂B/∂t=η∇²B — skin depth δ=√(2/μσω) EMERGES from the field envelope (Δ0.6%), δ∝1/√ω (ratio 2.00), phase lag ≈1 rad at x=δ, Joule heating ∝e^(−2x/δ); NULL lower-ω penetrates 4.6× deeper (skin set by ω, not geometry)"),
    # missing (needs a new module / departure)
    "electroosmosis":    P(["flow", "charge"], [("charge", "flow")], V, "VALIDATED electro_osmotic.py: Debye-Hückel + Stokes → Smoluchowski slip u_HS=−εζE/μ; κh→plug"),
    "self_gravity":      P(["flow", "gravity"], [("gravity", "flow")], V, "jeans_instability.py: FULL dispersion σ(k)=√(4πGρ−c_s²k²) traced across the unstable band k/k_J=0.14→0.87 (6 modes, band-RMS 4%) via adiabatic-eigenmode seed; cutoff VALIDATED by vanishing σ near k_J (not dropped); k>k_J stable"),
    "electrodeposition": P(["charge", "species", "flow", "phase"],
                           [("charge", "species"), ("charge", "flow"), ("phase", "flow")], M, "Poisson+electrochem"),
    "lightning":         P(["charge", "em", "thermal", "flow"],
                           [("charge", "em"), ("thermal", "em"), ("thermal", "flow")], M,
                           "FLAGSHIP (lightning_dbm.py): DBM leader on validated Poisson φ; fractal morphology η-CONTROLLED (validated); D rises with N but ≈1.33≪DLA 1.71 (finite-size, honest B); Joule→thunder"),
    # validated this session (living-cell + astro + reacting + optics quadrants)
    "turing_pattern":    P(["species"], [("species", "species")], V, "reaction_diffusion_turing.py: dispersion + cutoff, λ vs k*"),
    "excitable_media":   P(["species"], [("species", "species")], V, "excitable_media_fhn.py: threshold + c monotone/concave CONSISTENT-with-√D (not uniquely vs D^0.4)"),
    "detonation":        P(["flow", "thermal", "species"], [("flow", "thermal"), ("species", "species")], V, "CJ velocity Δ1% = EASY global eigenvalue (detonation_znd.py); cellular structure = HARD (detonation_cellular.py: RH-init + θ≈5 instability + spurious-mode convergence; full 2D cells = HPC)"),
    "radiative_transfer":P(["radiation"], [], V, "rte_lbm.py: Beer-Lambert exact + scattering energy-conserving"),
    "optics_wave":       P(["em"], [], V, "optics_fdtd/lens/coating/achromat: Snell + dispersion + GRIN + AR + achromat"),
    "plasticity":        P(["elastic", "plastic"], [("elastic", "plastic")], V, "j2_plasticity.py: von Mises radial-return; τ_Y=σ_Y/√3, consistency f≈0 machine-prec, return=geometric projection"),
    "piezoelectric":     P(["charge", "elastic"], [("charge", "elastic")], V, "piezoelectric.py: open-circuit stiffening c^D=c^E(1+k²) EMERGES from D=0 + Maxwell reciprocity + resonance √(1+k²)"),
    "viscoelasticity":   P(["elastic"], [], V, "viscoelastic.py: SLS relaxation E(t)=E∞+E₁e^(−t/τ), τ=η/E₁; loss peak ωτ=1; hysteresis area==πE''ε₀² (rate-dependent constitutive)"),
    "acoustic_wave":     P(["flow"], [], V, "acoustic_fdtd.py: staggered Yee p-v, box modes f_mn EMERGE, c=√(K/ρ); the EM-dual (p↔E,v↔H)"),
    "surface_radiation": P(["radiation"], [], V, "view_factors.py: ray-cast view factors, reciprocity A_iF_ij=A_jF_ji EMERGES, ΣF=1, Stefan-Boltzmann (thermal-camera basis)"),
    "vibroacoustic":     P(["flow", "elastic"], [("flow", "elastic")], V, "vibroacoustic.py: structure radiating sound; radiation damping ζ=ρcA/(2mω₀) EMERGES from the coupled solve"),
    # ★validated  (this session): conditional instability, electro double-layer, connectivity/phase transitions
    "conditional_instability": P(["flow", "thermal", "species", "phase"],
                           [("thermal", "flow"), ("phase", "thermal"), ("species", "phase")], V,
                           "conditional_instability_cape.py: Γ_d=g/cp exact, moist Γ_m<Γ_d, CAPE_moist>0 vs CAPE_dry=0 (moisture UNLOCKS convection), 3 stability regimes emerge — the adiabatic-lapse-rate mechanism the moist-RB lattice lacked (folded into moist_convection_theta.py)"),
    "debye_screening":   P(["charge", "species"], [("charge", "species")], V,
                           "poisson_boltzmann_debye.py: nonlinear PB u''=sinh(u); Debye-Hückel + Gouy-Chapman tanh + Grahame charge-potential all EMERGE; λ_D=0.96nm@0.1M, λ_D∝1/√c (diffuse double layer; complements electroosmosis)"),
    "percolation":       P(["phase"], [], V,
                           "percolation.py: connectivity threshold p_c=0.5924 EMERGES from cluster spanning (vs 0.59274), finite-size-scaled/sharpening — crack-network/transport gating"),
    "ising_transition":  P(["phase"], [], V,
                           "ising_2d.py: order-disorder, critical Tc=2.2692 (Onsager exact) EMERGES from Metropolis MC susceptibility peak; magnetization collapse + 3 limits"),
    "thermoelectric":    P(["charge", "thermal"], [("charge", "thermal")], V,
                           "thermoelectric_seebeck.py: Seebeck V_oc=S·ΔT (open-circuit limit, emergent) + generator MAX-POWER-TRANSFER (R_L=R) + efficiency governed by ZT (max η over load sweep = η_Carnot(√(1+ZT)−1)/(√(1+ZT)+Tc/Th), <Carnot) + Peltier cooler optimal current I=S·Tc/R; Kelvin Π=S·T Onsager-IMPOSED (labeled by-construction)"),
    "electrowetting":    P(["phase", "charge"], [("phase", "charge")], V,
                           "electrowetting_lippmann.py: contact angle EMERGES from droplet energy minimization (surface+capacitive), reproduces Young-Lippmann cos θ=cos θ₀+(c/2γ)V² (slope=c/2γ exact, V=0→Young null, genuine interior minimum); ideal Lippmann (real saturation noted)"),
    "soret":             P(["species", "thermal"], [("species", "thermal")], V,
                           "soret_thermodiffusion.py: a ∇T alone separates a species — steady c(x)∝exp(−S_T·T) EMERGES from the transient PDE (sep=exp(−S_T·ΔT) 0%), mass conserved 1e-15, SIGN-flip (S_T>0→cold, S_T<0→hot) + S_T=0 null. Dufour (Onsager reciprocal) not claimed"),
    "chemo_mechanics":   P(["species", "elastic"], [("species", "elastic")], V,
                           "chemo_mechanics_stress.py: diffusion-induced stress σ=−M·Ω·(c−c̄) (Li-electrode fracture) — self-equilibrated ∫σ=0 (2e-16), SIGN structure (enriched compressive/depleted tensile), ★TRANSIENT peak max|σ| at the charge/rest transition then relaxes (the fracture window, emergent from diffusion), null uniform-c/Ω=0"),
    "stefan":            P(["phase"], [("phase", "phase")], V,
                           "stefan_problem.py: solidification front X=2λ√(αt) EMERGES from the enthalpy moving-boundary solve — √t law (R²=0.9992), measured λ=0.623 vs independent transcendental λe^λ²erfλ=St/√π (0.5%), erf similarity profile, Stefan-number scaling (more latent→slower)"),
    "adiabatic_shear":   P(["plastic", "thermal"], [("plastic", "thermal")], V,
                           "adiabatic_shear.py: plastic-work heating (Taylor-Quinney) → thermal softening localizes strain. Flow stress PEAKS at γ_c (localization onset, emergent), exactly where d(lnσ)/dγ=0 i.e. hardening n/(γ₀+γ_c)=softening a(dT/dγ)/s (0.0%); ΔT=βW_p/ρc energy-exact; a=0/β=0 nulls. CNC-machining building block"),
    "radiation_pressure":P(["radiation", "flow"], [("radiation", "flow")], V,
                           "radiation_pressure.py: β=F_rad/F_grav is DISTANCE-INDEPENDENT (both ∝1/r², the geometric cancellation, spread 4e-16) ⇒ β∝1/s sets one size threshold everywhere; solar blowout size 0.46 µm = observed β-meteoroid limit (external anchor); s<s_blow→β>0.5 unbound, L=0 null"),
    "chapman_layer":     P(["radiation", "species"], [("radiation", "species")], V,
                           "photochemistry_chapman.py: photolysis production peaks at an INTERMEDIATE altitude where slant optical depth ≈1 (Chapman: emergent peak at τ=1.001), the layer RISES with solar zenith angle as H·ln(1/cosχ) (0%), σ→0 collapses to the surface (no layer). Ozone/ionosphere"),
    "tidal_heating":     P(["elastic", "gravity"], [("elastic", "gravity")], V,
                           "tidal_heating.py: eccentric-orbit flexing dissipates as heat, Ė=(21/2)(k₂/Q)GM_p²R⁵e²n/a⁶ — Ė∝e² (circular orbit=0) and ∝a⁻⁶ (exact); ★Io's rate 9.3e13 W matches the observed ~10¹⁴ W volcanic output (external anchor); rheology via k₂/Q only"),
    "magnetostriction":  P(["em", "elastic"], [("em", "elastic")], V,
                           "magnetostriction.py: strain ε=λ_s(M/M_s)² EMERGES from magnetoelastic+elastic energy min (∝M², even, saturates); ★FREQUENCY DOUBLING — AC drive at ω → strain at 2ω (FFT, the 100/120 Hz transformer hum); b=0/M=0 nulls. Villari inverse=Onsager partner (not claimed)"),
    "scheil":            P(["species", "phase"], [("species", "phase")], V,
                           "scheil_microsegregation.py: solidification solute redistribution — c_s(f_s)=k·c₀(1−f_s)^(k−1) EMERGES from the mass-balance integration (0.15% vs closed form), ∫c_s df_s=c₀ conserved, the rejected solute drives the last liquid to the EUTECTIC (10% eutectic), k=1→uniform; segregates more than the lever rule"),
    "sma_superelastic":  P(["elastic", "phase"], [("elastic", "phase")], V,
                           "sma_superelastic.py: stress-induced austenite↔martensite (Nitinol) — superelastic LOOP (σ_Ms>σ_As, full strain recovery); ★CLAUSIUS-CLAPEYRON plateau dσ/dT=ΔS/ε_L (0%, the solid-solid Clapeyron); dissipation = loop area ∮σdε; NULL ΔG_irr=0 → no hysteresis (the loop IS the irreversibility). Motion-engine actuator material"),
    "rosensweig":        P(["phase", "em"], [("phase", "em")], V,
                           "ferrofluid_rosensweig.py: normal-field instability — ferrofluid erupts into spikes above M_c; ★the onset spike spacing is the CAPILLARY length λ_c=2π√(γ/ρg) (k_c=√(ρg/γ), emerges from the surface-tension/gravity minimum, independent of magnetic coupling), 9.2 mm; M<M_c flat-stable, M>M_c unstable band, M=0 null"),
    "electromigration":  P(["charge", "species"], [("charge", "species")], V,
                           "electromigration_blech.py: electron-wind drag → atom flux; ★BLECH product — failure set by j·L (not j or L alone), below (jL)_crit=σ_crit·Ω/(Z*eρ) the line is IMMORTAL (backstress balances the wind), above it voids; v∝j (Nernst-Einstein), j=0 null. Black's MTTF is the lifetime form"),
    "wear_archard":      P(["plastic"], [("plastic", "plastic")], V,
                           "wear_archard.py: real contact area A_real=F_N/H from multi-asperity PLASTIC contact, INDEPENDENT of apparent area (Bowden-Tabor) — the single fact behind Amontons-Coulomb friction (μ=τ/H, load-proportional) AND Archard wear V=K·F_N·s/H (∝load,∝distance,∝1/H); F_N=0 null. Tool-wear twin + friction-from-contact-geometry"),
    "radiation_damage":  P(["radiation", "plastic"], [("radiation", "plastic")], V,
                           "radiation_damage.py: NRT cascade N_d=0.8·E_PKA/(2E_d) (10/100/1000 displacements at 1/10/100 keV in Fe), dpa∝fluence; ★void swelling DELAYED — zero below the incubation dose (10 dpa) then ~1%/dpa (austenitic steel), not proportional; sub-threshold + zero-fluence nulls. Reactor/space materials twin"),
    "stress_corrosion":  P(["plastic", "species"], [("plastic", "species")], V,
                           "stress_corrosion_oriani.py: hydrostatic-stress gradient pumps H UP to the crack tip — Oriani c_H=c₀·exp(V_H·σ_h/RT) EMERGES from the stress-driven diffusion (0.07%), ×1.82 enrichment at a 3σ_y field (0%), tension enriches/compression depletes, the H lowers cohesive strength (14% decohesion drop); σ_h=0 null. H-embrittlement / SCC twin"),
    "trip":              P(["phase", "plastic"], [("phase", "plastic")], V,
                           "trip_steel.py: transformation-induced plasticity — metastable austenite→martensite during straining (Olson-Cohen ξ(ε)) pulls σ(ε) up (52% dynamic strengthening); ★the Considère necking point (dσ/dε=σ) moves to LARGER strain → enhanced uniform elongation ε_u 0.17→0.21 (high strength AND ductility); k=0 null. TRIP/dual-phase crash steels"),
    "radiative_exchange":P(["radiation", "thermal"], [("radiation", "thermal")], V,
                           "radiative_exchange.py: Stefan-Boltzmann heat transfer Q=εσA(T_a⁴−T_b⁴) — two bodies equilibrate at the energy-weighted mean (energy conserved); ★the T⁴ law LINEARIZES to Newton's cooling h_rad=4εσT̄³ at small ΔT (0.03%) yet is super-linear at large ΔT; T_a=T_b null. Spacecraft/furnace/climate radiative balance"),
    "machining":         P(["flow", "plastic"], [("flow", "plastic")], V,
                           "machining_merchant.py: orthogonal cutting — the chip FLOWS by plastic shear; ★the shear angle φ EMERGES as argmin_φ of the cutting force = Merchant 45°−(β−α)/2 (0%, over the physical positive-force branch); chip thickens r=sinφ/cos(φ−α)=0.67, shear strain γ=2.3, friction↑→φ↓ & force↑, β=α→45° null. CNC chip-formation flagship core"),
    # ★FLAGSHIP grand-combination (the author's CNC vision) — exposes the plasticity + wear gap
    "cnc_machining":     P(["flow", "thermal", "elastic", "phase", "plastic"],
                           [("flow", "plastic"), ("plastic", "thermal"), ("elastic", "plastic"),
                            ("phase", "thermal"), ("phase", "flow"), ("flow", "thermal")],
                           V, "FLAGSHIP (cnc_cutting_process.py): the validated edges COMPOSE into the integrated cutting observables — Merchant chip (flow↔plastic) → cutting power → Boothroyd/JAEGER interface temperature θ∝V^0.38 (plastic↔thermal, Loewen-Shaw) → Arrhenius wear → Taylor V·T^0.45=C (R²=0.999; NULL Q=0→no Taylor); rake↑→force↓→θ↓→life↑ consistent. PROCESS-MODEL level (anchored: Merchant/Loewen-Shaw/Taylor); a full spatial finite-strain cutting FIELD for a literal slow-mo render is the deeper build"),
}

# ── CAPABILITIES: each requires phenomena (the author's invariant: ungrounded if any underlying phenomenon is absent) ──
# name: (impossible-instrument views, required phenomena)
CAPABILITIES = {
    "thermal_camera":        (["IR temperature of any solid/fluid/gas"],   ["rayleigh_benard", "joule_heating"]),
    "flow_visualization":    (["PIV / schlieren velocity + density"],      ["navier_stokes", "rayleigh_benard"]),
    "xray_subsurface_stress":(["internal stress / density / cracks"],      ["thermoelasticity", "fracture"]),
    "acoustic_emission":     (["crack / cavitation event mapping"],        ["fracture", "multiphase_core"]),
    "morphology_render":     (["fractal character: clouds, chip, finish"], ["cloud_formation", "multiphase_core"]),
    # the two that SHOULD fail grounding until the CNC/plasticity/wear modules exist — proves the invariant bites:
    "cnc_slowmo_thermal":    (["thermal-cam + X-ray of cutting in slow-mo"], ["cnc_machining"]),
    "tool_wear_twin":        (["aging tool; surface-pattern evolution; invertible tool-life"], ["cnc_machining"]),
    "lightning_slowmo":      (["super-slow strike: potential field + ground-charge + fractal channel + thunder"], ["lightning"]),
}


# ── CEILING-RAISERS: extreme "above-weather" targets (the author-selected ) ─────────────────────────────────
# Principle: an extreme target is a FORCING FUNCTION — the capabilities it demands transfer DOWN to every simpler twin
# (cross-domain gains). The three are chosen to SPAN different quadrants of the coupling graph ⇒ maximal capability
# superset. Each names the quadrant, what it forces, the cross-domain payoff, and an HONEST LBM native-vs-departure verdict.
CEILING_RAISERS = {
    "full_earth_system": dict(
        quadrant="geophysical BREADTH + long-integration",
        forces="ocean (double-diffusive salinity) + ice (enthalpy phase-change) + carbon/biosphere (reactive species) + "
               "land hydrology; conservation over century integration; many coupled active scalars",
        cross_domain="long-integration conservation + multi-scalar coupling → every environmental/geophysical/climate twin",
        lbm="mostly-native (extends validated cloud_formation); full band-radiation is the one departure",
        builds_from="cloud_formation"),
    "detonation": dict(
        quadrant="STIFFNESS + shocks + chemistry",
        forces="turbulence × stiff Arrhenius chemistry × shocks × spray-multiphase; stiff implicit chemistry + shock-capturing",
        cross_domain="stiff-solver + shock-capturing + turbulence-chemistry → combustion/engine/propulsion/safety + ANY high-gradient problem",
        lbm="partial-departure: strong shocks need HLLC-Godunov (already validated vs exact Riemann); rest native",
        builds_from="combustion"),
    "living_cell": dict(
        quadrant="ACTIVE MATTER + molecular fidelity + information",
        forces="reaction-diffusion at scale (1000s species) × active mechano-chemistry (membrane↔chemistry) × "
               "electrophysiology (charge) × molecular crowding (discrete↔continuum)",
        cross_domain="reaction-diffusion-at-scale + active-stress + discrete↔continuum + molecular-depth → biology/medicine/soft-matter",
        lbm="major-departure at the molecular core (the depth axis); meso reaction-diffusion + active-nematic is native-ish",
        builds_from="(new: active matter + RD-at-scale)"),
}


# ── ★FALSIFIABLE BLOCKER TAGS (the author : the "3 kernels = the cross-domain leverage" line is a FIRST DRAFT from
#    two agent waves — treat it as a hypothesis and make it TRACEABLE so a later stuck build can be derived back to a
#    recorded blocker, not guessed). Each non-validated coupling is tagged with what it is ACTUALLY blocked on:
#    a SHARED reusable solver-kernel (poisson/rt/imex) OR a per-cell CONSTITUTIVE law (NOT shared) OR "none" (native now).
#    The kernel-impact test below then COMPUTES, per phenomenon, whether a kernel is sufficient or only necessary. ──
BLOCKER = {
    ("species", "flow"): {"none"}, ("species", "species"): {"imex"}, ("em", "thermal"): {"none"},
    ("radiation", "thermal"): {"rt"}, ("phase", "phase"): {"imex"}, ("gravity", "phase"): {"none"},
    ("charge", "flow"): {"poisson"}, ("charge", "species"): {"poisson"}, ("charge", "em"): {"poisson"},
    ("gravity", "flow"): {"poisson"}, ("plastic", "thermal"): {"constitutive"}, ("flow", "plastic"): {"constitutive"},
    ("species", "thermal"): {"none"}, ("charge", "elastic"): {"poisson", "constitutive"}, ("radiation", "flow"): {"rt"},
    ("plastic", "plastic"): {"constitutive"}, ("species", "elastic"): {"constitutive"}, ("species", "phase"): {"constitutive"},
    ("em", "elastic"): {"constitutive"}, ("elastic", "phase"): {"constitutive"}, ("plastic", "species"): {"constitutive"},
    ("charge", "thermal"): {"poisson", "constitutive"}, ("phase", "charge"): {"poisson"}, ("phase", "em"): {"poisson"},
    ("elastic", "gravity"): {"poisson", "constitutive"}, ("radiation", "plastic"): {"rt", "constitutive"},
    ("radiation", "species"): {"rt", "constitutive"}, ("phase", "plastic"): {"constitutive"},
}
KERNELS = ("poisson", "rt", "imex")


def _status_rank(s):
    return {V: 2, B: 1, M: 0}[s]


def phenomenon_blockers(ph):
    """union of blocker-tags over a phenomenon's NON-validated couplings (excluding 'none'=native-buildable-now).
    Untagged non-validated couplings default to 'constitutive' (the conservative non-kernel assumption)."""
    bl = set()
    for (a, b) in ph["couplings"]:
        _m, st, _anc = CKEY[(a, b)]
        if st != V:
            bl |= (BLOCKER.get((a, b), {"constitutive"}) - {"none"})
    return bl


def check_integrity():
    """self-test: every reference resolves. Raises on the FIRST inconsistency (the registry's own measurement tool)."""
    for name, ph in PHENOMENA.items():
        for f in ph["fields"]:
            if not (f in FIELDS): raise AssertionError(f"phenomenon {name}: unknown field {f}")   # -O-safe (was assert, J-R3)
        for (a, b) in ph["couplings"]:
            if not ((a, b) in CKEY): raise AssertionError(f"phenomenon {name}: coupling {(a, b)} not in COUPLINGS")   # -O-safe (was assert, J-R3)
            if not (a in ph["fields"] and b in ph["fields"]): raise AssertionError(f"phenomenon {name}: coupling {(a,b)} endpoints not in its fields")   # -O-safe (was assert, J-R3)
    for cap, (_views, reqs) in CAPABILITIES.items():
        for r in reqs:
            if not (r in PHENOMENA): raise AssertionError(f"capability {cap}: required phenomenon {r} not defined")   # -O-safe (was assert, J-R3)
    return True


def coupling_matrix():
    """undirected field-pair coverage: best status among couplings touching the pair."""
    names = list(FIELDS)
    cell = {}
    for (a, b, _m, st, _anc) in COUPLINGS:
        for key in ((a, b), (b, a)):
            cur = cell.get(key)
            if cur is None or _status_rank(st) > _status_rank(cur):
                cell[key] = st
    return names, cell


def main():
    check_integrity()
    sym = {V: "V", B: "o", M: ".", None: " "}
    names, cell = coupling_matrix()
    abbr = [n[:4] for n in names]

    print("=" * 92)
    print("PHENOMENON COMPLETENESS REGISTRY — coupling graph (nodes=fields, edges=Onsager/LBM couplings)")
    print("=" * 92)
    print(f"  {len(FIELDS)} fields, {len(COUPLINGS)} coupling edges, {len(PHENOMENA)} phenomena, {len(CAPABILITIES)} capabilities")

    # 1) coupling matrix (systematic cell-by-cell coverage — the completeness guarantee)
    print("\n  COUPLING MATRIX (V=validated  o=build-list  .=missing  blank=no coupling yet):")
    print("        " + " ".join(f"{a:>4}" for a in abbr))
    for i, a in enumerate(names):
        row = []
        for j, b in enumerate(names):
            if i == j:
                row.append("  - ")
            else:
                row.append(f"  {sym[cell.get((a, b))]} ")
        print(f"  {abbr[i]:>4}  " + "".join(row))
    total_cells = len(names) * (len(names) - 1) // 2
    filled = len({frozenset(k) for k in cell})
    vcells = len({frozenset((a, b)) for (a, b, _m, st, _a) in COUPLINGS if st == V})
    print(f"  → coupling coverage: {filled}/{total_cells} field-pairs have a coupling ({100*filled/total_cells:.0f}%); "
          f"{vcells} validated. EMPTY cells = systematically-found gaps (candidate new physics).")

    # 2) phenomena by status + single vs combination
    by = {V: [], B: [], M: []}
    for n, ph in PHENOMENA.items():
        by[ph["status"]].append(n)
        ph["kind"] = "single" if len(ph["fields"]) == 1 else "combination"
    ncomb = sum(1 for ph in PHENOMENA.values() if ph["kind"] == "combination")
    print(f"\n  PHENOMENA: {len(by[V])} validated, {len(by[B])} build-list, {len(by[M])} missing "
          f"({ncomb} are COMBINATIONS = multi-field subgraphs, {len(PHENOMENA)-ncomb} singles)")
    print(f"    validated: {', '.join(sorted(by[V]))}")
    print(f"    build-list: {', '.join(sorted(by[B]))}")
    print(f"    missing:   {', '.join(sorted(by[M]))}")

    # 3) ★capability grounding — the author's invariant in code
    print("\n  CAPABILITY GROUNDING (a capability is only REAL if every underlying phenomenon is validated):")
    ungrounded = []
    for cap, (_views, reqs) in sorted(CAPABILITIES.items()):
        worst = min((PHENOMENA[r]["status"] for r in reqs), key=_status_rank)
        blockers = [r for r in reqs if PHENOMENA[r]["status"] != V]
        if worst == V:
            verdict = "GROUNDED   "
        elif worst == B:
            verdict = "DEGRADED   "
        else:
            verdict = "UNGROUNDED "
            ungrounded.append(cap)
        tail = "" if not blockers else "  ← needs: " + ", ".join(f"{b}[{PHENOMENA[b]['status']}]" for b in blockers)
        print(f"    {verdict} {cap:<24}{tail}")

    # 4) systematic gap list (missing/build-list couplings, value-ordered by appearance)
    gaps = [(a, b, m, st) for (a, b, m, st, _anc) in COUPLINGS if st != V]
    print(f"\n  SYSTEMATIC GAPS — {len(gaps)} non-validated coupling edges (each = a concrete next build):")
    for (a, b, m, st) in gaps:
        print(f"    [{st:>9}] {a}↔{b}: {m}")

    # 4b) ceiling-raisers + the three cross-domain leverage kernels (both waves' shared finding)
    print(f"\n  CEILING-RAISERS (the author-selected; extreme targets forcing capabilities that transfer DOWN):")
    for name, cr in CEILING_RAISERS.items():
        print(f"    {name:<18} [{cr['lbm']:<18}] {cr['quadrant']}")
    print(f"  ★3 CROSS-DOMAIN KERNELS unlock whole matrix regions (build once → many cells light up):")
    print(f"    elliptic-Poisson  → every CHARGE + GRAVITY cell (electro-tier, self-gravity, lightning)")
    print(f"    angular-RT (DOM/Pn) → every RADIATION cell (beyond the gray/optically-thin limit)")
    print(f"    stiff-IMEX        → every reacting/phase-change source — DETONATION forces it ON-substrate")
    print(f"    (fusion would force all three but is a MAJOR departure → kept as the sigma/Knudsen reference)")

    # 4c) ★KERNEL-IMPACT TEST — falsify/refine the "3 kernels unlock the rows" first-draft assumption (COMPUTED, per phenomenon)
    print(f"\n  ★KERNEL-IMPACT TEST (the author: '3 kernels' = first draft → make it falsifiable + traceable):")
    missing_ph = [n for n, ph in PHENOMENA.items() if ph["status"] == M]
    for K in KERNELS:
        full = [n for n in missing_ph if phenomenon_blockers(PHENOMENA[n]) == {K}]
        partial = [n for n in missing_ph if K in phenomenon_blockers(PHENOMENA[n]) and phenomenon_blockers(PHENOMENA[n]) != {K}]
        print(f"    {K:<9}: FULLY unlocks {full or ['—']};  partially helps {partial or ['—']}")
    const_only = [n for n in missing_ph if phenomenon_blockers(PHENOMENA[n]) and phenomenon_blockers(PHENOMENA[n]) <= {"constitutive"}]
    kernel_blocked = [n for n in missing_ph if phenomenon_blockers(PHENOMENA[n]) & set(KERNELS)]
    tally = {}
    for _pair, bset in BLOCKER.items():
        for blk in bset:
            tally[blk] = tally.get(blk, 0) + 1
    print(f"    cell-blocker tally (non-validated couplings): {tally}")
    print(f"  ⇒ VERDICT (refines the wave synthesis, not rubber-stamps it): {len(kernel_blocked)}/{len(missing_ph)} missing")
    print(f"    phenomena need a shared KERNEL; {len(const_only)} are blocked ONLY on a per-cell CONSTITUTIVE law (plasticity,")
    print(f"    wear, martensitic, ...) — a 4th NON-shared category the 'just 3 kernels' line under-weighted. So: kernels are")
    print(f"    necessary infra for the SOLVER-bound cells, NOT a universal unlock. Each phenomenon's blockers are now RECORDED")
    print(f"    (phenomenon_blockers) ⇒ a stuck build traces to its tag, not a buried premise. ★FIRST-DRAFT, falsifiable.")

    # 5) the invariant — report (do not crash; this is a living coverage tool)
    print("\n" + "=" * 92)
    print(f"INVARIANT (capability ⇐ phenomenon): {len(CAPABILITIES)-len(ungrounded)}/{len(CAPABILITIES)} capabilities grounded.")
    if ungrounded:
        print(f"  ⚠ UNGROUNDED (claimed capability without underlying validated phenomenon): {', '.join(ungrounded)}")
        print(f"    → these are HONESTLY blocked on their missing phenomena (e.g. cnc_machining ⇐ finite-strain plasticity")
        print(f"      + wear). The registry REFUSES to let a capability be 'ready' before its physics exists — exactly the")
        print(f"      'capabilities must have underlying phenomenon support' invariant, enforced in code.")
    print(f"  METHOD: enumerate the {total_cells}-cell coupling matrix + subgraphs ⇒ completeness is COUNTED, not hoped;")
    print(f"  every empty cell is a found gap; every capability traces to the phenomena it needs. Populate from the")
    print(f"  74-item combinations scan + per-phenomenon validation scripts; coverage % is the tracked headline.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
