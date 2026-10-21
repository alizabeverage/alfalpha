# ALFALPHA_TP — two-process-constrained abundance fit mode (append-only log)

Discipline: append-only, same as the phase logs. Artifacts outrank descriptions; no tuning of
inputs to improve agreement; the standard full-element mode is never altered in behaviour and its
byte-for-byte reference is checked. All new code lives under `alfa/` and `tests/`; nothing in the
standard model path changes semantics.

═══════════════════════════════════════════════════════════════════════════════════════════════
## STEP 0 — WHAT WAS READ, AND THE CORRECTION THAT THIS MODE REPLACES (read before the rest)
═══════════════════════════════════════════════════════════════════════════════════════════════

Read in full before writing: `alfa/grids.py` (forward model), `alfa/setup_params.py`,
`alfa_config.py` (likelihood/driver), `alfa/post_process.py` + `alfa/utils.py` (the post-fit
correction), and the phase-5 machinery (`~/Research/alf_repro/two_process.py`, `compute_qs.py`,
`Downloads/PHASE5.md`).

### 0.1 The current post-fit library correction (quoted, verbatim behaviour)
`alfa/utils.py::correct_abundance(zh, xh, element)`:
- non-alpha elements:            `[X/H] = xh + zh`
- alpha elements, with a MILES-library MW alpha-pattern interpolated at `zh` (treated as [Fe/H]):
      `mgh -> xh + zh + del_mgfe(zh)`;  `ah -> xh + zh + del_alfe(zh)` (del_alfe from lib_ofe);
      `cah,tih,sih -> xh + zh + del_cafe(zh)`
  library tables (utils.py:66):
      lib_feh  = [-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2, 0.0, 0.2]
      lib_mgfe = [ 0.4, 0.4, 0.4, 0.4,0.34,0.22,0.14,0.11,0.05,0.04]
      lib_cafe = [0.32, 0.3,0.28,0.26,0.26,0.17,0.12,0.06, 0.0, 0.0]
      lib_ofe  = [ 0.6, 0.5, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.0, 0.0]

WHERE APPLIED: post-fit only, in `post_process.py` in the `.sum` block (Fe/Mg refs at
post_process.py:133-136; per-element loop ~:143-171; Thomas [Z/H] ~:165-176). It is NEVER called in
the forward model — `grids.get_model` applies the raw response offsets `interp([logage,zH,value])`
with no library correction. **The new mode moves this correction INSIDE the forward model and
asserts the post-fit path is bypassed.**

═══════════════════════════════════════════════════════════════════════════════════════════════
## STEP 1 — THE FORWARD-MODEL ALGEBRA (locked with the PI, 2026-07-28)
═══════════════════════════════════════════════════════════════════════════════════════════════
Three decisions were confirmed interactively (the first answers conflicted and were reconciled to a
model where every parameter is active and the Fe response does not collapse):

- MW baseline relation = the existing `lib_mgfe(lib_feh)` table (swappable). It IS the MILES/MW
  pattern the SSPs are assumed to carry, so the baseline is self-consistent with standard mode.
- `feh`, `mgh` are OFFSETS on the library (standard-alf convention), NOT absolute abundances.
- baseline metallicity `feh_lib = zh` (the library grid metallicity).

Resulting algebra (implemented in `alfa/two_process_mode.py::compute_deltas`):

    feh_true  = zh + feh
    mgh_true  = zh + mgh
    mgfe_true = mgh - feh
    feh_lib   = zh                       mgfe_lib = library_mgfe(zh)   mgh_lib = zh + library_mgfe(zh)
    target[X]   = two_process_abundances(feh_true, mgfe_true, ref='H', mgfe_pl=0.65)
    baseline[X] = two_process_abundances(feh_lib,  mgfe_lib,  ref='H', mgfe_pl=0.65)
    delta[X]    = target[X] - baseline[X]        # applied via the shipped responses, uniformly

By construction `delta_Fe == feh` and `delta_Mg == mgh - library_mgfe(zh)`; every other element is
slaved. The library's non-solar MW pattern lives entirely in `baseline`.

Worked check at (zh=-0.2, feh=-0.1, mgh=+0.15): feh_true=-0.30, mgfe_true=+0.25, mgfe_lib=0.11,
delta_Fe=-0.10, delta_Mg=+0.10-... = +0.04. Confirmed against `compute_deltas` in the unit tests.

═══════════════════════════════════════════════════════════════════════════════════════════════
## STEP 2 — ELEMENT MAP AND POLICIES (declared, printed at build, never silent)
═══════════════════════════════════════════════════════════════════════════════════════════════
Printed by `two_process_mode.describe_mapping()` when a TP-mode `Grids` is built:

  APPLIED (17): C->ch N->nh O->ah Na->nah Mg->mgh Si->sih K->kh Ca->cah Ti->tih V->vh Cr->crh
                Mn->mnh Fe->feh Co->coh Ni->nih Sr->srh Ba->bah
  O treatment : [O/H] applied through the 'as' response (ah); O/Ne/S share that column, so S and Ne
                ride along (documented; matches PHASE5 decision O-2). This is the validated R_as
                treatment carried over.
  HELD SOLAR (2): Cu, Eu  (response exists; two_process does not predict them -> delta 0)
  DROPPED (4)   : Al, S, Y, Ce  (predicted by two_process; NO response column in the alf files)
  N validity    : two_process N is trusted only on [Mg/H] in (-0.5, +0.15); outside it (or NaN) the
                  [Mg/H] fed to N is CLAMPED to the boundary holding the alpha enhancement fixed
                  (moving [Fe/H], NOT mgfe -- raising mgfe would push above the 0.65 plateau and make
                  the whole point unphysical), and the fit is flagged. Fraction clamped logged as
                  `frac_N_clamped`.
  C exponent cap: C's response node is 0.15 dex; |delta_C| capped at `c_max_exponent*0.15`
                  (default 1.0) and flagged. Fraction capped logged as `frac_C_capped`.
  ALUMINIUM: currently DROPPED (no Al response column ships in alfalpha's atlas_ssp files). This is
             the one element PHASE5 identified as the dominant composer residual (Al I 3944/3961).
             When an Al response column is added to the grid, add `Al->alh` to `ELEMENT_TO_KEY` and
             it is applied automatically; nothing else changes.

═══════════════════════════════════════════════════════════════════════════════════════════════
## STEP 3 — IMPLEMENTATION (files touched)
═══════════════════════════════════════════════════════════════════════════════════════════════
NEW:
- `alfa/two_process/`  — the phase-5 abundance model, VENDORED unchanged from ~/Research/alf_repro/
  (two_process.py, compute_qs.py, data/W22_*.csv). One packaging edit only: the intra-package import
  `from compute_qs import compute_qs` -> `from .compute_qs import compute_qs`. Byte-identical otherwise.
- `alfa/two_process_mode.py` — the interface layer: swappable MW table, element map, policies,
  `compute_deltas(zh,feh,mgh,config)`, `describe_mapping()`, `TwoProcessConfig` (with to_dict/
  from_dict so it round-trips through the fit's .json).
- `tests/regression_capture.py` — freezes/checks the standard-mode forward model bit-for-bit.
- `tests/test_two_process_mode.py` — interface unit tests.
- `tests/tp_validation.py` — the §5 validation ladder (mock recovery / correction-off / crosscheck).

EDITED (all additive/guarded; standard path unchanged when `two_process_mode=False`):
- `alfa/grids.py`: `Grids.__init__` gains `two_process_mode`, `two_process_config` (prints the map at
  build); `get_model` gains a guarded pre-step that derives element deltas from (zH,feh,mgh) and
  injects them, so the EXISTING response loop applies them unchanged.
- `alfa/fitting_info.py`: default `two_process_mode=False`, `two_process_config=None` (persisted).
- `alfa/post_process.py`: TP-mode Grids construction; the standard `correct_abundance` block is
  guarded off in TP mode; new `_two_process_summary` reports feh_true/mgfe_true and the two-process
  predicted [X/H],[X/Fe] for every element, logs `frac_N_clamped`/`frac_C_capped`, and ASSERTS no
  standard [X/..] keys were written (the correction path is bypassed).
- `alfa_config.py`: `two_process_mode` toggle -> reduced parameter set (velz,sigma,logage,zH,feh,
  mgh,jitter,emission) and passes the config into `Grids`.
- `alfa/polynorm.py`: ONE unrelated bug fix required to run anything on this env's numpy>=2:
  `math.ceil(np.diff(region)/100)` -> `...np.diff(region).item()...` (ceil of a 1-elem array raises
  on numpy 2). Pre-existing latent bug, not TP-specific; recorded here because it was touched.

═══════════════════════════════════════════════════════════════════════════════════════════════
## STEP 4 — BACKWARD COMPATIBILITY (item 4)
═══════════════════════════════════════════════════════════════════════════════════════════════
`tests/regression_capture.py` freezes `get_model` output (5731 pts, native grid, inst_res=0) for a
fixed 21-key parameter vector exercising SSP interpolation, individual element responses, teff and
emission lines. Captured BEFORE any TP change.

  AFTER all edits:  max|dwave| = 0.000e+00   max|dspec| = 0.000e+00   -> PASS (bit-for-bit)

The TP branch in `get_model` is entered only via `getattr(self,'two_process_mode',False)`, default
False, so the standard mode is provably untouched.

═══════════════════════════════════════════════════════════════════════════════════════════════
## STEP 5 — VALIDATION LADDER
═══════════════════════════════════════════════════════════════════════════════════════════════
Runner: `tests/tp_validation.py`. The `--fast` runs below are laptop-scale (emcee, 3800-5600 A
window, deg=18); the knobs are arguments so the full-resolution versions are one call away.

### 5.0 Interface unit tests (`tests/test_two_process_mode.py`) — ALL PASS
  Fe delta == feh exactly; Mg delta == mgh - lib_mgfe(zh); target==baseline -> max|delta|=0.0;
  slaved elements == two_process(target)-two_process(baseline); C cap bites and clamps at the node;
  N clamps at the CN-track boundary and stays finite; a normal point (-0.3) clamps/drops nothing;
  dropped (Al,Y,Ce,S) and heldsol (Cu,Eu) never applied.

### 5a MOCK RECOVERY (composer-composed, composer-fit closure test) — PASS
Truth (zH,feh,mgh)=(-0.20,-0.10,+0.12), S/N=100, 24 walkers x 600 steps:
  param   truth   median   pull        param  truth  median  pull
  zH     -0.200  -0.208   -0.45        feh   -0.100  -0.091 +0.91
  mgh    +0.120  +0.115   -0.47        logage +0.700 +0.707 +0.53
  (sigma +2.1 sigma, a fast-run continuum/broadening artifact, not an abundance target)
  -> the three abundance parameters recover within 1 sigma. corner: tests/tp_validation_out/.

### 5b CORRECTION-OFF (what the library baseline is worth) — PASS, as predicted
Same mock (composed WITH baseline) fit by a `subtract_baseline=False` model. Registered prediction:
slaved [X/H] biases by exactly the negative of the library baseline pattern. Result:
  elem  obs bias  pred(-baseline)      elem  obs bias  pred(-baseline)
  O      +0.101    +0.104              Ti    +0.108    +0.110
  Na     +0.189    +0.189              Cr    +0.216    +0.209
  Si     +0.120    +0.125              Ni    +0.179    +0.182
  Ca     +0.134    +0.137              Ba    +0.272    +0.223
  obs bias == pred bias to 0.048 dex (Ba worst; the rest <=0.01). The correction is worth
  ~0.10-0.27 dex per slaved element at this point, and the OFF model biases exactly as predicted.

### 5c REAL-DATA CROSSCHECK — machinery built; converged result OWED
`realdata_crosscheck()` fits `indata/test` in full-element mode and TP mode and compares the
TP-predicted [X/H] against the full-mode posterior [X/H] (via correct_abundance) element by element.
A SCIENCE result needs a converged fit (cluster-scale); recorded here only that the code path runs
end-to-end. A converged run is the owed deliverable — numbers are NOT recorded until then, per the
no-junk-results discipline.
  SMOKE (30 walkers x 300 steps, indata/test): both fits run, the crosscheck table is produced,
  full-element and TP posteriors are read and compared element by element. The full-element fit is
  visibly unconverged ([Mg/H]~0.76, [C/H]~0.50) and the TP fit has not moved off its start, so the
  agree/NO column is meaningless at this step count — as labelled. Code path VERIFIED; converged
  crosscheck (dynesty or long emcee) OWED. Run: `python tests/tp_validation.py 5c` (no --fast).

═══════════════════════════════════════════════════════════════════════════════════════════════
## STEP 6 — LIGHT-ELEMENT POLICY FOR C AND N (decision 2026-07-28, supersedes STEP 2 for C/N)
═══════════════════════════════════════════════════════════════════════════════════════════════
Raised by the PI: C and N should not simply be slaved to the raw two-process prediction. They are
the least-trustworthy predictions in the model — C and N come from the Roberts+24 CN SUBGIANT
TRACKS (natal C/N recovery rescaled to APOGEE C+N), are defined only on a narrow [Mg/H] window, and
C's response is the most extrapolated element — AND they drive the strongest molecular bands (CN
3883, CH G-band, CN red), so a wrong slaved value propagates into zH/age/Mg through those bands.
This is the same concern PHASE5.md A.1 acted on when it forced C, N (and Mn) solar.

MACHINERY (implemented, all three selectable via `TwoProcessConfig`):
  light_elements      = ('C','N')                     # the specially-handled set
  light_element_mode  in {'solar','follow_mg','two_process'}
    solar       : delta 0 — held at the library baseline, not applied (PHASE5-consistent null).
    follow_mg   : delta_C = delta_N = delta_Mg — the two band-drivers track the robustly-measured
                  Mg (alpha) response instead of the CN tracks.
    two_process : the actual prediction, with the C exponent cap and N validity clamp (STEP 2).
  The cap/clamp code now runs ONLY in 'two_process' mode (moved into `_nitrogen_delta` and the
  C branch); 'solar'/'follow_mg' skip the extra two_process calls entirely.

DECISIONS TAKEN:
  - DEFAULT light_element_mode = **follow_mg**. For the massive/quiescent galaxies this mode targets,
    C and N are genuinely enhanced (not solar); tying them to Mg gives a physically-plausible non-
    solar value without trusting the CN tracks. Solar would under-fill the CN/CH bands and bias the
    fit; raw two_process puts the least-reliable predictions on the strongest bands. Known residual:
    N in massive galaxies often runs ABOVE Mg (steeper [N/Fe]-sigma), so follow_mg mildly under-
    predicts N — accepted as far smaller than the solar or CN-track error.
  - Mn LEFT SLAVED (not added to the light set), unlike PHASE5. Mn is mirror-only in alfalpha (same
    caveat PHASE5 flagged) but is a much weaker feature than the CN/CH bands, so the propagation risk
    that motivates special-handling C/N does not apply. Recorded as a deliberate departure from
    PHASE5's C/N/Mn triple.
  verify: `describe_mapping` now prints SLAVED(2p)=15 + LIGHT={C,N}->mode + HELDSOL/DROPPED; unit
  test `test_light_element_modes` checks solar(unapplied)/follow_mg(==delta_Mg)/two_process(applied).

REVALIDATION under the new default (follow_mg):
  - regression: standard mode still bit-for-bit (max|dspec|=0).
  - 5a mock recovery (S/N=100, 24x600): zH pull +0.06, feh +0.10, mgh -0.68 — all <1sigma. PASS.
  - 5b correction-off: unaffected (its reported elements O,Na,Si,Ca,Ti,Cr,Ni,Ba are not light);
    obs bias == -library baseline to 0.048 dex stands.
  - interface unit tests: all pass (now incl. the three light modes).
