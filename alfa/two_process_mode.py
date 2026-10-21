"""Two-process-constrained abundance mode for alfalpha (spec: ALFALPHA_TP.md).

This module is the interface between the two-process chemical-evolution model
(``alfa.two_process``) and alfalpha's shipped response functions. Given the
three fitted parameters ``(zh, feh, mgh)`` it returns the per-element response
deltas that ``grids.Grids.get_model`` applies, plus the policy flags that must
be logged per fit.

THE FORWARD-MODEL ALGEBRA (locked with the PI; see ALFALPHA_TP.md §1)
--------------------------------------------------------------------
``feh`` and ``mgh`` are OFFSETS on the library, exactly as in standard alf::

    feh_true  = zh + feh
    mgh_true  = zh + mgh
    mgfe_true = mgh_true - feh_true          (= mgh - feh)

The MILES library at grid metallicity ``zh`` is itself a Milky-Way population
sitting at a two-process point on the MW disk track::

    feh_lib   = zh
    mgfe_lib  = library_mgfe(zh)             # swappable MW [Mg/Fe]-[Fe/H] table
    mgh_lib   = zh + library_mgfe(zh)

The response for each element carries the spectrum FROM the library baseline TO
the target, so the applied delta is ``target - baseline`` UNIFORMLY, including
Fe and Mg::

    target[X]   = two_process_abundances(feh_true, mgfe_true, ref='H', mgfe_pl)
    baseline[X] = two_process_abundances(feh_lib,  mgfe_lib,  ref='H', mgfe_pl)
    delta[X]    = target[X] - baseline[X]

By construction ``delta_Fe == feh`` and ``delta_Mg == mgh - library_mgfe(zh)``.
The library's non-solar MW pattern therefore lives entirely inside ``baseline``;
NOTHING is corrected in post-processing (that is the whole point of the mode,
and ``post_process`` asserts the old ``correct_abundance`` path is bypassed).

ELEMENT POLICIES (declared, never silent; logged per fit)
---------------------------------------------------------
* APPLIED  : has a shipped response column; delta applied via that response.
* HELDSOL  : has a response but two_process does not predict it -> delta 0.
* DROPPED  : two_process predicts it but no response column exists -> not applied.
* O -> the 'as' response: alf ships no O column; O/Ne/S share the ``a`` (as)
  response, so [O/H] is applied through ``ah``. S rides along (a known, small,
  documented excess -- see ALFALPHA_TP.md); Ne is spectroscopically nil.
* Nitrogen validity: two_process N is only defined on the CN-track window and
  diverges where its core-collapse q goes negative. Outside the window the
  target [Mg/H] fed to N is CLAMPED to the boundary and the fit is flagged.
* Carbon exponent cap: C's response node is 0.15 dex; |delta_C| is capped at
  ``c_max_exponent * 0.15`` and the fit is flagged when the cap bites.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from alfa.two_process import two_process_abundances, DEFAULT_MGFE_PL


# ---------------------------------------------------------------------------
# Swappable MW [Mg/Fe]-[Fe/H] relation for the library baseline.
# DEFAULT: the MILES-library MW pattern already embedded in alfa.utils
# .correct_abundance (lib_feh / lib_mgfe). Using it here keeps the two-process
# baseline self-consistent with what standard alf assumes the SSPs carry.
# ---------------------------------------------------------------------------
MW_LIBRARY_FEH = np.array([-1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
MW_LIBRARY_MGFE = np.array([0.4, 0.4, 0.4, 0.4, 0.34, 0.22, 0.14, 0.11, 0.05, 0.04])


def library_mgfe(feh, feh_grid=MW_LIBRARY_FEH, mgfe_grid=MW_LIBRARY_MGFE):
    """Library [Mg/Fe] at metallicity ``feh`` from the (swappable) MW table.

    Linear interpolation with flat extrapolation beyond the tabulated range
    (np.interp clamps), matching the bounded behaviour used elsewhere.
    """
    return np.interp(feh, feh_grid, mgfe_grid)


# ---------------------------------------------------------------------------
# Mapping two_process element symbols -> alfalpha response parameter keys.
# ---------------------------------------------------------------------------
# APPLIED: element symbol -> alfalpha key (whose response is `<key[:-1]>_interp`)
ELEMENT_TO_KEY = {
    "C": "ch", "N": "nh", "O": "ah", "Na": "nah", "Mg": "mgh", "Si": "sih",
    "K": "kh", "Ca": "cah", "Ti": "tih", "V": "vh", "Cr": "crh", "Mn": "mnh",
    "Fe": "feh", "Co": "coh", "Ni": "nih", "Sr": "srh", "Ba": "bah",
}
# response exists in the grid but two_process does not predict it -> held solar
HELDSOL = ("Cu", "Eu")
# two_process predicts it but there is NO response column in the alf files.
# S is additionally covered indirectly: O/Ne/S share the 'as' (ah) response.
DROPPED = ("Al", "S", "Y", "Ce")

# response node ("conv_e") per alfalpha key -- from grids.Rfn.set_up_interpolators.
# 0.30 everywhere except C (0.15). (Na has a +0.3/0.6/0.9 ladder; its node is
# still 0.30 and the interpolant handles the ladder.)
CONV_E = {k: 0.30 for k in ELEMENT_TO_KEY.values()}
CONV_E["ch"] = 0.15


@dataclass
class TwoProcessConfig:
    """Knobs for the two-process fit mode. All are logged at build time."""
    mgfe_pl: float = DEFAULT_MGFE_PL              # core-collapse [Mg/Fe] plateau (0.65)
    # "light elements" get special handling: C and N are the least-trustworthy
    # two-process predictions (CN subgiant tracks, narrow validity, stretched C
    # response) AND the strongest molecular-band drivers, so a wrong slaved value
    # biases the whole fit (cf. PHASE5.md A.1, which forced C, N, Mn solar).
    light_elements: Tuple[str, ...] = ("C", "N")
    light_element_mode: str = "follow_mg"         # 'solar' | 'follow_mg' | 'two_process'
    #   solar       : delta 0 (held at library baseline; not applied)
    #   follow_mg   : delta == delta_Mg (track the robustly-measured Mg response)
    #   two_process : apply the actual two-process prediction, with the cap/clamp below
    c_max_exponent: float = 1.0                   # cap |delta_C| at this * 0.15 (two_process mode)
    n_valid_mgh: Tuple[float, float] = (-0.5, 0.15)  # [Mg/H] window where N is trusted (two_process)
    subtract_baseline: bool = True                # False => library treated as solar
                                                  #   (the "baseline OFF" validation test)
    mw_feh_grid: np.ndarray = field(default_factory=lambda: MW_LIBRARY_FEH.copy())
    mw_mgfe_grid: np.ndarray = field(default_factory=lambda: MW_LIBRARY_MGFE.copy())

    def lib_mgfe(self, feh):
        return library_mgfe(feh, self.mw_feh_grid, self.mw_mgfe_grid)

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot (persisted in the fit's .json settings)."""
        return {
            "mgfe_pl": float(self.mgfe_pl),
            "light_elements": list(self.light_elements),
            "light_element_mode": str(self.light_element_mode),
            "c_max_exponent": float(self.c_max_exponent),
            "n_valid_mgh": [float(self.n_valid_mgh[0]), float(self.n_valid_mgh[1])],
            "subtract_baseline": bool(self.subtract_baseline),
            "mw_feh_grid": np.asarray(self.mw_feh_grid).tolist(),
            "mw_mgfe_grid": np.asarray(self.mw_mgfe_grid).tolist(),
        }

    @classmethod
    def from_dict(cls, d) -> "TwoProcessConfig":
        """Rebuild from to_dict() output (or None -> defaults)."""
        if d is None:
            return cls()
        nvm = d.get("n_valid_mgh", (-0.5, 0.15))
        return cls(
            mgfe_pl=float(d.get("mgfe_pl", DEFAULT_MGFE_PL)),
            light_elements=tuple(d.get("light_elements", ("C", "N"))),
            light_element_mode=str(d.get("light_element_mode", "follow_mg")),
            c_max_exponent=float(d.get("c_max_exponent", 1.0)),
            n_valid_mgh=(float(nvm[0]), float(nvm[1])),
            subtract_baseline=bool(d.get("subtract_baseline", True)),
            mw_feh_grid=np.asarray(d.get("mw_feh_grid", MW_LIBRARY_FEH), dtype=float),
            mw_mgfe_grid=np.asarray(d.get("mw_mgfe_grid", MW_LIBRARY_MGFE), dtype=float),
        )


def describe_mapping(config: TwoProcessConfig | None = None) -> str:
    """Human-readable element-policy table, printed once at grid build."""
    config = config or TwoProcessConfig()
    light = set(config.light_elements)
    applied = ", ".join(f"{e}->{k}" for e, k in ELEMENT_TO_KEY.items() if e not in light)
    lines = [
        "two-process fit mode — element policies",
        f"  mgfe_pl            = {config.mgfe_pl}  (ref='H', delta = target-baseline)",
        f"  MW baseline table  = {len(config.mw_feh_grid)}-pt lib_mgfe([Fe/H])  (swappable)",
        f"  SLAVED (2p)        : {applied}",
        f"  O treatment        : [O/H] via the 'as' response (ah); S/Ne ride along",
        f"  LIGHT ELEMENTS     : {', '.join(config.light_elements)}  ->  mode='{config.light_element_mode}'",
        f"     ('solar'=held at baseline; 'follow_mg'=delta==delta_Mg; 'two_process'=predicted+cap/clamp)",
        f"  HELD SOLAR ({len(HELDSOL)})   : {', '.join(HELDSOL)}",
        f"  DROPPED ({len(DROPPED)})      : {', '.join(DROPPED)}  (no response column)",
    ]
    if config.light_element_mode == "two_process":
        lines.append(f"  C exponent cap     : |delta_C| <= {config.c_max_exponent} * 0.15")
        lines.append(f"  N validity [Mg/H]  : {config.n_valid_mgh} (clamp + flag outside)")
    return "\n".join(lines)


def compute_deltas(zh, feh, mgh, config: TwoProcessConfig | None = None
                   ) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Response deltas to inject into ``get_model`` for one ``(zh, feh, mgh)``.

    Returns
    -------
    deltas : dict
        ``{alfalpha_key: delta_dex}`` for every APPLIED element (Fe, Mg and all
        slaved elements incl. O via 'ah'). Ready to drop into the params dict
        that ``grids.get_model`` iterates over.
    info : dict
        Derived quantities and policy flags for logging: ``feh_true``,
        ``mgfe_true``, ``feh_lib``, ``mgfe_lib``, ``n_clamped`` (bool + value),
        ``c_capped`` (bool + exponent), ``dropped``, ``heldsol``.
    """
    config = config or TwoProcessConfig()
    mgfe_pl = config.mgfe_pl

    # --- target and library-baseline two-process points --------------------
    feh_true = zh + feh
    mgh_true = zh + mgh
    mgfe_true = mgh_true - feh_true            # == mgh - feh

    feh_lib = zh
    mgfe_lib = float(config.lib_mgfe(zh))

    target = two_process_abundances(feh_true, mgfe_true, ref="H", mgfe_pl=mgfe_pl)
    if config.subtract_baseline:
        baseline = two_process_abundances(feh_lib, mgfe_lib, ref="H", mgfe_pl=mgfe_pl)
    else:
        # "baseline OFF": pretend the library is solar-scaled (standard-alf's
        # implicit assumption). Every applied delta becomes the absolute [X/H].
        baseline = {e: 0.0 for e in target}

    light = set(config.light_elements)
    mode = config.light_element_mode
    info: Dict[str, object] = {
        "feh_true": float(feh_true), "mgfe_true": float(mgfe_true),
        "feh_lib": float(feh_lib), "mgfe_lib": mgfe_lib,
        "dropped": list(DROPPED), "heldsol": list(HELDSOL),
        "light_elements": list(config.light_elements), "light_mode": mode,
        "forced_solar": [], "follow_mg": [],
        "n_clamped": False, "n_clamp_mgh": None,
        "c_capped": False, "c_exponent": None,
    }

    # --- deltas for the ordinary slaved elements (Fe, Mg, O, Si, ... ) ------
    deltas: Dict[str, float] = {}
    for elem, key in ELEMENT_TO_KEY.items():
        if elem in light:
            continue                                   # handled below by policy
        d = float(target[elem]) - float(baseline[elem])
        if not np.isfinite(d):
            info.setdefault("undefined", []).append(elem)   # held solar, flagged
            continue
        deltas[key] = d

    # Mg delta is needed for the follow_mg policy (Mg is never a light element).
    delta_mg = deltas.get(ELEMENT_TO_KEY["Mg"],
                          float(target["Mg"]) - float(baseline["Mg"]))

    # --- light elements (C, N by default) per the chosen policy ------------
    for elem in config.light_elements:
        key = ELEMENT_TO_KEY.get(elem)
        if key is None:
            continue
        if mode == "solar":
            info["forced_solar"].append(elem)           # not applied -> delta 0
            continue
        if mode == "follow_mg":
            deltas[key] = delta_mg                       # track the Mg response
            info["follow_mg"].append(elem)
            continue
        # mode == 'two_process': apply the actual prediction, safeguarded.
        if elem == "N":
            d, info["n_clamped"], info["n_clamp_mgh"] = _nitrogen_delta(
                config, feh_true, mgfe_true, mgh_true, feh_lib, mgfe_lib)
        else:
            d = float(target[elem]) - float(baseline[elem])
        if not np.isfinite(d):
            info.setdefault("undefined", []).append(elem)
            continue
        if key == "ch":
            node = CONV_E["ch"]
            expo = d / node
            info["c_exponent"] = float(expo)
            if abs(expo) > config.c_max_exponent:
                d = np.sign(d) * config.c_max_exponent * node
                info["c_capped"] = True
        deltas[key] = d

    return deltas, info


def _nitrogen_delta(config, feh_true, mgfe_true, mgh_true, feh_lib, mgfe_lib):
    """delta_N = N(target) - N(baseline), each evaluated with [Mg/H] clamped into
    the CN-track validity window (holding the alpha enhancement mgfe fixed and
    moving [Fe/H] -- raising mgfe would push above the plateau). Only used when
    light_element_mode == 'two_process'. Returns (delta, clamped_bool, clamp_mgh).
    """
    n_lo, n_hi = config.n_valid_mgh
    eps = 1e-3                       # CN-track boundaries are exclusive
    mgh_lib = feh_lib + mgfe_lib

    def _n_at(mgfe_side, mgh_side):
        mgh_c = float(np.clip(mgh_side, n_lo + eps, n_hi - eps))
        val = two_process_abundances(mgh_c - mgfe_side, mgfe_side, ref="H",
                                     mgfe_pl=config.mgfe_pl, elements=["N"])["N"]
        return float(val), (mgh_c != mgh_side)

    n_tgt, tgt_bit = _n_at(mgfe_true, mgh_true)
    n_base, base_bit = _n_at(mgfe_lib, mgh_lib)
    return (n_tgt - n_base, bool(tgt_bit or base_bit),
            float(np.clip(mgh_true, n_lo, n_hi)))
