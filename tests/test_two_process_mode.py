"""Unit tests for the two-process interface layer (no grids needed)."""
import os
import numpy as np

os.environ.setdefault("ALFA_INFILES", os.path.expanduser("~/Software/alfalpha/infiles/"))

from alfa.two_process_mode import (
    compute_deltas, library_mgfe, TwoProcessConfig, describe_mapping,
    ELEMENT_TO_KEY,
)
from alfa.two_process import two_process_abundances


def approx(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


def test_fe_mg_deltas():
    zh, feh, mgh = -0.2, -0.1, 0.15
    deltas, info = compute_deltas(zh, feh, mgh)
    # delta_Fe == feh exactly (Fe response = the fitted offset)
    assert approx(deltas["feh"], feh), deltas["feh"]
    # delta_Mg == mgh - library_mgfe(zh)
    assert approx(deltas["mgh"], mgh - library_mgfe(zh)), deltas["mgh"]
    # derived truths
    assert approx(info["feh_true"], zh + feh)
    assert approx(info["mgfe_true"], mgh - feh)
    assert approx(info["feh_lib"], zh)
    print("  [ok] Fe/Mg deltas & derived truths")


def test_solar_library_point_zero_delta():
    # If the target equals the library baseline, every delta is 0.
    zh = 0.0
    feh = 0.0
    mgh = float(library_mgfe(zh))  # target mgfe == library mgfe
    deltas, info = compute_deltas(zh, feh, mgh)
    m = max(abs(v) for v in deltas.values())
    assert m < 1e-9, f"max|delta| = {m}"
    print(f"  [ok] target==baseline -> max|delta| = {m:.2e}")


def test_slaved_matches_two_process_difference():
    zh, feh, mgh = -0.3, 0.1, 0.25
    deltas, info = compute_deltas(zh, feh, mgh)
    tgt = two_process_abundances(info["feh_true"], info["mgfe_true"], ref="H", mgfe_pl=0.65)
    base = two_process_abundances(info["feh_lib"], info["mgfe_lib"], ref="H", mgfe_pl=0.65)
    for elem, key in ELEMENT_TO_KEY.items():
        if elem in ("C", "N"):  # subject to caps/clamps, checked separately
            continue
        expect = float(tgt[elem]) - float(base[elem])
        if np.isfinite(expect):
            assert approx(deltas.get(key, np.nan), expect, 1e-9), (elem, deltas.get(key), expect)
    print("  [ok] slaved elements == target-baseline")


def test_light_element_modes():
    zh, feh, mgh = -0.2, -0.1, 0.15
    # solar: C, N not applied
    ds, isol = compute_deltas(zh, feh, mgh, TwoProcessConfig(light_element_mode="solar"))
    assert "ch" not in ds and "nh" not in ds, ds
    assert set(isol["forced_solar"]) == {"C", "N"}
    # follow_mg (default): delta_C == delta_N == delta_Mg
    df, ifm = compute_deltas(zh, feh, mgh, TwoProcessConfig(light_element_mode="follow_mg"))
    assert approx(df["ch"], df["mgh"]) and approx(df["nh"], df["mgh"]), df
    assert set(ifm["follow_mg"]) == {"C", "N"}
    # two_process: applied from the prediction
    dt, itp = compute_deltas(zh, feh, mgh, TwoProcessConfig(light_element_mode="two_process"))
    assert "ch" in dt and "nh" in dt
    print("  [ok] light modes: solar (unapplied), follow_mg (==delta_Mg), two_process (predicted)")


def test_carbon_cap():
    # C cap only bites in two_process mode. Force a tiny cap at a finite-C point.
    cfg = TwoProcessConfig(light_element_mode="two_process", c_max_exponent=0.1)
    deltas, info = compute_deltas(0.0, 0.0, 0.3, cfg)
    assert info["c_exponent"] is not None, "C came back undefined at this point"
    assert info["c_capped"], info
    assert abs(deltas["ch"]) <= 0.1 * 0.15 + 1e-12
    print(f"  [ok] C cap bit: raw exponent {info['c_exponent']:.2f} -> delta_C={deltas['ch']:.4f}")
    d2, i2 = compute_deltas(-0.1, 0.0, 0.05,
                            TwoProcessConfig(light_element_mode="two_process", c_max_exponent=2.0))
    assert not i2["c_capped"], i2
    print(f"  [ok] C not capped at mild point (exponent {i2['c_exponent']:.2f})")


def test_nitrogen_clamp():
    # N clamp only runs in two_process mode.
    cfg = TwoProcessConfig(light_element_mode="two_process")
    deltas, info = compute_deltas(-0.8, 0.0, 0.0, cfg)
    assert info["n_clamped"], info
    assert np.isfinite(deltas.get("nh", np.nan)), info
    print(f"  [ok] N clamped at [Mg/H]={info['n_clamp_mgh']:.2f}, delta_N={deltas['nh']:.4f}")
    d2, i2 = compute_deltas(-0.3, 0.0, 0.1, cfg)
    assert not i2["n_clamped"], i2
    assert "undefined" not in i2, i2.get("undefined")
    print("  [ok] normal point (-0.3): no N clamp, no undefined elements")


def test_dropped_and_heldsol_absent():
    deltas, info = compute_deltas(-0.2, 0.0, 0.1)
    for k in ("alh", "yh", "ceh", "sh", "cuh", "euh"):
        assert k not in deltas, k
    print("  [ok] dropped (Al,Y,Ce,S) & heldsol (Cu,Eu) not applied")


if __name__ == "__main__":
    print(describe_mapping())
    print("-" * 60)
    for fn in [test_fe_mg_deltas, test_solar_library_point_zero_delta,
               test_slaved_matches_two_process_difference, test_light_element_modes,
               test_carbon_cap, test_nitrogen_clamp, test_dropped_and_heldsol_absent]:
        fn()
    print("-" * 60)
    print("ALL INTERFACE UNIT TESTS PASSED")
