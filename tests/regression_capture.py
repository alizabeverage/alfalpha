"""Capture / check a standard-mode (Conroy18) forward-model reference.

Run with `--capture` BEFORE any two-process changes to freeze the reference,
then with no args AFTER to assert the standard model path is bit-for-bit
unchanged (spec item 4).

The reference is the raw get_model output on the native SSP wavelength grid
with inst_res=0 (no smoothing, fully deterministic) for a fixed parameter
vector that exercises the SSP interpolation, the alpha-lockstep branch is
avoided, several individual element responses, teff, and emission lines.
"""
import os
import sys

import numpy as np

os.environ.setdefault("ALFA_INFILES", os.path.expanduser("~/Software/alfalpha/infiles/"))
os.environ.setdefault("ALFA_OUT", os.path.expanduser("~/Software/alfalpha/outfiles/"))

from alfa.grids import Grids

REF = os.path.join(os.path.dirname(__file__), "reference_standard_model.npz")

# A parameter vector touching age/Z interpolation, individual elements,
# teff, emission lines, sigma broadening and a redshift.
PARAMS = {
    "velz": 30.0, "sigma": 220.0, "logage": 0.7, "zH": -0.2,
    "feh": -0.1, "ch": 0.05, "nh": 0.1, "nah": 0.2, "mgh": 0.15,
    "sih": 0.05, "kh": 0.0, "cah": 0.08, "tih": 0.03, "ah": 0.1,
    "crh": -0.05, "teff": 15.0,
    "logemline_h": -1.0, "logemline_oiii": -1.5, "velz2": 10.0, "sigma2": 250.0,
    "jitter": 1.0,
}


def build():
    grids = Grids(inst_res=0, kroupa_shortcut=True)
    spec = grids.get_model(PARAMS)  # native wave grid
    return grids.ssp.wave.copy(), spec


def main():
    wave, spec = build()
    if "--capture" in sys.argv:
        np.savez(REF, wave=wave, spec=spec, params=np.array(list(PARAMS.items()), dtype=object))
        print(f"captured reference -> {REF}  ({spec.size} pts, mean={np.nanmean(spec):.6e})")
        return
    ref = np.load(REF, allow_pickle=True)
    dw = np.max(np.abs(wave - ref["wave"]))
    ds = np.max(np.abs(spec - ref["spec"]))
    print(f"max|dwave| = {dw:.3e}   max|dspec| = {ds:.3e}")
    assert dw == 0.0, "wavelength grid changed"
    assert ds == 0.0, "STANDARD MODE CHANGED — not bit-for-bit identical"
    print("PASS: standard mode reproduces the stored reference bit-for-bit")


if __name__ == "__main__":
    main()
