"""Two-process-mode validation ladder (ALFALPHA_TP.md §5).

Rungs:
  5a mock_recovery      : compose a mock at known (zH, feh, mgh), add noise, fit
                          in TP mode, check recovery within errors + corner plot.
  5b correction_off     : fit the SAME mock with the library baseline OFF and
                          show the slaved elements bias by ~ the library pattern
                          (this measures what the correction is worth).
  5c realdata_crosscheck: (scaffold) fit one galaxy in full-element mode and TP
                          mode; compare TP-predicted [X/H] to the full posteriors.

Kept deliberately light (emcee, a fitting window) so it runs on a laptop; the
knobs (S/N, walkers, steps, window) are arguments so the full-resolution
version is one call away.
"""
import os
import sys

import numpy as np

os.environ.setdefault("ALFA_INFILES", os.path.expanduser("~/Software/alfalpha/infiles/"))
os.environ.setdefault("ALFA_OUT", os.path.expanduser("~/Software/alfalpha/outfiles/"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alfa.grids import Grids
from alfa.read_data import Data
from alfa.polynorm import polynorm
from alfa.setup_params import get_properties, setup_params
from alfa.two_process_mode import TwoProcessConfig, compute_deltas
from alfa.two_process import two_process_abundances

OUT = os.path.join(os.path.dirname(__file__), "tp_validation_out")
os.makedirs(OUT, exist_ok=True)

FIT_PARAMS = ["velz", "sigma", "logage", "zH", "feh", "mgh", "jitter"]
TRUTH = {"velz": 0.0, "sigma": 200.0, "logage": 0.7, "zH": -0.2,
         "feh": -0.10, "mgh": 0.12, "jitter": 1.0}


def make_data(grids, truth, wave, sn, ires, seed=0):
    """Compose a noisy mock spectrum as a Data object."""
    rng = np.random.default_rng(seed)
    clean = grids.get_model(truth, outwave=wave)
    err = clean / sn
    flux = clean + rng.normal(0.0, err)
    d = Data()
    d.wave, d.flux, d.err = wave, flux, err
    d.mask = np.ones_like(wave)
    d.ires = np.full_like(wave, ires)
    d.fitting_regions = np.array([[wave.min(), wave.max()]])
    return d


def _make_lnprob(grids, data, priors, deg):
    lo = np.array([priors[k][0] for k in FIT_PARAMS])
    hi = np.array([priors[k][1] for k in FIT_PARAMS])

    def lnprob(theta):
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf
        params = get_properties(theta, FIT_PARAMS)
        mflux = grids.get_model(params, outwave=data.wave)
        _, mfluxnorm = polynorm(data, mflux, deg=deg)
        jit = params["jitter"]
        return -0.5 * np.nansum((data.flux - mfluxnorm) ** 2 / (data.err ** 2 * jit ** 2)
                                + np.log(2 * np.pi * data.err ** 2 * jit ** 2))
    return lnprob


def _run_emcee(lnprob, priors, nwalkers, nsteps, discard, seed=0):
    import emcee
    rng = np.random.default_rng(seed)
    ndim = len(FIT_PARAMS)
    p0 = np.array([TRUTH[k] for k in FIT_PARAMS])
    span = np.array([(priors[k][1] - priors[k][0]) for k in FIT_PARAMS])
    pos = p0 + 0.02 * span * rng.standard_normal((nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    sampler.run_mcmc(pos, nsteps, progress=False)
    return sampler.get_chain(discard=discard, flat=True)


def _priors():
    _, pr = setup_params(np.array(FIT_PARAMS), model="Conroy18")
    # widen a touch for a clean mock fit
    pr["feh"] = [-0.5, 0.5]; pr["mgh"] = [-0.5, 0.5]; pr["zH"] = [-1.0, 0.25]
    pr["sigma"] = [100, 400]; pr["logage"] = [0.0, 1.0]; pr["velz"] = [-100, 100]
    pr["jitter"] = [0.3, 3.0]
    return pr


def mock_recovery(sn=100.0, ires=100.0, nwalkers=32, nsteps=1500, discard=500,
                  wmin=3800.0, wmax=5600.0, seed=0):
    print(f"=== 5a MOCK RECOVERY  (S/N={sn}, {nwalkers}x{nsteps}) ===")
    grids = Grids(inst_res=ires, kroupa_shortcut=True, two_process_mode=True)
    wave = np.arange(wmin, wmax, 1.0)
    data = make_data(grids, TRUTH, wave, sn, ires, seed=seed)
    priors = _priors()
    lnprob = _make_lnprob(grids, data, priors, deg=18)
    chain = _run_emcee(lnprob, priors, nwalkers, nsteps, discard, seed=seed)

    print(f"  {'param':7s} {'truth':>8s} {'median':>8s} {'-1sig':>7s} {'+1sig':>7s}  pull")
    ok = True
    for i, k in enumerate(FIT_PARAMS):
        med = np.median(chain[:, i])
        lo, hi = np.percentile(chain[:, i], [16, 84])
        sig = 0.5 * (hi - lo)
        pull = (med - TRUTH[k]) / sig if sig > 0 else np.nan
        flag = "" if abs(pull) < 2 or k in ("jitter",) else "  <-- >2sig"
        if k in ("zH", "feh", "mgh") and abs(pull) >= 2:
            ok = False
        print(f"  {k:7s} {TRUTH[k]:8.3f} {med:8.3f} {lo-med:7.3f} {hi-med:+7.3f}  {pull:+.2f}{flag}")

    try:
        import corner
        fig = corner.corner(chain, labels=FIT_PARAMS,
                            truths=[TRUTH[k] for k in FIT_PARAMS], show_titles=True)
        fig.savefig(os.path.join(OUT, "mock_recovery_corner.png"), dpi=120)
        plt.close(fig)
        print(f"  corner -> {OUT}/mock_recovery_corner.png")
    except Exception as e:
        print("  (corner skipped:", e, ")")
    print("  RECOVERY:", "PASS" if ok else "CHECK (zH/feh/mgh off by >2sigma)")
    return chain


def correction_off(sn=100.0, ires=100.0, nwalkers=32, nsteps=1500, discard=500,
                   wmin=3800.0, wmax=5600.0, seed=0):
    """Fit the mock (composed WITH baseline) using a baseline-OFF model.

    Predicted effect: the slaved elements are recovered biased by ~ the library
    baseline pattern, because the model no longer removes it. We report, per
    slaved element, the predicted [X/H] at the recovered (feh,mgh) vs the truth.
    """
    print(f"=== 5b CORRECTION-OFF  (S/N={sn}) ===")
    grids_on = Grids(inst_res=ires, kroupa_shortcut=True, two_process_mode=True)
    wave = np.arange(wmin, wmax, 1.0)
    data = make_data(grids_on, TRUTH, wave, sn, ires, seed=seed)

    cfg_off = TwoProcessConfig(subtract_baseline=False)
    grids_off = Grids(inst_res=ires, kroupa_shortcut=True, two_process_mode=True,
                      two_process_config=cfg_off)
    priors = _priors()
    lnprob = _make_lnprob(grids_off, data, priors, deg=18)
    chain = _run_emcee(lnprob, priors, nwalkers, nsteps, discard, seed=seed)

    med = {k: np.median(chain[:, i]) for i, k in enumerate(FIT_PARAMS)}
    # truth vs baseline-off recovered, translated to slaved [X/H] predictions
    tp = TwoProcessConfig()
    feh_t, mgfe_t = TRUTH["zH"] + TRUTH["feh"], TRUTH["mgh"] - TRUTH["feh"]
    feh_r, mgfe_r = med["zH"] + med["feh"], med["mgh"] - med["feh"]
    pred_true = two_process_abundances(feh_t, mgfe_t, ref="H", mgfe_pl=tp.mgfe_pl)
    pred_reco = two_process_abundances(feh_r, mgfe_r, ref="H", mgfe_pl=tp.mgfe_pl)
    # predicted bias == library baseline that the OFF model failed to remove
    base = two_process_abundances(TRUTH["zH"], tp.lib_mgfe(TRUTH["zH"]),
                                  ref="H", mgfe_pl=tp.mgfe_pl)

    print(f"  recovered zH={med['zH']:+.3f} feh={med['feh']:+.3f} mgh={med['mgh']:+.3f}")
    print(f"  {'elem':4s} {'true[X/H]':>9s} {'reco[X/H]':>9s} {'obs bias':>8s} {'pred bias':>9s}")
    maxerr = 0.0
    for e in ["O", "Na", "Si", "Ca", "Ti", "Cr", "Ni", "Ba"]:
        tv, rv = float(pred_true[e]), float(pred_reco[e])
        obs = rv - tv
        pred = -float(base[e])   # OFF model omits the baseline => bias == -baseline
        maxerr = max(maxerr, abs(obs - pred))
        print(f"  {e:4s} {tv:9.3f} {rv:9.3f} {obs:8.3f} {pred:9.3f}")
    print(f"  obs bias == pred bias (= -library baseline) to {maxerr:.3f} dex "
          f"-> this is what the correction is worth.")
    return chain


def _make_lnprob_params(grids, data, priors, deg, params):
    lo = np.array([priors[k][0] for k in params])
    hi = np.array([priors[k][1] for k in params])

    def lnprob(theta):
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf
        p = get_properties(theta, params)
        mflux = grids.get_model(p, outwave=data.wave)
        _, mfluxnorm = polynorm(data, mflux, deg=deg)
        jit = p.get("jitter", 1.0)
        return -0.5 * np.nansum((data.flux - mfluxnorm) ** 2 / (data.err ** 2 * jit ** 2)
                                + np.log(2 * np.pi * data.err ** 2 * jit ** 2))
    return lnprob


def realdata_crosscheck(datafile=None, nwalkers=40, nsteps=800, discard=300,
                        deg=None, seed=0):
    """5c: fit one galaxy in full-element mode and TP mode; compare TP-predicted
    [X/H] to the full-mode posteriors element by element.

    NOTE: a science-grade result needs a CONVERGED fit (cluster-scale). Run with
    large nsteps for real numbers; the default here is a runnable smoke test.
    """
    from alfa.utils import correct_abundance
    print(f"=== 5c REAL-DATA CROSSCHECK ({'SMOKE' if nsteps < 3000 else 'full'}) ===")
    if datafile is None:
        datafile = os.path.join(os.path.dirname(__file__), "..", "indata", "test")
    data = Data(datafile)
    ires = data.ires
    priors = _priors()
    _, pr_full = setup_params(model="Conroy18")

    # --- full-element mode (representative element set) ---
    full_params = ["velz", "sigma", "logage", "zH", "feh", "mgh", "ah", "sih",
                   "cah", "tih", "ch", "nh", "crh", "jitter"]
    for k in full_params:
        priors.setdefault(k, pr_full[k])
    g_full = Grids(inst_res=ires, inst_res_wave=data.wave, kroupa_shortcut=True)
    lp_full = _make_lnprob_params(g_full, data, priors, deg, full_params)
    import emcee
    rng = np.random.default_rng(seed)
    start = np.array([np.clip(0.0, priors[k][0], priors[k][1]) if k not in ("sigma",)
                      else 250.0 for k in full_params])
    span = np.array([priors[k][1] - priors[k][0] for k in full_params])
    pos = start + 0.02 * span * rng.standard_normal((nwalkers, len(full_params)))
    s_full = emcee.EnsembleSampler(nwalkers, len(full_params), lp_full)
    s_full.run_mcmc(pos, nsteps, progress=False)
    ch_full = s_full.get_chain(discard=discard, flat=True)

    # --- TP mode ---
    g_tp = Grids(inst_res=ires, inst_res_wave=data.wave, kroupa_shortcut=True,
                 two_process_mode=True)
    lp_tp = _make_lnprob_params(g_tp, data, priors, deg, FIT_PARAMS)
    pos2 = np.array([np.clip(0.0, priors[k][0], priors[k][1]) if k != "sigma" else 250.0
                     for k in FIT_PARAMS]) + 0.02 * np.array(
                     [priors[k][1] - priors[k][0] for k in FIT_PARAMS]) * rng.standard_normal(
                     (nwalkers, len(FIT_PARAMS)))
    s_tp = emcee.EnsembleSampler(nwalkers, len(FIT_PARAMS), lp_tp)
    s_tp.run_mcmc(pos2, nsteps, progress=False)
    ch_tp = s_tp.get_chain(discard=discard, flat=True)

    fp = np.array(full_params)
    zH_f = ch_full[:, fp == "zH"].ravel()
    tzH = ch_tp[:, np.array(FIT_PARAMS) == "zH"].ravel()
    tfeh = ch_tp[:, np.array(FIT_PARAMS) == "feh"].ravel()
    tmgh = ch_tp[:, np.array(FIT_PARAMS) == "mgh"].ravel()
    tp_pred = two_process_abundances(tzH + tfeh, tmgh - tfeh, ref="H",
                                     mgfe_pl=TwoProcessConfig().mgfe_pl)
    elem_key = {"O": "ah", "Si": "sih", "Ca": "cah", "Ti": "tih", "C": "ch",
                "N": "nh", "Cr": "crh", "Fe": "feh", "Mg": "mgh"}
    print(f"  {'elem':4s} {'full [X/H]':>16s} {'TP-pred [X/H]':>16s}  agree?")
    for e, key in elem_key.items():
        if key not in full_params:
            continue
        full_xh = correct_abundance(zH_f, ch_full[:, fp == key].ravel(), key)
        fm, fs = np.median(full_xh), 0.5 * np.subtract(*np.percentile(full_xh, [84, 16]))
        tv = np.asarray(tp_pred[e], float)
        tm, ts = np.nanmedian(tv), 0.5 * np.subtract(*np.nanpercentile(tv, [84, 16]))
        agree = abs(fm - tm) < 2 * np.hypot(fs, ts)
        print(f"  {e:4s} {fm:8.3f}+/-{fs:5.3f} {tm:8.3f}+/-{ts:5.3f}   {'yes' if agree else 'NO'}")
    print("  (agreement for well-measured elements validates; disagreement for weak ones is the point)")
    if nsteps < 3000:
        print("  SMOKE ONLY -- numbers are under-converged, not a science result.")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "5a"
    kw = {}
    if "--fast" in sys.argv:
        kw = dict(nwalkers=24, nsteps=600, discard=250)
    if which in ("5a", "all"):
        mock_recovery(**kw)
    if which in ("5b", "all"):
        correction_off(**kw)
    if which in ("5c", "all"):
        sk = dict(nwalkers=30, nsteps=300, discard=120) if "--fast" in sys.argv else {}
        realdata_crosscheck(**sk)
