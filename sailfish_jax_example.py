"""Sailfish (nf-sed's basin search + tempered SMC, minimal version) on an alfα spectrum through the JAX
forward model, seeded from multi-start optima instead of the normalizing flow, and compared with NUTS.

    python sailfish_jax_example.py [indata/test] [--starts 32] [--seeds 512] [--nuts outfiles/test_nuts_jax.npz]

Steps
  1. L-BFGS-B from `--starts` random points in the prior box (in the sampler's w coordinate); the optima are
     clustered (w-distance < 1) to report how many distinct basins the optimiser finds and how good each is.
  2. `--seeds` starting points: the optima, resampled uniformly and jittered by 0.05 in w (the flow's job).
  3. sailfish.fit: polish, cluster, short tempered run per basin, anneal the winner -> posterior draws.
  4. Compare with the NUTS draws (same model, same priors): median and width per parameter.

Requires ~/Software/nf-sed/examples/sailfish_minimal (SAILFISH_PATH env var overrides).
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

os.environ.setdefault("ALFA_INFILES", os.path.expanduser("~/Software/alfalpha/infiles/"))
os.environ.setdefault("ALFA_OUT", os.path.expanduser("~/Software/alfalpha/outfiles/"))
os.environ.setdefault("ALFA_NEND", "6600")
sys.path.insert(0, os.environ.get("SAILFISH_PATH", os.path.expanduser("~/Software/nf-sed/examples/sailfish_minimal")))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import sailfish as sf  # noqa: E402
from alfa.grids import Grids  # noqa: E402
from alfa.read_data import Data  # noqa: E402
from alfa.setup_params import setup_params  # noqa: E402
from alfa.jax_model import JaxAlfModel  # noqa: E402

NAMES = ("velz", "sigma", "logage", "zH", "feh", "ch", "nh", "mgh", "nah", "ah", "sih", "cah",
         "tih", "crh", "teff", "jitter", "logemline_h", "logemline_oiii", "velz2", "sigma2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spectrum", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "indata", "test"))
    ap.add_argument("--starts", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=512)
    ap.add_argument("--nuts", default=os.path.join(os.environ["ALFA_OUT"], "test_nuts_jax.npz"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=64, help="particles per batched likelihood call (memory)")
    a = ap.parse_args()

    data = Data(a.spectrum)
    grids = Grids(inst_res=data.ires, inst_res_wave=data.wave, kroupa_shortcut=True)
    jm = JaxAlfModel(grids, data)
    _, priors = setup_params(list(NAMES), model="Conroy18")
    priors["feh"] = [-0.5, 0.5]
    lo = np.array([priors[n][0] for n in NAMES], float)
    hi = np.array([priors[n][1] for n in NAMES], float)
    D = len(NAMES)

    def loglike(theta, x):                      # x (the data) is already inside jm; Problem wants the signature
        return jm.lnlike_dict({n: theta[i] for i, n in enumerate(NAMES)})

    P = sf.Problem(names=list(NAMES), lo=lo, hi=hi, loglike=loglike, simulate=None, x_obs=np.asarray(data.flux))

    # The minimal Sailfish vmaps value+gradient over ALL particles in one call. For this model (a 16384-point FFT and
    # several interpolations per particle, all kept for the backward pass) that is ~7 GB at 512 particles and swaps a
    # laptop. Same batched interface, evaluated in chunks of `--chunk` particles (lax.map over vmap).
    def make_target_chunked(P, chunk=a.chunk):
        x = jnp.asarray(P.x_obs)
        span = jnp.asarray(P.hi - P.lo)

        def one(w, beta):
            th = sf.w_to_theta(P, w)
            ladj = jnp.sum(jnp.log(span) + jax.nn.log_sigmoid(w) + jax.nn.log_sigmoid(-w))
            ll = P.loglike(th, x)
            return ladj + beta * ll, ll

        vg = jax.vmap(jax.value_and_grad(one, has_aux=True), in_axes=(0, None))

        @jax.jit
        def batched(w, beta):
            n = w.shape[0]
            pad = (-n) % chunk
            wp = jnp.concatenate([w, jnp.zeros((pad, w.shape[1]))]).reshape(-1, chunk, w.shape[1])
            (lp, ll), g = jax.lax.map(lambda wc: vg(wc, beta), wp)
            return lp.reshape(-1)[:n], g.reshape(-1, w.shape[1])[:n], ll.reshape(-1)[:n]

        def call(w, beta):
            lp, g, ll = batched(jnp.asarray(w), float(beta))
            return np.asarray(lp), np.asarray(g), np.asarray(ll)
        return call

    sf.make_target = make_target_chunked          # fit() looks make_target up in the module at call time
    tgt = sf.make_target(P)

    # --- 1. multi-start L-BFGS-B in w ------------------------------------------------------------------
    rng = np.random.default_rng(a.seed)
    neg = jax.jit(jax.value_and_grad(lambda w: -tgt_one(w)))

    def tgt_one(w):
        th = sf.w_to_theta(P, w)
        ladj = jnp.sum(jnp.log(jnp.asarray(hi - lo)) + jax.nn.log_sigmoid(w) + jax.nn.log_sigmoid(-w))
        return ladj + loglike(th, None)

    def neg_np(w):
        v, g = neg(jnp.asarray(w))
        return float(v), np.asarray(g, float)

    t0 = time.time()
    w_start = sf.theta_to_w(P, lo + (hi - lo) * rng.random((a.starts, D)))
    optima, lp = [], []
    for k in range(a.starts):
        res = minimize(neg_np, w_start[k], jac=True, method="L-BFGS-B", options=dict(maxiter=3000))
        optima.append(res.x); lp.append(-res.fun)
    optima, lp = np.array(optima), np.array(lp)
    order, label, centres = np.argsort(-lp), -np.ones(a.starts, int), []
    for i in order:
        if label[i] < 0:
            label[(np.linalg.norm(optima - optima[i], axis=1) < 1.0) & (label < 0)] = len(centres)
            centres.append(i)
    print(f"multi-start: {a.starts} L-BFGS-B runs in {time.time() - t0:.0f} s; {len(centres)} distinct optima (w-distance >= 1):")
    for j, i in enumerate(centres):
        th = np.asarray(sf.w_to_theta(P, optima[i]))
        print(f"  basin {j}: {np.sum(label == j):2d} starts, log target {lp[i]:9.1f}, "
              f"logage {th[2]:.2f} zH {th[3]:+.2f} feh {th[4]:+.2f} mgh {th[7]:+.2f} sigma {th[1]:.0f} velz {th[0]:+.0f}")

    # --- 2. seeds from the optima (what the flow would have provided) --------------------------------------
    seeds_w = optima[rng.integers(0, a.starts, a.seeds)] + 0.05 * rng.standard_normal((a.seeds, D))

    # --- 3. Sailfish ---------------------------------------------------------------------------------------
    r = sf.fit(P, seeds_w, seed=a.seed)
    draws = r["draws"]

    # --- 4. compare with NUTS ------------------------------------------------------------------------------
    out = os.path.join(os.environ["ALFA_OUT"], os.path.basename(a.spectrum) + "_sailfish_jax.npz")
    np.savez(out, names=np.array(NAMES), draws=draws, optima=optima, optima_logtarget=lp, candidates=str(r["candidates"]),
             n_evals=r["n_evals"], wall=str(r["wall"]))
    print(f"saved {out}")
    if os.path.exists(a.nuts):
        nd = np.load(a.nuts)
        nuts = nd["draws"].reshape(-1, nd["draws"].shape[-1])
        nn = list(nd["names"])
        print(f"\n{'parameter':16s} {'NUTS median':>12s} {'NUTS sd':>8s} {'SF median':>12s} {'SF sd':>8s} {'dmed/sd':>8s} {'sd ratio':>9s}")
        for i, n in enumerate(NAMES):
            x, y = nuts[:, nn.index(n)], draws[:, i]
            print(f"{n:16s} {np.median(x):12.3f} {x.std():8.3f} {np.median(y):12.3f} {y.std():8.3f} "
                  f"{(np.median(y) - np.median(x)) / x.std():8.2f} {y.std() / x.std():9.2f}")
        mgfe_n = nuts[:, nn.index("mgh")] - nuts[:, nn.index("feh")]
        mgfe_s = draws[:, 7] - draws[:, 4]
        print(f"\n[Mg/Fe]: NUTS {np.median(mgfe_n):.3f} +/- {mgfe_n.std():.3f}; Sailfish {np.median(mgfe_s):.3f} +/- {mgfe_s.std():.3f}")
    else:
        print(f"(no NUTS file at {a.nuts}; skipping the comparison)")


if __name__ == "__main__":
    main()
