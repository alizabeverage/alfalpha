"""NUTS on an alfα spectrum through the JAX forward model (alfa/jax_model.py) with blackjax.

    python nuts_jax_example.py [indata/test] [--warmup 500] [--draws 1000] [--chains 2]

The parameter set and priors are the Conroy18 ones from alfa_config.py (20 parameters). Sampling runs in an
unbounded coordinate w with theta = lo + (hi - lo) * sigmoid(w) (uniform box prior; the Jacobian is included),
climbs with L-BFGS-B from --starts random points in the box (plus the alfα default), reports the distinct
optima found, starts every chain at the best one (jittered), and uses blackjax's
window adaptation (step size + diagonal mass matrix) before the kept draws.

Outputs under $ALFA_OUT: <name>_nuts_jax.npz (draws, chain diagnostics) and <name>_nuts_jax_corner.png.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

os.environ.setdefault("ALFA_INFILES", os.path.expanduser("~/Software/alfalpha/infiles/"))
os.environ.setdefault("ALFA_OUT", os.path.expanduser("~/Software/alfalpha/outfiles/"))
os.environ.setdefault("ALFA_NEND", "6600")        # model grid to 1.3 um: enough for optical data, faster

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import blackjax  # noqa: E402

from alfa.grids import Grids  # noqa: E402
from alfa.read_data import Data  # noqa: E402
from alfa.setup_params import setup_params  # noqa: E402
from alfa.jax_model import JaxAlfModel  # noqa: E402

NAMES = ("velz", "sigma", "logage", "zH", "feh", "ch", "nh", "mgh", "nah", "ah", "sih", "cah",
         "tih", "crh", "teff", "jitter", "logemline_h", "logemline_oiii", "velz2", "sigma2")


# ------------------------------------------------------------------ diagnostics (no arviz dependency)
def split_rhat(x):
    """x: (chains, draws). Gelman-Rubin split R-hat."""
    c, n = x.shape
    half = n // 2
    x = np.concatenate([x[:, :half], x[:, half:2 * half]], 0)
    m, n = x.shape
    means = x.mean(1)
    B = n * means.var(ddof=1)
    W = x.var(1, ddof=1).mean()
    return np.sqrt(((n - 1) / n * W + B / n) / W)


def ess(x):
    """x: (chains, draws). Effective sample size via initial positive sequence of the pooled autocorrelation."""
    c, n = x.shape
    x = x - x.mean(1, keepdims=True)
    acov = np.zeros(n)
    for ch in x:
        f = np.fft.rfft(ch, 2 * n)
        acov += np.fft.irfft(f * np.conj(f))[:n] / n
    acov /= c
    rho = acov / acov[0]
    tau, t = 1.0, 1
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        tau += 2 * pair
        t += 2
    return c * n / tau


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spectrum", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "indata", "test"))
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--starts", type=int, default=32,
                    help="L-BFGS-B climbs before sampling (1 = from the alfα default only); the chains start at the best")
    ap.add_argument("--dense", action="store_true",
                    help="adapt a dense (full) mass matrix instead of a diagonal one; needed when parameters are strongly correlated")
    ap.add_argument("--poly", choices=["profile", "sample", "marginal"], default="profile",
                    help="profile = least-squares polynomial at every call (alfα); sample = coefficients are parameters; "
                         "marginal = coefficients integrated out analytically, recovered per draw afterwards")
    ap.add_argument("--poly-prior-sd", type=float, default=1e4, help="marginal mode: prior sd on each coefficient")
    a = ap.parse_args()

    data = Data(a.spectrum)
    grids = Grids(inst_res=data.ires, inst_res_wave=data.wave, kroupa_shortcut=True)
    jm = JaxAlfModel(grids, data, poly_mode=a.poly, poly_prior_sd=a.poly_prior_sd)
    defaults, priors = setup_params(list(NAMES), model="Conroy18")
    priors["feh"] = [-0.5, 0.5]                                   # as alfa_config.py
    names = list(NAMES)
    if a.poly == "sample":
        # start the polynomial at its least-squares values at the alfα default position; prior boxes:
        # +/- 2 dex on each region's normalisation, [-1, 1] on each relative shape coefficient
        pinit = jm.poly_init(dict(zip(NAMES, defaults)))
        for n in jm.poly_names:
            names.append(n)
            defaults.append(pinit[n])
            priors[n] = [pinit[n] - 2, pinit[n] + 2] if n.endswith("lognorm") else [-1.0, 1.0]
    names = tuple(names)
    lnlike = jm.make_lnlike(names)
    lo = jnp.array([priors[n][0] for n in names], float)
    hi = jnp.array([priors[n][1] for n in names], float)
    D = len(names)
    print(f"{D} parameters, polynomial mode '{a.poly}'")

    def to_theta(w):
        return lo + (hi - lo) * jax.nn.sigmoid(w)

    def to_w(theta):
        p = jnp.clip((theta - lo) / (hi - lo), 1e-6, 1 - 1e-6)
        return jnp.log(p) - jnp.log1p(-p)

    def logdensity(w):
        s = jax.nn.sigmoid(w)
        log_jac = jnp.sum(jnp.log(hi - lo) + jnp.log(s) + jnp.log1p(-s))
        return lnlike(to_theta(w)) + log_jac

    # --- start: multi-start L-BFGS-B, chains begin at the best optimum ------------------------------
    key = jax.random.key(a.seed)
    neg = jax.jit(jax.value_and_grad(lambda w: -logdensity(w)))

    def neg_np(w):
        v, g = neg(jnp.asarray(w))
        return float(v), np.asarray(g, float)

    rng = np.random.default_rng(a.seed)
    w0 = np.asarray(to_w(jnp.array(defaults, float)))
    w_init = [w0] + [np.asarray(to_w(jnp.asarray(lo + (hi - lo) * rng.random(D)))) for _ in range(a.starts - 1)]
    t0 = time.time()
    optima, lp = [], []
    for wc in w_init:
        res = minimize(neg_np, wc, jac=True, method="L-BFGS-B", options=dict(maxiter=3000))
        optima.append(res.x); lp.append(-res.fun)
    optima, lp = np.array(optima), np.array(lp)
    # cluster the optima: greedy, best first, radius 1 in w
    order, label, centres = np.argsort(-lp), -np.ones(len(lp), int), []
    for i in order:
        if label[i] < 0:
            label[(np.linalg.norm(optima - optima[i], axis=1) < 1.0) & (label < 0)] = len(centres)
            centres.append(i)
    print(f"multi-start: {a.starts} L-BFGS-B climbs ({len(w_init) - 1} from random points in the prior box + the alfα "
          f"default) in {time.time() - t0:.0f} s; {len(centres)} distinct optima:")
    for j, i in enumerate(centres):
        th = np.asarray(to_theta(jnp.asarray(optima[i])))
        print(f"  basin {j}: {np.sum(label == j):2d} climbs, log-density {lp[i]:9.1f}" +
              "".join(f" {n} {th[names.index(n)]:+.3g}" for n in ("logage", "zH", "feh", "mgh", "sigma", "velz") if n in names))
    if len(centres) > 1 and lp[centres[0]] - lp[centres[1]] < 10:
        print("  WARNING: a second basin lies within 10 log-density units of the best; the posterior may be "
              "multimodal and these chains sample only the best basin (run Sailfish, or NUTS from each basin).")
    best = optima[centres[0]]
    starts = []
    for c in range(a.chains):
        key, k = jax.random.split(key)
        starts.append(jnp.asarray(best) + 0.05 * jax.random.normal(k, (D,)))

    # --- NUTS with window adaptation -------------------------------------------------------------
    draws = np.zeros((a.chains, a.draws, D))
    stats = dict(accept=np.zeros((a.chains, a.draws)), divergent=np.zeros((a.chains, a.draws), bool),
                 n_steps=np.zeros((a.chains, a.draws)), logdensity=np.zeros((a.chains, a.draws)))
    step_sizes = []
    for c in range(a.chains):
        key, k_adapt, k_sample = jax.random.split(key, 3)
        t0 = time.time()
        adapt = blackjax.window_adaptation(blackjax.nuts, logdensity, target_acceptance_rate=0.8,
                                           is_mass_matrix_diagonal=not a.dense)
        (state, params), _ = adapt.run(k_adapt, starts[c], num_steps=a.warmup)
        t_warm = time.time() - t0
        step_sizes.append(float(params["step_size"]))
        kernel = blackjax.nuts(logdensity, **params).step

        def one_step(state, k):
            state, info = kernel(k, state)
            return state, (state.position, info.acceptance_rate, info.is_divergent,
                           info.num_integration_steps, state.logdensity)

        t0 = time.time()
        keys = jax.random.split(k_sample, a.draws)
        _, (pos, acc, div, nst, ld) = jax.lax.scan(one_step, state, keys)
        pos.block_until_ready()
        t_samp = time.time() - t0
        draws[c] = np.asarray(jax.vmap(to_theta)(pos))
        stats["accept"][c], stats["divergent"][c] = np.asarray(acc), np.asarray(div)
        stats["n_steps"][c], stats["logdensity"][c] = np.asarray(nst), np.asarray(ld)
        print(f"chain {c}: warmup {a.warmup} steps in {t_warm:.0f} s; {a.draws} draws in {t_samp:.0f} s "
              f"({1e3 * t_samp / a.draws:.0f} ms/draw, {stats['n_steps'][c].mean():.0f} leapfrogs/draw); "
              f"step size {step_sizes[-1]:.3f}; acceptance {stats['accept'][c].mean():.2f}; "
              f"divergent {int(stats['divergent'][c].sum())}")

    # --- summary -----------------------------------------------------------------------------------
    print(f"\n{'parameter':16s} {'median':>9s} {'-1sig':>8s} {'+1sig':>8s} {'R-hat':>6s} {'ESS':>6s}")
    for i, n in enumerate(names):
        x = draws[:, :, i]
        q16, q50, q84 = np.percentile(x, [16, 50, 84])
        print(f"{n:16s} {q50:9.3f} {q50 - q16:8.3f} {q84 - q50:8.3f} {split_rhat(x):6.3f} {ess(x):6.0f}")
    flat = draws.reshape(-1, D)
    print(f"\n[Mg/Fe] = mgh - feh: {np.median(flat[:, 7] - flat[:, 4]):.3f} "
          f"+/- {np.std(flat[:, 7] - flat[:, 4]):.3f}")
    extra = {}
    if a.poly == "marginal":
        # the polynomial for every draw: its Gaussian conditional given the data at that draw
        cond = jax.jit(jax.vmap(jm.make_poly_conditional(names)))
        means, sds = (np.asarray(x) for x in cond(jnp.asarray(flat)))
        extra = dict(poly_cond_mean=means, poly_cond_sd=sds)
        k0 = 0
        print("polynomial per draw (conditional on each draw):")
        for r, reg in enumerate(jm.regions):
            K = reg["A_full"].shape[1]
            lognorm = np.log10(1.0 + means[:, k0])
            print(f"  region {r}: log10 normalisation median {np.median(lognorm):.4f}, spread over draws "
                  f"{lognorm.std():.4f} dex, typical conditional sd {np.median(sds[:, k0] / (1 + means[:, k0]) / np.log(10)):.1e} dex; "
                  f"shape coefficients c1..c{K - 1} spread over draws {np.round(means[:, k0 + 1:k0 + K].std(0), 3)}")
            k0 += K

    # --- save --------------------------------------------------------------------------------------
    name = os.path.basename(a.spectrum)
    out = os.path.join(os.environ["ALFA_OUT"], f"{name}_nuts_jax" + ({"sample": "_polysample", "marginal": "_polymarginal"}.get(a.poly, "")) + ("_dense" if a.dense else ""))
    np.savez(out + ".npz", names=np.array(names), draws=draws, step_sizes=np.array(step_sizes), **stats, **extra)
    try:
        import corner
        import matplotlib
        matplotlib.use("Agg")
        show = ["velz", "sigma", "logage", "zH", "feh", "mgh", "ch", "nh", "nah", "jitter"]
        if a.poly == "sample":
            show += ["poly0_lognorm", "poly0_a1", "poly1_lognorm", "poly1_a1"]
        idx = [names.index(s) for s in show]
        fig = corner.corner(flat[:, idx], labels=show, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                            title_fmt=".3f", title_kwargs=dict(fontsize=9))
        fig.savefig(out + "_corner.png", dpi=110)
        print(f"saved {out}.npz and {out}_corner.png")
    except Exception as e:                                       # noqa: BLE001
        print(f"saved {out}.npz (corner plot skipped: {e})")


if __name__ == "__main__":
    main()
