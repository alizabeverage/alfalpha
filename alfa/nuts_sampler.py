"""The 'nuts' sampler for alfα: gradient-based NUTS (blackjax) on the JAX forward model (alfa/jax_model.py).

Selected in alfa_config.py with ``fitting_info.sampler = 'nuts'``; the settings are the ``nuts_*`` fields of
``Info`` (warmup, draws, chains, starts, poly, dense, seed). Conroy18 grids only (sMILES and two-process
mode are not ported to JAX and are refused).

What one call to ``run_nuts`` does
  1. climbs with L-BFGS-B from ``nuts_starts`` points (the alfα default + random points in the prior box),
     clusters the optima and prints how many distinct basins were found and how good each is;
  2. runs ``nuts_chains`` NUTS chains from the best optimum (window adaptation of step size and mass
     matrix over ``nuts_warmup`` steps, then ``nuts_draws`` draws each);
  3. prints per-parameter medians, 16/84 % widths, split R-hat and effective sample size;
  4. writes ``<ALFA_OUT><filename>_nuts.npz`` (draws (chains, draws, D), names, diagnostics, and for
     poly='marginal' the per-draw conditional polynomial coefficients), which ``post_process`` reads.

Polynomial treatment (``nuts_poly``): 'marginal' (default) integrates the continuum polynomial out
analytically; 'profile' is alfα's least-squares-at-every-call; 'sample' adds the coefficients as
parameters (slow, needs ``nuts_dense``). See JaxAlfModel's docstring.
"""
from __future__ import annotations

import os
import time

import numpy as np
from scipy.optimize import minimize

import jax
import jax.numpy as jnp

from alfa.jax_model import JaxAlfModel
from alfa.setup_params import setup_params

NUTS_DEFAULTS = dict(nuts_warmup=500, nuts_draws=1000, nuts_chains=2, nuts_starts=32,
                     nuts_poly="marginal", nuts_dense=False, nuts_seed=0, nuts_poly_prior_sd=1e4)


# ------------------------------------------------------------------ diagnostics
def split_rhat(x):
    """x: (chains, draws). Gelman-Rubin split R-hat."""
    c, n = x.shape
    half = n // 2
    x = np.concatenate([x[:, :half], x[:, half:2 * half]], 0)
    m, n = x.shape
    B = n * x.mean(1).var(ddof=1)
    W = x.var(1, ddof=1).mean()
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


def ess(x):
    """x: (chains, draws). Effective sample size (initial positive sequence of the pooled autocorrelation)."""
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
    return float(c * n / tau)


# ------------------------------------------------------------------ the sampler
def run_nuts(fitting_info, data, grids, verbose=True):
    """Fit ``data`` with NUTS. Returns the flat posterior samples (chains * draws, D) in the order of
    ``fitting_info.parameters_to_fit``; also writes ``<ALFA_OUT><filename>_nuts.npz``."""
    import blackjax

    if fitting_info.model != "Conroy18":
        raise ValueError("sampler 'nuts' needs model='Conroy18' (the sMILES grids are not ported to JAX)")
    if getattr(fitting_info, "two_process_mode", False):
        raise ValueError("sampler 'nuts' does not support two_process_mode (its element mapping is not in JAX)")
    cfg = {k: getattr(fitting_info, k, v) for k, v in NUTS_DEFAULTS.items()}
    names = tuple(str(n) for n in fitting_info.parameters_to_fit)
    D = len(names)
    deg = None if fitting_info.poly_degree == "default" else int(fitting_info.poly_degree)

    jm = JaxAlfModel(grids, data, poly_degree=deg, poly_mode=cfg["nuts_poly"], poly_prior_sd=cfg["nuts_poly_prior_sd"])
    full_names = list(names)
    defaults, _ = setup_params(list(names), model="Conroy18")
    priors = {n: list(fitting_info.priors[n]) for n in names}
    if cfg["nuts_poly"] == "sample":
        pinit = jm.poly_init(dict(zip(names, defaults)))
        for n in jm.poly_names:
            full_names.append(n)
            defaults.append(pinit[n])
            priors[n] = [pinit[n] - 2, pinit[n] + 2] if n.endswith("lognorm") else [-1.0, 1.0]
    full_names = tuple(full_names)
    Dfull = len(full_names)
    lnlike = jm.make_lnlike(full_names)
    lo = jnp.array([priors[n][0] for n in full_names], float)
    hi = jnp.array([priors[n][1] for n in full_names], float)

    def to_theta(w):
        return lo + (hi - lo) * jax.nn.sigmoid(w)

    def to_w(theta):
        p = jnp.clip((theta - lo) / (hi - lo), 1e-6, 1 - 1e-6)
        return jnp.log(p) - jnp.log1p(-p)

    def logdensity(w):
        s = jax.nn.sigmoid(w)
        return lnlike(to_theta(w)) + jnp.sum(jnp.log(hi - lo) + jnp.log(s) + jnp.log1p(-s))

    # --- 1. multi-start L-BFGS-B ----------------------------------------------------------------
    neg = jax.jit(jax.value_and_grad(lambda w: -logdensity(w)))

    def neg_np(w):
        v, g = neg(jnp.asarray(w))
        return float(v), np.asarray(g, float)

    rng = np.random.default_rng(cfg["nuts_seed"])
    w_init = [np.asarray(to_w(jnp.array(defaults, float)))]
    w_init += [np.asarray(to_w(jnp.asarray(lo + (hi - lo) * rng.random(Dfull)))) for _ in range(cfg["nuts_starts"] - 1)]
    t0 = time.time()
    optima, lp = [], []
    for wc in w_init:
        res = minimize(neg_np, wc, jac=True, method="L-BFGS-B", options=dict(maxiter=3000))
        optima.append(res.x); lp.append(-res.fun)
    optima, lp = np.array(optima), np.array(lp)
    order, label, centres = np.argsort(-lp), -np.ones(len(lp), int), []
    for i in order:
        if label[i] < 0:
            label[(np.linalg.norm(optima - optima[i], axis=1) < 1.0) & (label < 0)] = len(centres)
            centres.append(i)
    multimodal = len(centres) > 1 and lp[centres[0]] - lp[centres[1]] < 10
    if verbose:
        print(f"[nuts] multi-start: {len(w_init)} L-BFGS-B climbs in {time.time() - t0:.0f} s; {len(centres)} distinct optima:")
        for j, i in enumerate(centres):
            th = np.asarray(to_theta(jnp.asarray(optima[i])))
            print(f"  basin {j}: {np.sum(label == j):2d} climbs, log-density {lp[i]:9.1f}" +
                  "".join(f" {n} {th[full_names.index(n)]:+.3g}" for n in ("logage", "zH", "feh", "mgh", "sigma", "velz") if n in full_names))
        if multimodal:
            print("  WARNING: a second basin lies within 10 log-density units of the best; the posterior may be "
                  "multimodal and these chains sample only the best basin.")
    best = jnp.asarray(optima[centres[0]])

    # --- 2. NUTS ---------------------------------------------------------------------------------
    key = jax.random.key(cfg["nuts_seed"])
    C, N = cfg["nuts_chains"], cfg["nuts_draws"]
    draws = np.zeros((C, N, Dfull))
    accept, divergent, n_steps, logdens = (np.zeros((C, N)) for _ in range(4))
    step_sizes, walls = [], []
    for c in range(C):
        key, k_start, k_adapt, k_sample = jax.random.split(key, 4)
        start = best + 0.05 * jax.random.normal(k_start, (Dfull,))
        t0 = time.time()
        adapt = blackjax.window_adaptation(blackjax.nuts, logdensity, target_acceptance_rate=0.8,
                                           is_mass_matrix_diagonal=not cfg["nuts_dense"])
        (state, params), _ = adapt.run(k_adapt, start, num_steps=cfg["nuts_warmup"])
        kernel = blackjax.nuts(logdensity, **params).step

        def one_step(state, k):
            state, info = kernel(k, state)
            return state, (state.position, info.acceptance_rate, info.is_divergent, info.num_integration_steps, state.logdensity)

        _, (pos, acc, div, nst, ld) = jax.lax.scan(one_step, state, jax.random.split(k_sample, N))
        pos.block_until_ready()
        walls.append(time.time() - t0)
        step_sizes.append(float(params["step_size"]))
        draws[c] = np.asarray(jax.vmap(to_theta)(pos))
        accept[c], divergent[c], n_steps[c], logdens[c] = (np.asarray(x) for x in (acc, div, nst, ld))
        if verbose:
            print(f"[nuts] chain {c}: warmup {cfg['nuts_warmup']} + {N} draws in {walls[-1]:.0f} s "
                  f"({n_steps[c].mean():.0f} leapfrogs/draw); step size {step_sizes[-1]:.3f}; "
                  f"acceptance {accept[c].mean():.2f}; divergent {int(divergent[c].sum())}")

    # --- 3. diagnostics ---------------------------------------------------------------------------
    rhat = np.array([split_rhat(draws[:, :, i]) for i in range(Dfull)])
    n_eff = np.array([ess(draws[:, :, i]) for i in range(Dfull)])
    if verbose:
        print(f"\n{'parameter':16s} {'median':>9s} {'-1sig':>8s} {'+1sig':>8s} {'R-hat':>6s} {'ESS':>6s}")
        for i, n in enumerate(full_names):
            q16, q50, q84 = np.percentile(draws[:, :, i], [16, 50, 84])
            print(f"{n:16s} {q50:9.3f} {q50 - q16:8.3f} {q84 - q50:8.3f} {rhat[i]:6.3f} {n_eff[i]:6.0f}")
        bad = [n for n, r, e in zip(full_names, rhat, n_eff) if r > 1.05 or e < 100]
        if bad:
            print(f"WARNING: poorly converged (R-hat > 1.05 or ESS < 100): {bad}; increase nuts_draws / nuts_warmup")

    # --- 4. save ----------------------------------------------------------------------------------
    extra = {}
    if cfg["nuts_poly"] == "marginal":
        cond = jax.jit(jax.vmap(jm.make_poly_conditional(full_names)))
        means, sds = (np.asarray(x) for x in cond(jnp.asarray(draws.reshape(-1, Dfull))))
        extra = dict(poly_cond_mean=means, poly_cond_sd=sds, poly_names=np.array(jm.poly_names))
    out = os.path.join(fitting_info.ALFA_OUT, f"{fitting_info.filename}_nuts.npz")
    np.savez(out, names=np.array(full_names), draws=draws, rhat=rhat, ess=n_eff, accept=accept, divergent=divergent,
             n_steps=n_steps, logdensity=logdens, step_sizes=np.array(step_sizes), wall=np.array(walls),
             optima=np.asarray(jax.vmap(to_theta)(jnp.asarray(optima))), optima_logdensity=lp, multimodal=multimodal,
             config=str(cfg), **extra)
    if verbose:
        print(f"[nuts] saved {out}")
    return draws[:, :, :D].reshape(-1, D)


def load_nuts_samples(fitting_info):
    """Flat samples (chains * draws, D) in the order of parameters_to_fit, from the npz run_nuts wrote."""
    d = np.load(os.path.join(fitting_info.ALFA_OUT, f"{fitting_info.filename}_nuts.npz"))
    D = len(fitting_info.parameters_to_fit)
    return d["draws"][:, :, :D].reshape(-1, D)
