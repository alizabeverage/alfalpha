"""The JAX forward model (alfa/jax_model.py) reproduces the numpy one and is differentiable.

Setup: indata/test (LEGA-C-like spectrum, 2 fitting regions, 87 masked pixels), the Conroy18
grids smoothed to the data's instrumental resolution, the 20-parameter set of alfa_config.py.
ALFA_NEND=6600 (grid to 1.3 um) keeps the grid build to a few seconds; the port is grid-agnostic.

Bars (set before the first run):
  * normalised model spectrum: max |jax - numpy| / |numpy| < 1e-5 over the fitted pixels, 50 draws
  * log-likelihood: |jax - numpy| < 1e-6 * |numpy|, 50 draws
  * gradient: finite on every draw; against central finite differences, relative error < 1e-3 on the
    smooth parameters (sigma, sigma2, logemline_*, jitter); the interpolation-kinked parameters
    (logage, zH, elements, teff, velz, velz2) are compared with the same bar but a draw that straddles
    a node is allowed to fail it, so for them the bar is "< 1e-3 on at least 80 % of draws".
Run:  pytest tests/test_jax_model.py -s   (the -s prints the timing block)
"""
import os
import sys
import time

import numpy as np
import pytest

os.environ.setdefault("ALFA_INFILES", os.path.expanduser("~/Software/alfalpha/infiles/"))
os.environ.setdefault("ALFA_OUT", os.path.expanduser("~/Software/alfalpha/outfiles/"))
os.environ.setdefault("ALFA_NEND", "6600")

jax = pytest.importorskip("jax")

from alfa.grids import Grids  # noqa: E402
from alfa.read_data import Data  # noqa: E402
from alfa.polynorm import polynorm  # noqa: E402
from alfa.setup_params import setup_params, get_properties  # noqa: E402
from alfa.jax_model import JaxAlfModel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ("velz", "sigma", "logage", "zH", "feh", "ch", "nh", "mgh", "nah", "ah", "sih", "cah",
         "tih", "crh", "teff", "jitter", "logemline_h", "logemline_oiii", "velz2", "sigma2")
SMOOTH = ("sigma", "sigma2", "logemline_h", "logemline_oiii", "jitter")
N_DRAWS = 50


@pytest.fixture(scope="module")
def setup():
    data = Data(os.path.join(HERE, "..", "indata", "test"))
    grids = Grids(inst_res=data.ires, inst_res_wave=data.wave, kroupa_shortcut=True)
    jm = JaxAlfModel(grids, data)
    _, priors = setup_params(NAMES, model="Conroy18")
    priors["feh"] = [-0.5, 0.5]
    priors["sigma"] = [110.0, 500.0]      # > model_res: the numpy path skips smoothing below 100 km/s
    lo = np.array([priors[n][0] for n in NAMES], float)
    hi = np.array([priors[n][1] for n in NAMES], float)
    rng = np.random.default_rng(20261020)
    thetas = lo + (hi - lo) * rng.random((N_DRAWS, len(NAMES)))
    return dict(data=data, grids=grids, jm=jm, lo=lo, hi=hi, thetas=thetas)


def numpy_lnlike(theta, data, grids):
    params = get_properties(theta, NAMES)
    mflux = grids.get_model(params, outwave=data.wave)
    poly, mnorm = polynorm(data, mflux)
    var = data.err ** 2 * params["jitter"] ** 2
    return mnorm, -0.5 * np.nansum((data.flux - mnorm) ** 2 / var + np.log(2 * np.pi * var))


def test_model_and_likelihood_match_numpy(setup):
    jm, data, grids = setup["jm"], setup["data"], setup["grids"]
    f_norm = jm.make_model_norm(NAMES)
    f_ll = jm.make_lnlike(NAMES)
    use = np.asarray(jm.use)
    worst_spec, worst_ll = 0.0, 0.0
    for theta in setup["thetas"]:
        mnorm_np, ll_np = numpy_lnlike(theta, data, grids)
        _, mnorm_jx = f_norm(theta)
        ll_jx = float(f_ll(theta))
        rel = np.max(np.abs(np.asarray(mnorm_jx)[use] - mnorm_np[use]) / np.abs(mnorm_np[use]))
        worst_spec = max(worst_spec, rel)
        worst_ll = max(worst_ll, abs(ll_jx - ll_np) / abs(ll_np))
        assert np.isfinite(ll_jx)
    print(f"\nworst relative spectrum difference {worst_spec:.2e}; worst relative lnlike difference {worst_ll:.2e}")
    assert worst_spec < 1e-5, worst_spec
    assert worst_ll < 1e-6, worst_ll


def test_gradient_is_finite_and_matches_finite_differences(setup):
    jm, lo, hi = setup["jm"], setup["lo"], setup["hi"]
    f_ll = jm.make_lnlike(NAMES)
    g_ll = jax.jit(jax.grad(f_ll))
    h = 1e-5 * (hi - lo)
    n_check = 10
    rel_err = np.zeros((n_check, len(NAMES)))
    for k, theta in enumerate(setup["thetas"][:n_check]):
        g = np.asarray(g_ll(theta))
        assert np.all(np.isfinite(g)), g
        for i in range(len(NAMES)):
            e = np.zeros(len(NAMES)); e[i] = h[i]
            fd = (float(f_ll(theta + e)) - float(f_ll(theta - e))) / (2 * h[i])
            rel_err[k, i] = abs(g[i] - fd) / max(abs(fd), 1e-8)
    print("\nrelative |grad - finite difference|, median / max over draws:")
    for i, n in enumerate(NAMES):
        print(f"  {n:16s} {np.median(rel_err[:, i]):.1e} / {rel_err[:, i].max():.1e}")
    for i, n in enumerate(NAMES):
        frac_ok = np.mean(rel_err[:, i] < 1e-3)
        if n in SMOOTH:
            assert rel_err[:, i].max() < 1e-3, (n, rel_err[:, i])
        else:
            assert frac_ok >= 0.8, (n, rel_err[:, i])


def test_timing(setup, capsys):
    jm, data, grids = setup["jm"], setup["data"], setup["grids"]
    theta = setup["thetas"][0]
    f_ll = jm.make_lnlike(NAMES)
    g_ll = jax.jit(jax.grad(f_ll))
    f_ll(theta).block_until_ready(); g_ll(theta).block_until_ready()      # compile
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        numpy_lnlike(theta, data, grids)
    t_np = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for _ in range(n):
        f_ll(theta).block_until_ready()
    t_jx = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for _ in range(n):
        g_ll(theta).block_until_ready()
    t_gx = (time.perf_counter() - t0) / n
    with capsys.disabled():
        print(f"\nTIMING (model grid {int(jm.wave.shape[0])} px, data {int(jm.outwave.shape[0])} px, "
              f"{jm.n_use} fitted; {len(NAMES)} parameters; {jax.devices()[0].platform})")
        print(f"  numpy get_model + polynorm + lnlike : {1e3 * t_np:7.2f} ms")
        print(f"  jitted JAX lnlike (warm)            : {1e3 * t_jx:7.2f} ms")
        print(f"  jitted JAX grad(lnlike) (warm)      : {1e3 * t_gx:7.2f} ms")


def test_sampled_polynomial_reproduces_the_profiled_model_at_its_optimum(setup):
    """poly_mode='sample' with the coefficients at their least-squares values == the profiled model,
    and moving a shape coefficient changes the normalised model only inside its region."""
    from alfa.jax_model import JaxAlfModel
    jm_s = JaxAlfModel(setup["grids"], setup["data"], poly_mode="sample")
    jm_p = setup["jm"]
    assert len(jm_s.poly_names) == 6 + 12, jm_s.poly_names       # degree 5 and 11 for the two regions
    for theta in setup["thetas"][:5]:
        params = get_properties(theta, NAMES)
        params_s = {**params, **jm_s.poly_init(params)}
        _, mp = jm_p.model_norm(params)
        _, ms = jm_s.model_norm(params_s)
        assert np.max(np.abs(np.asarray(ms - mp)) / np.abs(np.asarray(mp) + 1e-300)) < 1e-10
        assert abs(float(jm_s.lnlike_dict(params_s)) - float(jm_p.lnlike_dict(params))) < 1e-6
        # perturb one shape coefficient of region 0: region 1 untouched, lnlike worse
        params_t = {**params_s, "poly0_a1": params_s["poly0_a1"] + 0.05}
        _, mt = jm_s.model_norm(params_t)
        idx1 = np.asarray(jm_s.regions[1]["idx"]); idx0 = np.asarray(jm_s.regions[0]["idx"])
        only0 = np.setdiff1d(idx0, idx1)
        assert np.array_equal(np.asarray(mt)[idx1], np.asarray(ms)[idx1])
        assert not np.allclose(np.asarray(mt)[only0], np.asarray(ms)[only0])
        assert float(jm_s.lnlike_dict(params_t)) < float(jm_s.lnlike_dict(params_s))


def test_marginal_likelihood_against_dense_brute_force(setup):
    """poly_mode='marginal': the Woodbury/Cholesky marginal equals a dense N x N evaluation of
    N(flux; model, diag(err^2 j^2) + sd^2 B B^T) per region (3 draws, finite prior sd = 1e3; B is the level-normalised basis);
    the conditional mean equals the profiled least-squares solution and
    marginal = profile - 0.5 logdet(B^T W B) - 0.5 K log sd^2 for a broad prior (sd = 1e4)."""
    from alfa.jax_model import JaxAlfModel
    jm_p = setup["jm"]
    jm_m = JaxAlfModel(setup["grids"], setup["data"], poly_mode="marginal", poly_prior_sd=1e3)
    jm_b = JaxAlfModel(setup["grids"], setup["data"], poly_mode="marginal", poly_prior_sd=1e4)
    for theta in setup["thetas"][:3]:
        params = get_properties(theta, NAMES)
        model = np.asarray(jm_p.model(params))
        # dense brute force
        ll_dense = 0.0
        for r in jm_p.regions:
            m = model[np.asarray(r["midx"])]
            res = np.asarray(r["flux_m"]) - m
            v = (np.asarray(r["err_m"]) * params["jitter"]) ** 2
            B = (m / m.mean())[:, None] * np.asarray(r["A_m"])          # the level-normalised basis
            V = np.diag(v) + 1e3 ** 2 * B @ B.T
            sign, logdet = np.linalg.slogdet(V)
            ll_dense += -0.5 * (res @ np.linalg.solve(V, res) + logdet + len(res) * np.log(2 * np.pi))
        ll_m = float(jm_m.lnlike_dict(params))
        assert abs(ll_m - ll_dense) < 1e-8 * abs(ll_dense), (ll_m, ll_dense)
        # broad prior: conditional mean == least squares; marginal == profile - 0.5 logdet(B^T W B) - 0.5 K log sd^2
        ll_p = float(jm_p.lnlike_dict(params))
        corr = 0.0
        for r, cond in zip(jm_b.regions, jm_b.poly_conditional(params)):
            c_ls = np.asarray(jm_b._poly_lstsq(jm_b.model(params), r))
            assert np.max(np.abs(np.asarray(cond["mean"]) - c_ls) / np.maximum(np.abs(c_ls), 1e-3)) < 1e-5   # the 1e4 prior shrinks by ~1e-6
            m = model[np.asarray(r["midx"])]
            w = 1.0 / (np.asarray(r["err_m"]) * params["jitter"]) ** 2
            B = (m / m.mean())[:, None] * np.asarray(r["A_m"])
            corr += -0.5 * np.linalg.slogdet(B.T @ (B * w[:, None]))[1] - 0.5 * B.shape[1] * np.log(1e4 ** 2)
        ll_b = float(jm_b.lnlike_dict(params))
        assert abs(ll_b - (ll_p + corr)) < 1e-7 * abs(ll_b), (ll_b, ll_p + corr)   # exact only as sd -> inf


def test_marginal_gradient_matches_finite_differences(setup):
    from alfa.jax_model import JaxAlfModel
    jm_m = JaxAlfModel(setup["grids"], setup["data"], poly_mode="marginal")
    f_ll = jm_m.make_lnlike(NAMES)
    g_ll = jax.jit(jax.grad(f_ll))
    lo, hi = setup["lo"], setup["hi"]
    h = 1e-5 * (hi - lo)
    for theta in setup["thetas"][:3]:
        g = np.asarray(g_ll(theta))
        assert np.all(np.isfinite(g))
        for n in SMOOTH:
            i = NAMES.index(n)
            e = np.zeros(len(NAMES)); e[i] = h[i]
            fd = (float(f_ll(theta + e)) - float(f_ll(theta - e))) / (2 * h[i])
            assert abs(g[i] - fd) / max(abs(fd), 1e-8) < 1e-3, (n, g[i], fd)
