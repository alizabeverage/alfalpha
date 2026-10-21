"""JAX port of alfα's Conroy18 forward model (``alfa.grids.Grids.get_model``), the
polynomial normalisation (``alfa.polynorm.polynorm``) and the ``alfa_config.lnlike``
likelihood, so the model can be jitted and differentiated.

Usage
-----
    from alfa.grids import Grids
    from alfa.read_data import Data
    from alfa.jax_model import JaxAlfModel

    data = Data("indata/test")
    grids = Grids(inst_res=data.ires, inst_res_wave=data.wave)   # the usual numpy object
    jm = JaxAlfModel(grids, data)                                 # copies the (smoothed) grids to JAX
    names = ("velz", "sigma", "logage", "zH", "feh", "mgh", "jitter")
    lnlike = jm.make_lnlike(names)       # jitted: theta (len(names),) -> scalar
    grad = jax.grad(lnlike)

``jm.model(params)`` is the model spectrum on the data wavelength grid for a params
dict (same keys as ``get_model``); ``jm.model_norm(params)`` adds the polynomial
normalisation; ``jm.lnlike_dict(params)`` is the likelihood.

What is ported, and how it matches the numpy path
--------------------------------------------------
* SSP and element-response grids are taken from an already-built ``Grids`` object,
  so whatever instrumental smoothing ``Grids.__init__`` applied is inherited.
* ``RegularGridInterpolator(method="linear", fill_value=None)`` is reproduced by
  ``multilinear`` below: tensor-product linear interpolation, with LINEAR
  EXTRAPOLATION beyond the outermost nodes (scipy's behaviour for fill_value=None).
* The velocity smoothing reproduces ``smoothing.smooth_vel_fft``: resample to the
  same power-of-two log-lambda grid, multiply the rFFT by the analytic Gaussian
  taper, resample back. One difference: when ``sigma <= model_res`` (100 km/s) the
  numpy code returns the spectrum unsmoothed, while this version still goes through
  the resample-and-back (the taper is 1), so the two differ by the resampling error
  there. Keep ``sigma > model_res`` (the default prior starts at 100).
* Emission lines, the redshift + ``np.interp`` rebin and the Chebyshev polynomial
  normalisation (per fitting region, degree = ceil(width/100 AA) capped at 14, or a
  fixed ``poly_degree``) are ported one to one; the normal equations are solved
  with ``jnp.linalg.solve`` instead of an explicit inverse.
* The likelihood sums over the pixels the numpy ``nansum`` keeps: inside a fitting
  region and unmasked (``mask == 1``; ``Data`` sets ``err`` to NaN elsewhere).

Not ported
----------
* ``two_process_mode``: ``alfa.two_process_mode.compute_deltas`` goes through scipy
  ``interp1d`` tables, Python-level clamps and policy dictionaries. It can be run
  OUTSIDE this model to produce the per-element deltas, which this model accepts
  as ordinary element keys, but then ``feh``/``mgh`` gradients do not flow through
  the two-process mapping.
* The sMILES grids (``alfa.grids_sMILES``).

Smoothness (where the gradient has kinks)
-----------------------------------------
The model is continuous everywhere but its gradient jumps at interpolation nodes:
* ``logage``, ``zH`` (and ``imf1``/``imf2``): at the SSP and response-grid nodes
  (ages 1, 3, 5, 7, 9, 11, 13.5 Gyr for the SSP; 1, 3, 5, 9, 13 Gyr for the
  responses; [Z/H] = -1.5, -1, -0.5, 0, 0.25).
* every element abundance: at its response nodes (0 and +/-0.3 dex; +/-0.15 for
  ``ch``; +/-50 K for ``teff``; -0.3, 0, 0.3, 0.6, 0.9 for ``nah``). Within the
  default priors a chain crosses the 0 node constantly.
* ``velz``, ``velz2``: the ``np.interp`` rebin and the per-line ``max(.., 1 AA)``
  width floor are piecewise linear in the velocities; the kinks are dense but tiny.
``sigma``, ``sigma2``, the ``logemline_*`` amplitudes and ``jitter`` are smooth.

Measured (laptop CPU, 2026-10-20, tests/test_jax_model.py: indata/test, the 20-parameter
set of alfa_config.py, ALFA_NEND=6600 so a 6501-pixel model grid, 3991 data pixels, float64):
    agreement with numpy: normalised spectrum to 2e-13 relative, log-likelihood to 3e-14 (50 draws)
    jax.grad vs central finite differences: relative error < 1e-5 on every parameter but velz
        (7e-4 at worst: the rebin kinks), median 1e-8 (10 draws)
    numpy get_model + polynorm + lnlike : 2.4 ms
    jitted JAX lnlike (warm)            : 0.8 ms
    jitted JAX grad(lnlike) (warm)      : 1.9 ms   (first call compiles: a few seconds)
"""
from __future__ import annotations

import math

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)  # the numpy path is float64; needed to match it to 1e-5
import jax.numpy as jnp  # noqa: E402
from numpy.polynomial.chebyshev import chebvander  # noqa: E402

from alfa.grids import ckms, alpha_elements, wave_emlines, emline_strs, emnormall  # noqa: E402
from alfa.utils import wave_to_x  # noqa: E402

ELEMENT_KEYS = ("feh", "ah", "ch", "nh", "nah", "mgh", "sih", "kh", "cah", "tih", "vh", "crh",
                "mnh", "coh", "nih", "cuh", "srh", "bah", "euh")


# ----------------------------------------------------------------------------- interpolation
def _locate(grid, x):
    """Cell index and fractional position of x on a sorted 1-D grid, scipy-style:
    the index is clipped to [0, n-2] and the fraction is NOT clipped (linear extrapolation)."""
    i = jnp.clip(jnp.searchsorted(grid, x, side="right") - 1, 0, grid.shape[0] - 2)
    t = (x - grid[i]) / (grid[i + 1] - grid[i])
    return i, t


def multilinear(table, grids, xs):
    """Tensor-product linear interpolation of ``table`` (shape n1 x ... x nk x nl) at the point
    ``xs`` (k scalars) on the axes ``grids`` (k sorted 1-D arrays). Returns shape (nl,).
    Equal to scipy's RegularGridInterpolator(method='linear', bounds_error=False, fill_value=None)."""
    for g, x in zip(grids, xs):
        i, t = _locate(g, x)
        table = (1.0 - t) * table[i] + t * table[i + 1]
    return table


# ----------------------------------------------------------------------------- the model
class JaxAlfModel:
    """The alfα forward model + polynomial normalisation + likelihood in JAX.

    Parameters
    ----------
    grids : alfa.grids.Grids
        A built Grids object (its grids are copied; it is not modified).
    data : alfa.read_data.Data
        The spectrum: ``wave``, ``flux``, ``err`` (NaN where masked), ``mask``, ``fitting_regions``.
    poly_degree : int or None
        None = alfα's default (one degree per 100 AA, capped at 14); an int fixes it.
    poly_mode : "profile", "sample" or "marginal"
        "profile" (alfα's behaviour): the polynomial coefficients are set to their weighted
        least-squares optimum at every likelihood call (a profile likelihood).
        "marginal": the coefficients are integrated out analytically. In region r the normalised
        model is model * (1 + A c) = model + B c with B = model[:, None] * A, linear in c, so with
        a Gaussian prior on c (see poly_prior_sd) the fitted pixels are Gaussian with covariance
        diag(err^2 jitter^2) + B Lambda B^T. Its log density is evaluated with the Woodbury
        identity and the matrix determinant lemma (one K x K Cholesky per region, K = degree + 1).
        For a broad prior it equals the profile log-likelihood minus 0.5 logdet(B^T W B) minus
        0.5 K log(poly_prior_sd^2), W = 1 / (err^2 jitter^2): the sampler sees only the stellar
        parameters, and the polynomial's uncertainty is propagated instead of profiled away.
        ``poly_conditional(params)`` returns, per region, the Gaussian conditional of c given the
        data at those parameters (mean (B^T W B + I/sd^2)^-1 B^T W r, covariance the inverse of
        that matrix), so the coefficients are recovered for every posterior draw afterwards.
    poly_prior_sd : float
        Prior sd on every coefficient of the level-normalised polynomial, i.e. on c_k * mean(model)
        with c_k the coefficients of 1 + sum_k c_k T_k (marginal mode). In those units the constant
        term is the data's mean flux (~130 on indata/test), so the default 1e4 is broad there; make
        it much larger than the data's flux level for a prior-independent result. The level
        normalisation makes the marginal likelihood independent of the model's arbitrary flux units
        (see the comment in ``_poly_marginal_region``).
        "sample": the coefficients are ordinary parameters, so they are marginalised by the
        sampler. Each region r with degree d adds ``poly{r}_lognorm`` (log10 of the constant term)
        and ``poly{r}_a1`` ... ``poly{r}_a{d}`` (Chebyshev shape coefficients relative to the
        constant term): poly = 10**lognorm * (1 + sum_k a_k T_k(x)). This is the same family as
        the profiled polynomial 1 + sum_k c_k T_k(x) (10**lognorm = 1 + c_0, a_k = c_k / (1 + c_0)), just
        scaled for sampling. ``poly_names`` lists them; ``poly_init(params)`` gives their
        least-squares values at a point, for starting positions and prior boxes.
    """

    def __init__(self, grids, data, poly_degree=None, poly_mode="profile", poly_prior_sd=1e4):
        if poly_mode not in ("profile", "sample", "marginal"):
            raise ValueError("poly_mode must be 'profile', 'sample' or 'marginal'")
        self.poly_mode = poly_mode
        self.poly_prior_sd = float(poly_prior_sd)
        ssp, rfn = grids.ssp, grids.rfn
        f64 = jnp.float64

        # --- SSP grid: log10 flux, wavelength last -------------------------------------
        self.wave = jnp.asarray(ssp.wave, f64)
        self.imf_vary = bool(ssp.imf_vary)
        if self.imf_vary:
            self.ssp_log = jnp.asarray(np.transpose(np.log10(ssp.ssp_grid), (1, 2, 3, 4, 0)), f64)
            self.ssp_axes = tuple(jnp.asarray(g, f64) for g in
                                  (ssp.imfgrid, ssp.imfgrid, ssp.logagegrid, ssp.logzgrid))
        else:
            self.ssp_log = jnp.asarray(np.transpose(np.log10(ssp.ssp_grid), (1, 2, 0)), f64)
            self.ssp_axes = tuple(jnp.asarray(g, f64) for g in (ssp.logagegrid, ssp.logzgrid))

        # --- element response grids: ratio to solar, (nage, nz, nabund, nl) --------------
        # mirrors Rfn.set_up_interpolators exactly (same nodes, same stacking)
        self.rfn_axes = (jnp.asarray(rfn.logagegrid, f64), jnp.asarray(rfn.logzgrid, f64))
        self.rfn = {}
        for e in ("na", "ca", "fe", "c", "n", "a", "ti", "mg", "si", "teff", "cr", "mn", "ba",
                  "ni", "co", "eu", "sr", "k", "v", "cu"):
            solar = rfn.solar
            if e == "na":
                stack = [rfn.nam, solar, rfn.nap, rfn.nap6, rfn.nap9]
                nodes = [-0.3, 0.0, 0.3, 0.6, 0.9]
            elif e + "p" in rfn.rfn_cols and e + "m" in rfn.rfn_cols:
                stack = [getattr(rfn, e + "m"), solar, getattr(rfn, e + "p")]
                nodes = [-0.3, 0.0, 0.3]
                if e == "c":
                    nodes = [-0.15, 0.0, 0.15]
                if e == "teff":
                    nodes = [-50.0, 0.0, 50.0]
            else:
                stack = [solar, getattr(rfn, e + "p")]
                nodes = [0.0, 0.3]
            p = np.array([s / solar for s in stack])            # (nabund, nl, nage, nz)
            self.rfn[e] = (jnp.asarray(np.transpose(p, (2, 3, 0, 1)), f64), jnp.asarray(nodes, f64))

        # --- velocity smoothing setup (= smoothing.resample_wave + smooth_vel_fft) -------
        self.model_res = float(grids.model_res)
        w = np.asarray(ssp.wave, float)
        nnew = int(2.0 ** np.ceil(np.log2(len(w))))
        w_fft = np.exp(np.linspace(np.log(w.min()), np.log(w.max()), nnew))
        dv = ckms * np.median(np.diff(np.log(w_fft)))
        self.w_fft = jnp.asarray(w_fft, f64)
        self.ss2 = jnp.asarray(np.fft.rfftfreq(nnew, d=dv) ** 2, f64)

        # --- emission lines --------------------------------------------------------------
        self.emlines = {}
        for key in np.unique(emline_strs):
            sel = emline_strs == key
            self.emlines[str(key)] = (jnp.asarray(wave_emlines[sel], f64), jnp.asarray(emnormall[sel], f64))

        # --- data and the polynomial normalisation (all static) --------------------------
        self.outwave = jnp.asarray(data.wave, f64)
        self.flux = jnp.asarray(data.flux, f64)
        err = np.asarray(data.err, float)
        mask = np.asarray(data.mask).astype(bool) & np.isfinite(err)
        self.regions = []
        in_region = np.zeros(len(data.wave), bool)
        for region in data.fitting_regions:
            s = (data.wave >= region[0]) & (data.wave <= region[1])
            deg = min(math.ceil(np.diff(region).item() / 100), 14) if poly_degree is None else int(poly_degree)
            idx = np.where(s)[0]
            m = mask[idx]
            x = wave_to_x(data.wave[idx], m)
            A_full = chebvander(x, deg)
            self.regions.append(dict(
                idx=jnp.asarray(idx), midx=jnp.asarray(idx[m]),
                A_full=jnp.asarray(A_full, f64), A_m=jnp.asarray(A_full[m], f64),
                flux_m=jnp.asarray(data.flux[idx][m], f64), err_m=jnp.asarray(err[idx][m], f64)))
            in_region[idx] = True
        self.poly_names = []
        for r, reg in enumerate(self.regions):
            deg = reg["A_full"].shape[1] - 1
            self.poly_names += [f"poly{r}_lognorm"] + [f"poly{r}_a{k}" for k in range(1, deg + 1)]
        self.poly_names = tuple(self.poly_names)
        self.use = jnp.asarray(mask & in_region)
        self.err_safe = jnp.asarray(np.where(mask & in_region, err, 1.0), f64)
        self.n_use = int((mask & in_region).sum())

    # ------------------------------------------------------------------ pieces
    def ssp_spectrum(self, params):
        if self.imf_vary:
            xs = (params["imf1"], params["imf2"], params["logage"], params["zH"])
        else:
            xs = (params["logage"], params["zH"])
        return 10.0 ** multilinear(self.ssp_log, self.ssp_axes, xs)

    def response(self, element, logage, zH, value):
        table, nodes = self.rfn[element]
        return multilinear(table, self.rfn_axes + (nodes,), (logage, zH, value))

    def add_emlines(self, spec, params):
        for key, val in params.items():
            if "logemline" not in key:
                continue
            lam, norm = self.emlines[key]
            ve = lam / (1.0 + params["velz2"] / ckms)
            lsig = jnp.maximum(ve * params["sigma2"] / ckms, 1.0)                 # min dlam = 1 AA
            prof = norm[:, None] * jnp.exp(-((self.wave[None, :] - ve[:, None]) ** 2) / lsig[:, None] ** 2 / 2.0)
            spec = spec + 10.0 ** val * prof.sum(0)
        return spec

    def smooth(self, spec, sigma):
        """Velocity broadening from the model resolution to ``sigma`` (km/s), FFT on the log grid."""
        s2 = jnp.maximum(sigma ** 2 - self.model_res ** 2, 0.0)
        taper = jnp.exp(-2.0 * (jnp.pi ** 2) * s2 * self.ss2)
        s_fft = jnp.interp(self.w_fft, self.wave, spec)
        conv = jnp.fft.irfft(jnp.fft.rfft(s_fft) * taper)
        return jnp.interp(self.wave, self.w_fft, conv)

    # ------------------------------------------------------------------ the model
    def model_native(self, params):
        """The model spectrum on the native (rest-frame) grid, before the redshift and rebin."""
        spec = self.ssp_spectrum(params)
        logage, zH = params["logage"], params["zH"]
        if "alpha" in params:
            for key in alpha_elements:
                spec = spec * self.response(key[:-1], logage, zH, params["alpha"])
        for key, value in params.items():
            if "alpha" in params and key in alpha_elements:
                continue
            if key in ELEMENT_KEYS:
                spec = spec * self.response(key[:-1], logage, zH, value)
            elif key == "teff":
                spec = spec * self.response("teff", jnp.log10(13.0), zH, value)
        if any("logemline" in k for k in params):
            spec = self.add_emlines(spec, params)
        return self.smooth(spec, params["sigma"])

    def model(self, params):
        """= Grids.get_model(params, outwave=data.wave)."""
        spec = self.model_native(params)
        oneplusz = 1.0 + params["velz"] / ckms
        return jnp.interp(self.outwave, self.wave * oneplusz, spec)

    def _poly_lstsq(self, model, r):
        """Weighted least-squares Chebyshev coefficients c (1 + A c ~ flux / model) for region r."""
        mm = model[r["midx"]]
        y = r["flux_m"] / mm - 1.0
        yvar = (r["err_m"] / mm) ** 2
        A = r["A_m"]
        ATA = A.T @ (A / yvar[:, None])
        return jnp.linalg.solve(ATA, A.T @ (y / yvar))

    def polynorm(self, model):
        """= polynorm.polynorm(data, model): (poly, model * poly), coefficients at their least-squares
        optimum. Pixels outside every region are 0 here (NaN in numpy)."""
        poly = jnp.zeros_like(model)
        for r in self.regions:
            c = self._poly_lstsq(model, r)
            poly = poly.at[r["idx"]].set(1.0 + r["A_full"] @ c)      # later regions overwrite, as in numpy
        return poly, model * poly

    def polynorm_sampled(self, model, params):
        """(poly, model * poly) with the polynomial taken from the params (poly_mode='sample')."""
        poly = jnp.zeros_like(model)
        for r, reg in enumerate(self.regions):
            deg = reg["A_full"].shape[1] - 1
            a = jnp.stack([jnp.asarray(1.0)] + [params[f"poly{r}_a{k}"] for k in range(1, deg + 1)])
            poly_r = 10.0 ** params[f"poly{r}_lognorm"] * (reg["A_full"] @ a)
            poly = poly.at[reg["idx"]].set(poly_r)
        return poly, model * poly

    def poly_init(self, params):
        """Least-squares values of the sampled polynomial parameters at ``params`` (a dict, keys as
        ``poly_names``). In either mode; use it for starting positions and to centre prior boxes."""
        model = self.model(params)
        out = {}
        for r, reg in enumerate(self.regions):
            c = np.asarray(self._poly_lstsq(model, reg))
            c0 = 1.0 + c[0]                      # the profiled polynomial is 1 + sum_k c_k T_k, T_0 = 1
            out[f"poly{r}_lognorm"] = float(np.log10(c0))
            for k in range(1, len(c)):
                out[f"poly{r}_a{k}"] = float(c[k] / c0)
        return out

    def _poly_marginal_region(self, model, r, jitter):
        """For one region: (log marginal likelihood of its fitted pixels, conditional mean of c,
        Cholesky factor of M = B^T W B + I / sd^2). See the class docstring."""
        m = model[r["midx"]]
        res = r["flux_m"] - m
        v = (r["err_m"] * jitter) ** 2
        w = 1.0 / v
        # The model's flux units are arbitrary and its level changes with age and metallicity. The prior
        # is therefore put on the coefficients of the LEVEL-NORMALISED model (m / mean m) * (1 + A c'),
        # c' = c * mean(m): a flat prior in absolute units would favour fainter models by
        # (level)^-K through the determinant term. With this scaling the penalty depends on the
        # continuum's shape only. c_mean / L below are for the ORIGINAL coefficients c.
        scale = jnp.mean(m)
        B = (m / scale)[:, None] * r["A_m"]
        K = B.shape[1]
        M = B.T @ (B * w[:, None]) + jnp.eye(K) / self.poly_prior_sd ** 2
        u = B.T @ (w * res)
        L = jnp.linalg.cholesky(M)
        cp_mean = jax.scipy.linalg.cho_solve((L, True), u)
        quad = jnp.sum(w * res ** 2) - u @ cp_mean
        logdet = jnp.sum(jnp.log(v)) + 2.0 * jnp.sum(jnp.log(jnp.diag(L))) + K * jnp.log(self.poly_prior_sd ** 2)
        n = res.shape[0]
        return -0.5 * (quad + logdet + n * jnp.log(2.0 * jnp.pi)), cp_mean / scale, L * scale

    def lnlike_marginal(self, params):
        """Log-likelihood with the polynomial coefficients integrated out (poly_mode='marginal')."""
        model = self.model(params)
        jitter = params["jitter"] if "jitter" in params else 1.0
        return sum(self._poly_marginal_region(model, r, jitter)[0] for r in self.regions)

    def poly_conditional(self, params):
        """Per region: dict(mean (K,), sd (K,), cov (K, K)) of the coefficients c of 1 + sum_k c_k T_k
        given the data at ``params`` (Gaussian conditional; mode-independent, uses poly_prior_sd)."""
        model = self.model(params)
        jitter = params["jitter"] if "jitter" in params else 1.0
        out = []
        for r in self.regions:
            _, c_mean, L = self._poly_marginal_region(model, r, jitter)
            cov = jax.scipy.linalg.cho_solve((L, True), jnp.eye(L.shape[0]))
            out.append(dict(mean=c_mean, sd=jnp.sqrt(jnp.diag(cov)), cov=cov))
        return out

    def model_norm(self, params):
        model = self.model(params)
        if self.poly_mode == "sample":
            return self.polynorm_sampled(model, params)
        if self.poly_mode == "marginal":                 # the conditional-mean polynomial, for plots
            poly = jnp.zeros_like(model)
            for r, cond in zip(self.regions, self.poly_conditional(params)):
                poly = poly.at[r["idx"]].set(1.0 + r["A_full"] @ cond["mean"])
            return poly, model * poly
        return self.polynorm(model)

    def lnlike_dict(self, params):
        """= alfa_config.lnlike (Gaussian likelihood with the jitter error scaling if 'jitter' is present);
        in poly_mode='marginal' the polynomial-marginalised likelihood instead."""
        if self.poly_mode == "marginal":
            return self.lnlike_marginal(params)
        _, mnorm = self.model_norm(params)
        var = self.err_safe ** 2
        if "jitter" in params:
            var = var * params["jitter"] ** 2
            term = (self.flux - mnorm) ** 2 / var + jnp.log(2.0 * jnp.pi * var)
        else:
            term = (self.flux - mnorm) ** 2 / var
        return -0.5 * jnp.sum(jnp.where(self.use, term, 0.0))

    # ------------------------------------------------------------------ vector interface
    @staticmethod
    def params_from_vector(theta, names):
        return {n: theta[i] for i, n in enumerate(names)}

    def make_lnlike(self, names):
        """A jitted function theta -> log-likelihood for the (static) parameter list ``names``."""
        names = tuple(names)
        return jax.jit(lambda theta: self.lnlike_dict(self.params_from_vector(theta, names)))

    def make_model(self, names):
        """A jitted function theta -> model spectrum on the data grid (no normalisation)."""
        names = tuple(names)
        return jax.jit(lambda theta: self.model(self.params_from_vector(theta, names)))

    def make_poly_conditional(self, names):
        """A jitted function theta -> (means, sds), each (sum of K over regions,): the conditional
        polynomial coefficients at theta, regions concatenated in order."""
        names = tuple(names)

        def f(theta):
            cond = self.poly_conditional(self.params_from_vector(theta, names))
            return jnp.concatenate([c["mean"] for c in cond]), jnp.concatenate([c["sd"] for c in cond])
        return jax.jit(f)

    def make_model_norm(self, names):
        """A jitted function theta -> (poly, normalised model)."""
        names = tuple(names)
        return jax.jit(lambda theta: self.model_norm(self.params_from_vector(theta, names)))
