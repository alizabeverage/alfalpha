"""The 'nuts' sampler path end to end on indata/test: run_nuts -> npz -> post_process -> .sum / .bestspec.
Small warmup/draws so it runs in ~1 min; the posterior quality is tested in test_jax_model.py and the
example scripts, not here. Writes to a temporary ALFA_OUT."""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("ALFA_INFILES", os.path.expanduser("~/Software/alfalpha/infiles/"))
os.environ.setdefault("ALFA_OUT", os.path.expanduser("~/Software/alfalpha/outfiles/"))
os.environ.setdefault("ALFA_NEND", "6600")
pytest.importorskip("jax")
pytest.importorskip("blackjax")

HERE = os.path.dirname(os.path.abspath(__file__))


def test_nuts_sampler_end_to_end(tmp_path):
    from alfa.fitting_info import Info
    from alfa.read_data import Data
    from alfa.grids import Grids
    from alfa.nuts_sampler import run_nuts, load_nuts_samples
    from alfa.post_process import post_process

    fi = Info()
    fi.ALFA_OUT = str(tmp_path) + "/"
    fi.sampler = "nuts"
    fi.model = "Conroy18"
    fi.parameters_to_fit = np.array(["velz", "sigma", "logage", "zH", "feh", "mgh", "jitter"])
    from alfa.setup_params import setup_params
    _, fi.priors = setup_params(fi.parameters_to_fit, model="Conroy18")
    fi.nuts_warmup, fi.nuts_draws, fi.nuts_chains, fi.nuts_starts = 100, 100, 2, 3

    data = Data(os.path.join(HERE, "..", "indata", "test"))
    fi.filename = "test"
    for k in ("wave", "flux", "err", "mask", "ires", "fitting_regions"):
        setattr(fi, "data_" + k, getattr(data, k))
    grids = Grids(inst_res=data.ires, inst_res_wave=data.wave, kroupa_shortcut=True)

    flat = run_nuts(fi, data, grids)
    assert flat.shape == (200, 7) and np.all(np.isfinite(flat))
    assert os.path.exists(tmp_path / "test_nuts.npz")
    d = np.load(tmp_path / "test_nuts.npz")
    assert list(d["names"]) == list(fi.parameters_to_fit)
    assert d["poly_cond_mean"].shape == (200, 6 + 12)
    assert np.allclose(load_nuts_samples(fi), flat)
    # the posterior sits where the long runs put it (loose: 100 draws)
    med = np.median(flat, 0)
    assert abs(med[1] - 274) < 15 and abs(med[5] - med[4] - 0.32) < 0.1

    # refusals
    fi_bad = Info(); fi_bad.__dict__.update(fi.__dict__); fi_bad.model = "sMILES"
    with pytest.raises(ValueError, match="sMILES"):
        run_nuts(fi_bad, data, grids)
    fi_bad.model = "Conroy18"; fi_bad.two_process_mode = True
    with pytest.raises(ValueError, match="two_process"):
        run_nuts(fi_bad, data, grids)

    # post-processing reads the npz and writes the standard outputs
    fi.save_settings()
    post_process(fitting_info=fi, plot_corner=False, plot_bestspec=False)
    assert os.path.exists(tmp_path / "test.sum") and os.path.exists(tmp_path / "test.bestspec")
    header = open(tmp_path / "test.sum").readline()
    assert "[Mg/Fe]50" in header and "mgh50" in header
