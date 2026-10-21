"""Compare the three treatments of the continuum polynomial on one spectrum (posterior medians, widths):
profile (alfα: best polynomial at every call), sample (coefficients as NUTS parameters), marginal
(coefficients integrated out analytically). Reads the npz files nuts_jax_example.py writes.

    python compare_poly_modes.py [name=test]
"""
import os
import sys

import numpy as np

out = os.environ.get("ALFA_OUT", os.path.expanduser("~/Software/alfalpha/outfiles/"))
name = sys.argv[1] if len(sys.argv) > 1 else "test"
runs = {"profile": f"{name}_nuts_jax.npz", "sample": f"{name}_nuts_jax_polysample_dense.npz",
        "marginal": f"{name}_nuts_jax_polymarginal.npz"}
data = {}
for k, f in runs.items():
    p = os.path.join(out, f)
    if os.path.exists(p):
        d = np.load(p)
        data[k] = (list(d["names"]), d["draws"].reshape(-1, d["draws"].shape[-1]))
ref_names, ref = data["profile"]
keys = list(data)
print(f"{'parameter':16s}" + "".join(f"{k + ' median':>16s}{k + ' sd':>12s}" for k in keys) + "   shift (sd units) vs profile: " + " ".join(k for k in keys[1:]))
for n in ref_names:
    i = ref_names.index(n)
    cols, shifts = "", ""
    for k in keys:
        nn, dr = data[k]
        x = dr[:, nn.index(n)]
        cols += f"{np.median(x):16.3f}{x.std():12.3f}"
        if k != "profile":
            shifts += f"{(np.median(x) - np.median(ref[:, i])) / ref[:, i].std():8.2f}"
    print(f"{n:16s}{cols}   {shifts}")
for k in keys:
    nn, dr = data[k]
    mgfe = dr[:, nn.index("mgh")] - dr[:, nn.index("feh")]
    print(f"[Mg/Fe] {k:9s} {np.median(mgfe):.3f} +/- {mgfe.std():.3f}")
