"""Vendored two-process abundance model (Weinberg et al. 2019, 2022).

Ported UNCHANGED from ~/Research/alf_repro/ (the phase-5 codebase), except for
one packaging edit: `two_process.py`'s `from compute_qs import compute_qs`
became `from .compute_qs import compute_qs` so it imports as a subpackage.
The q-tables under data/ (W22_*.csv) and compute_qs.py are byte-for-byte copies.

Public API used by the two-process fit mode:
    two_process_abundances(feh, mgfe, mgfe_pl=0.65, ref='H') -> OrderedDict[X] = [X/H]
"""
from .two_process import (
    two_process_abundances,
    two_process_amplitudes,
    ELEMENTS,
    ATOMIC_NUMBER,
    DEFAULT_MGFE_PL,
)

__all__ = [
    "two_process_abundances",
    "two_process_amplitudes",
    "ELEMENTS",
    "ATOMIC_NUMBER",
    "DEFAULT_MGFE_PL",
]
