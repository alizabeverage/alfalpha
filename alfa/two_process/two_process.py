"""Two-process abundance patterns from [Fe/H] and [Mg/Fe].

The two-process model (Weinberg et al. 2019, 2022) decomposes each element's
abundance into a core-collapse and a Type Ia contribution,

    X/H = Acc * qcc(X, [Mg/H])  +  Aia * qia(X, [Mg/H])

where the amplitudes (Acc, Aia) are fixed by the observed [Mg/H] and [Mg/Fe]:
Mg is pure core-collapse, so Acc = 10**[Mg/H], and Aia follows from how far
[Mg/Fe] has dropped below the core-collapse plateau `mgfe_pl`.

Give it a metallicity and an alpha enhancement, get back every other element:

    >>> from two_process import two_process_abundances
    >>> two_process_abundances(feh=-0.1, mgfe=0.25)
    {'C': ..., 'N': ..., 'O': ..., ..., 'Ce': ...}

The q vectors come from compute_qs(), i.e. APOGEE (W22) for most elements,
Roberts+24 subgiant tracks for natal C and N, and GALAH/Griffiths for Ti, Y,
Ba (and Sr, set equal to Y).
"""

import argparse
from collections import OrderedDict
from functools import lru_cache

import numpy as np
from scipy.interpolate import interp1d

from .compute_qs import compute_qs

__all__ = ['two_process_abundances', 'two_process_amplitudes',
           'ELEMENTS', 'ATOMIC_NUMBER']

# Atomic numbers of every element the q tables cover. Hard-coded instead of
# calling mendeleev.element() so this module has no extra dependency; the
# output ordering follows this dict.
ATOMIC_NUMBER = OrderedDict([
    ('C', 6), ('N', 7), ('O', 8), ('Na', 11), ('Mg', 12), ('Al', 13),
    ('Si', 14), ('S', 16), ('K', 19), ('Ca', 20), ('Ti', 22), ('V', 23),
    ('Cr', 24), ('Mn', 25), ('Fe', 26), ('Co', 27), ('Ni', 28), ('Sr', 38),
    ('Y', 39), ('Ba', 56), ('Ce', 58),
])

ELEMENTS = list(ATOMIC_NUMBER)

# 'C+N' is the APOGEE red-giant observable; C and N are reported separately
# (from the Roberts+24 tracks), so it is dropped from the output.
_COMBINED = 'C+N'

# Default core-collapse [Mg/Fe] plateau. 0.3 is the Milky Way value used in
# W22; 0.65 is the higher plateau adopted for the alf reproduction.
DEFAULT_MGFE_PL = 0.65


@lru_cache(maxsize=8)
def _q_interpolators(mgfe_pl):
    """Interpolators for qcc and qia vs [Mg/H], cached per plateau value.

    compute_qs() re-reads the W22 csv files on every call, so cache the result:
    the tables depend only on mgfe_pl.
    """
    qia, qcc, _, _ = compute_qs(mgfe_pl=mgfe_pl)

    elements = [e for e in qia.colnames[1:] if e != _COMBINED]
    unknown = set(elements) - set(ATOMIC_NUMBER)
    if unknown:
        raise ValueError('no atomic number for %s' % sorted(unknown))
    elements = [e for e in ELEMENTS if e in elements]

    def build(table):
        return [interp1d(table['mgh'], table[e], kind='linear',
                         bounds_error=False, fill_value='extrapolate')
                for e in elements]

    return tuple(elements), tuple(build(qcc)), tuple(build(qia))


def two_process_amplitudes(mgh, mgfe, mgfe_pl=DEFAULT_MGFE_PL):
    """Core-collapse and Type Ia amplitudes (Acc, Aia) for a given [Mg/H], [Mg/Fe].

    Aia goes to zero on the plateau (mgfe == mgfe_pl) and grows as [Mg/Fe] falls.
    """
    mgh, mgfe = np.asarray(mgh, dtype=float), np.asarray(mgfe, dtype=float)
    Acc = 10**mgh
    Aia = Acc * (10**-mgfe - 10**-mgfe_pl) / (1 - 10**-mgfe_pl)
    return Acc, Aia


def two_process_abundances(feh, mgfe, mgfe_pl=DEFAULT_MGFE_PL, ref='Fe',
                           elements=None):
    """Predicted abundances of all elements, given [Fe/H] and [Mg/Fe].

    Parameters
    ----------
    feh, mgfe : float or array_like
        [Fe/H] and [Mg/Fe]. Broadcast against each other, so passing arrays
        returns arrays of the same shape for each element.
    mgfe_pl : float
        Core-collapse [Mg/Fe] plateau. Must match the value the model was
        calibrated with -- it sets both the q vectors and the Ia amplitude.
    ref : {'Fe', 'H', 'Mg'}
        Reference element for the returned ratios: [X/Fe], [X/H] or [X/Mg].
    elements : sequence of str, optional
        Restrict/reorder the output. Default is every element the q tables
        cover, sorted by atomic number.

    Returns
    -------
    dict
        {element: value}, in atomic-number order, with the same shape as the
        broadcast inputs (numpy scalars for scalar input). Includes Fe and Mg,
        which reproduce the inputs by construction and are a useful check.

    Notes
    -----
    C and N come from subgiant tracks that are only defined for [Mg/H] > -0.5;
    below that they come back as NaN.
    """
    ref = ref.capitalize() if ref.lower() != 'h' else 'H'
    if ref not in ('Fe', 'H', 'Mg'):
        raise ValueError("ref must be 'Fe', 'H' or 'Mg', got %r" % ref)

    feh, mgfe = np.broadcast_arrays(np.asarray(feh, dtype=float),
                                    np.asarray(mgfe, dtype=float))
    mgh = feh + mgfe

    names, qcc_funcs, qia_funcs = _q_interpolators(mgfe_pl)
    Acc, Aia = two_process_amplitudes(mgh, mgfe, mgfe_pl)

    qcc = np.stack([f(mgh) for f in qcc_funcs])
    qia = np.stack([f(mgh) for f in qia_funcs])

    with np.errstate(divide='ignore', invalid='ignore'):
        xh = np.log10(Acc * qcc + Aia * qia)  # [X/H] for each element

    if ref == 'H':
        values = xh
    elif ref == 'Mg':
        values = xh - mgh  # Mg is pure core-collapse, so [Mg/H] == mgh exactly
    else:
        values = xh - xh[names.index('Fe')]

    out = OrderedDict(zip(names, values))
    if elements is not None:
        missing = [e for e in elements if e not in out]
        if missing:
            raise KeyError('no q vectors for %s' % missing)
        out = OrderedDict((e, out[e]) for e in elements)
    return out


def _main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('feh', type=float, help='[Fe/H]')
    p.add_argument('mgfe', type=float, help='[Mg/Fe]')
    p.add_argument('--mgfe-pl', type=float, default=DEFAULT_MGFE_PL,
                   help='core-collapse [Mg/Fe] plateau (default: %(default)s)')
    p.add_argument('--ref', default='Fe', choices=['Fe', 'H', 'Mg'],
                   help='reference element (default: %(default)s)')
    args = p.parse_args()

    abund = two_process_abundances(args.feh, args.mgfe,
                                   mgfe_pl=args.mgfe_pl, ref=args.ref)
    print('[Fe/H] = %+.3f, [Mg/Fe] = %+.3f, plateau = %.2f'
          % (args.feh, args.mgfe, args.mgfe_pl))
    for e, v in abund.items():
        print('  [%-2s/%-2s] = %+.4f' % (e, args.ref, v))


if __name__ == '__main__':
    _main()
