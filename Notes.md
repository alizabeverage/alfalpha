Run this to see that the variable instrumental resolution works

``` python
from alfa.grids import Grids
import numpy as np

g = Grids(kroupa_shortcut=False)
f = Table.read('/Users/alizabeverage/Research/SUSPENSE/jwst_nirspec_g140m_disp.fits')
res = np.interp(g.ssp.wave[s]/1e4,f['WAVELENGTH'],f['R'])
ires = c/res/2.355

# x = smooth_lsf_fft(g.ssp.wave[s],g.ssp.ssp_grid[s,3,3],outwave = g.ssp.wave[s], 
#                sigma = ires*g.ssp.wave[s]/c)
# plt.plot(g.ssp.wave[s],x)


x = smoothspec(g.ssp.wave[s],g.ssp.ssp_grid[s,3,3],inres=0,resolution=np.median(ires))
plt.plot(g.ssp.wave[s],x)


x = smoothspec(g.ssp.wave[s],g.ssp.ssp_grid[s,3,3],inres=0,resolution=ires*g.ssp.wave[s]/c,
              smoothtype='lsf')
plt.plot(g.ssp.wave[s],x)



## used in practice
g = Grids(kroupa_shortcut=True,inst_res=np.linspace(10,400,s.sum()),inst_res_wave=g.ssp.wave[s])
g1 = Grids(kroupa_shortcut=True,inst_res=200)

plt.plot(g1.ssp.wave[s], g1.ssp.ssp_grid[s,2,3])
plt.plot(g.ssp.wave[s], g.ssp.ssp_grid[s,2,3]+0.2)

```