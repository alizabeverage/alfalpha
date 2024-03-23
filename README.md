# alfalpha
Elemental abundance fitting code based on the Conroy+18 MILES+IRTF+MIST SSP models and ATLAS response functions

Make sure you have "ALFA_OUT", and "ALFA_INFILES" evironemnt variables set
"ALFA_INFILES": Location of model grids (typically under alfalpha/infiles/)
"ALFA_OUT": Where the output files are saved (can also be manually defined in alfa_template.py)

``` bash
export ALFA_HOME="/Users/alizabeverage/Software/alfalpha/"
export ALFA_INFILES="/Users/alizabeverage/Software/alfalpha/infiles/"
export ALFA_OUT="/Users/alizabeverage/Software/alfalpha/outfiles/"
```

To install the package:
``` bash
python3 -m build
```

You may need to install/upgrad the build module:
``` bash
python3 -m pip install --upgrade build
```


Here is a snippet to test how long it takes to load a single model:

``` python
from alfa.grids import Grids
from alfa.setup_params import setup_params, get_properties
import time
import numpy as np

# load the model grid
g = Grids()

# define the parameters you care about
parameters_to_fit=['velz','sigma','logage','zH','feh','mgh','ch','jitter']

# get the default positions and put them in a dictionary
pos, priors = setup_params(parameters_to_fit)
p = get_properties(pos,parameters_to_fit)

# alter the parameters if you want
p['sigma'] = 300
p['logage'] = np.log10(3.5)
p['zH'] = -0.03
p['feh'] = -0.2

t1 = time.time()
g.get_model(p)
t2 = time.time()

print(t2-t1)
```
