# alfalpha
Elemental abundance fitting code based on the Conroy+18 MILES+IRTF+MIST SSP models and ATLAS response functions

Start by making sure you have "ALFA_OUT", and "ALFA_INFILES" environment variables set:

- "ALFA_INFILES": Location of the Conroy+18 model grids (typically under alfalpha/infiles/)

- "ALFA_OUT": Where the output files should be saved

``` bash
export ALFA_OUT="/path/to/repo/alfalpha/outfiles/"
export ALFA_INFILES="/path/to/repo/alfalpha/infiles/"

# if you have the Conroy alf code installed already, this would instead look something like:
export ALFA_INFILES="/path/to/alf/alf-master/infiles/"
```

Then to install the package:
``` bash
python3 -m build
```

You may need to install/upgrade the build module:
``` bash
python3 -m pip install --upgrade build
```

To run the code:

``` bash
python3 alfa_config.py /path/to/data/file
```

For example (there is already a test spectrum in indata/):
``` bash
python3 alfa_config.py indata/test
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
