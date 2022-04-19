# mfi-photosynthetics
MFI - Photosynthetics

To reproduce the results for generating PSFs, one can import the file PSF\_Generator.py and running the following piece of code:

```Python
from PSG_Generator import *

xs = linspace(xmin, xmax, precision)
ys = linspace(ymin, ymax, precision)

psf = PSF(xs, ys, **defaults)
```

Here, `xs` and `ys` are 1-dimensional arrays containing the width and depth of the area on which the PSF is to be calculated, where the units are in meters. The variable `defaults` is a dictionary containing all default constants and dimensions. Alternatively, if desired, one could change this dictionary inside PSF\_Generator.py or give a different dictionary of constants as argument of PSF. The resulting variable psf is a 2-dimensional array containing all values of the point spread function on the given points. 
