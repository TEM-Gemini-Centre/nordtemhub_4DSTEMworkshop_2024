# PyPtychoSTEM

A python package performing direct ptychographic reconstructions using 4D STEM data, including the single side band (SSB) and wigner distribution deconvolution (WDD) methods. 

Post collection aberration correction is implemented for the SSB method. 

Parallelism is currently implemented over Qp in probe reciprocal space for SSB. The phase and amplitude for each Qp can be determined independently and thus this work is distributed over the number of workers. 

## Usage

To illustrate the basic use, an aberration free simulated graphene example 4D STEM dataset (~1 GB) can be downloaded [here](https://zenodo.org/record/4476506#.YBLo9NbTVH4).  To process this example dataset you can use the parameters.txt file contained this repository as the parameter file. It provides parameters that should work for the example dataset.

Post collection aberration correction is illustrated in the provided [Jupyter notebook](https://gitlab.com/pyptychostem/pyptychostem/-/blob/master/Examples/WS2/Aberration_correction_SSB_demo.ipynb) using this [WS<sub>2</sub> dataset](https://zenodo.org/record/6477629#.YmKkTFxBxH4) simulated with aberrations. An appropriate paramter file is provided in parameters_WSe2.txt.

To use your own 4D data set (.npy format), modify parameters.txt according to your own parameters. 
In each row, only adapt the right column (path or value) separated by a tab from the first column (keyword).


### Command line
Assuming your envorionment is setup correctly you can run the code from the command line in the directory your data is located

```bash
$ python path/to/pyptychostem/ptycho.py
```
or if you put a link to ptycho.py as ptycho somewhere on your path
```bash
$ ptycho
```
then the program will look for a parameter file (parameters.txt) in the current directory. 

The parameter file tells the program what parameters to use. A minimal parameter file looks like:

```python
file            data_file.npy
method          ssb
aperture        0.03
stepsize        0.15
voltage         60.0
rotation        0.0
```
In the above example we specify to use data_file.npy as the data file to load, the ssb method, an aperture of 30 mrad, a stepsize of 0.15 &#8491; an accelerating voltage of 60 kV and a CBED rotation angle of zero. These should be adjusted to match the data.

Note that if your calibration is wrong your output will likely not be what you expect. 
While the aperture stepsize and voltage are likely obvious, the rotation angle might be less so depending on your microscope.

You can check how well the calibration selects double disk overlaps with the plot_trotters function 
```python
data_4D.plot_trotters(rotation_angle) # test different values of rotation_angle
```

In addition you can use flags in the parameter file to control the output of the program, such as what plots are displayed.
For example setting
```python
plot_trotters	1
```
will use display the result of the plot_trotters function automatically.

### IDE 
Alternatively you can also run the program in your preferred IDE (e.g. Spyder) in essentially the same way. 
Make sure the code is in the appropriate path, and then you can run ptycho.py line by line inside the directory of the data you wish to process.
