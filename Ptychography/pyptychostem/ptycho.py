#!/usr/bin/env python3

from STEM4D import Data4D, SSB, WDD
import numpy as np
import multiprocessing
#import hyperspy.api as hs
#%matplotlib tk
#import hyperspy.extensions
import sys
try:
    import pixstem.api as ps
except:
    print('cannot load pixstem package...')
    print('iCOM not possible')


if len(sys.argv)>1:
    parfile = sys.argv[1]
else:
    parfile ='parameters.txt'
    
#params = np.genfromtxt(parfile,delimiter='\t',dtype=str)
    
par_dictionary = {}

file = open(parfile)

for line in file:
    if line.startswith('##'):
        continue
    split_line = line.rstrip().split('\t')
    print(split_line)

    if len(split_line)!=2:
        continue
    key, value = split_line
    par_dictionary[key] = value
    
plot_4D = int(par_dictionary.get('plot_4D',1))
plot_4D_reciprocal = int(par_dictionary.get('plot_4D_reciprocal',1))
plot_aperture = int(par_dictionary.get('plot_aperture',1))
plot_result = int(par_dictionary.get('plot_result',1))
plot_trotters = int(par_dictionary.get('plot_trotters',1))
plot_ps = int(par_dictionary.get('plot_power_spectrum',1))
method = par_dictionary.get('method','ssb')   


expansion_ratio = float(par_dictionary.get('CBED/BF',-1))
dose = int(par_dictionary.get('dose',-1))   

if  expansion_ratio<1:
    expansion_ratio = None
save = int(par_dictionary.get('save',1))

if method not in ['ssb','wdd','iCOM']:
    print('method not understood')


data_4D = Data4D(parfile)
data_4D.estimate_aperture_size() 
if dose>0:
    data_4D.apply_dose(dose)
#if hyperspy is installed this can be executed
if plot_4D:
    print('plotting 4D via pixstem')
    s = data_4D.plot_4D()
if plot_aperture:
    data_4D.plot_aperture()

data_4D.truncate_ronchigram(expansion_ratio=expansion_ratio) # crops ronchigram to area of interest

    
data_4D.apply_FT()

#if hyperspy is installed this can be executed
if plot_4D_reciprocal:
    data_4D.plot_4D_reciprocal()

#Plot power spectrum 
#data_4D.plot_FT()

#uncomment this to plot the trotters and adjust the rotation angle
if plot_trotters:
    data_4D.plot_trotters(data_4D.rotation_angle_deg,skip=1)# value that fits


if method == 'ssb':
    ssb = SSB(data_4D)
    ssb.run()
    ## the results are accessable via ssb.phase and ssb.amplitude
    if plot_result:
        ssb.plot_result(sample=1)
    if save:
        ssb.save()

if method == 'wdd':
    wdd = WDD(data_4D)
    wdd.run()
    ## the results are accessable via ssb.phase and ssb.amplitude
    #wdd.plot_result(sample=1)
    wdd.save()
    
    





