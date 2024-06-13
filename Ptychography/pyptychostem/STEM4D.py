    #!/usr/bin/env python3
#pyptychostem
import typing
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from pylab import cm
import PIL
from PIL import Image
import os
import pickle
import sys
from scipy.ndimage import center_of_mass
from scipy.ndimage import shift
import multiprocessing
from multiprocessing import Process, Array
#from IPython import get_ipython
import ctypes as c

from SVD_utils import *
from utils import *




class Data4D():
    
    def plot_4D_reciprocal_both(self):
            try:
                import hyperspy.api as hs
            except:
                hs = None
                print('cannot load hyperspy package...')
                print('4D plot not possible')
                return
            c = np.stack((np.log(np.abs(self.data_4D_Reciprocal)+1), (np.angle(self.data_4D_Reciprocal))), axis=0)
            s = hs.signals.Signal2D(c)

            s.axes_manager[0].name = 'frequency x'
            s.axes_manager[0].units = '1/A'
            s.axes_manager[0].scale = 1/self.step_size
            s.axes_manager[1].name = 'frequency y'
            s.axes_manager[1].units = '1/A'
            s.axes_manager[1].scale = 1/self.step_size
            s.axes_manager[2].name = 'Amplitude (0) or phase (1)'
            s.axes_manager[3].name = 'angle x'
            s.axes_manager[3].units = 'mrad'
            s.axes_manager[3].scale = self.aperturesize/self.aperture_radius*1000
            s.axes_manager[3].offset = self.scan_angles_x[0]

            s.axes_manager[4].name = 'angle y'
            s.axes_manager[4].units = 'mrad'
            s.axes_manager[4].scale = self.aperturesize/self.aperture_radius*1000
            s.axes_manager[4].offset = self.scan_angles_y[0]

            s.metadata.General.title = 'FT of 4D data'
            s.plot(cmap='viridis')

            return s    
        
    def __init__(self, parfile, data=None):
        if type(parfile) == str:
            self.init_parameters(parfile)
        elif type(parfile) == dict:
            parameters = parfile
            print(parameters)
            self.init_parameters_from_dict(parameters, data)
        else:
            print("Please specify parameter file path, or input parameters as dictionary and data as an array.")
        self.setup_scanning_parameters()
        #self.center_ronchigrams()
        #self.truncate_ronchigram()
        
    def save_metadata(self):
        dict1 = {'step_size':self.step_size,
                 'step_size_x_reciprocal':self.scan_angle_step_x*1000,
                 'step_size_y_reciprocal':self.scan_angle_step_y*1000,
                 'offset_x_reciprocal': self.scan_angles_x[0],
                 'offset_y_reciprocal': self.scan_angles_y[0]}
        a_file = open(self.path+'data4D_meta.pkl', 'wb')

        pickle.dump(dict1, a_file)
        
        a_file.close()
        
        
    def plot_4D(self,log=False):
        
        #get_ipython().run_line_magic('matplotlib', 'auto') 
        #dict0 = { 'name':'x', 'units':'A', 'scale':self.step_size, 'offset':0}
        #dict1 = { 'name':'y', 'units':'A', 'scale':self.step_size, 'offset':0}
        #dict2 = { 'name':'angle x', 'units':'rad', 'scale':self.scan_angle_step_x, 'offset':self.scan_angles_x[0]}
        #dict3 = { 'name':'angle <', 'units':'rad', 'scale':self.scan_angle_step_y, 'offset':self.scan_angles_y[0]}
        try:
            import hyperspy.api as hs
        except:
            hs = None
            print('cannot load hyperspy package...')
            print('4D plot not possible')
            return

        if log:
            s = hs.signals.Signal2D(np.log(self.data_4D))
        else:
            s = hs.signals.Signal2D(self.data_4D)
        s.axes_manager[0].name = 'x'
        s.axes_manager[0].units = 'A'
        s.axes_manager[0].scale = self.step_size
        s.axes_manager[1].name = 'y'
        s.axes_manager[1].units = 'A'
        s.axes_manager[1].scale = self.step_size
        s.axes_manager[2].name = 'angle x'
        s.axes_manager[2].units = 'mrad'
        s.axes_manager[2].scale = self.aperturesize/self.aperture_radius*1000
        s.axes_manager[2].offset = self.scan_angles_x[0]
        s.axes_manager[3].name = 'angle y'
        s.axes_manager[3].units = 'mrad'
        s.axes_manager[3].scale = self.aperturesize/self.aperture_radius*1000
        s.axes_manager[3].offset = self.scan_angles_y[0]
        s.metadata.General.title = '4D data'

        s.plot(cmap='viridis')
     
        return s
    
    def apply_dose(self, dose):
        dose_perpixel=dose*self.step_size**2
        for i in range (self.data_4D.shape[0]):
            for j in range (self.data_4D.shape[1]):
                self.data_4D[i,j,:,:]=np.random.poisson(self.data_4D[i,j,:,:]/(self.data_4D[i,j,:,:].sum())*dose_perpixel)
    
    def bin_Q(self, bin_factor=1):
        new_shape = (self.data_4D.shape[0], self.data_4D.shape[1],
                     self.data_4D.shape[2]//bin_factor,bin_factor, self.data_4D.shape[3]//bin_factor,bin_factor)
        reshaped_data = self.data_4D.reshape(new_shape)
        binned_data = reshaped_data.mean(axis=-1).mean(axis=-2)
        self.data_4D = binned_data
        print(f'dataset size:\t {self.data_4D.shape}')
         
    
    def init_parameters(self, parfile):
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
        self.file = par_dictionary.get('file','')
        self.path = os.path.abspath(parfile+'/..')+'/'
        os.chdir(self.path)
        print(self.path)
        self.data_4D=np.load(self.file)         
        self.aperturesize = float(par_dictionary.get('aperture',0))
        self.voltage = float(par_dictionary.get('voltage'))
        self.step_size = float(par_dictionary.get('stepsize',1))
        self.rotation_angle_deg  = -float(par_dictionary.get('rotation',0))
        self.rotation_angle = self.rotation_angle_deg/180*np.pi
        self.method  = par_dictionary.get('method','ssb')

        # choose any example data for plotting
        self.workers  = int(par_dictionary.get('workers',1))
        self.threshold = float(par_dictionary.get('threshold',0.3))
        self.wave_len = Wavelength(self.voltage)

    def init_parameters_from_dict(self, parameters, data):
        self.data_4D=data
        self.aperturesize = float(parameters.get('aperture',0))
        self.voltage = float(parameters.get('voltage'))
        self.step_size = float(parameters.get('stepsize',1))
        self.rotation_angle_deg  = -float(parameters.get('rotation',0))
        self.rotation_angle = self.rotation_angle_deg/180*np.pi
        self.method  = parameters.get('method','ssb')

        # choose any example data for plotting
        self.workers  = int(parameters.get('workers',1))
        self.threshold = float(parameters.get('threshold',0.3))
        self.wave_len = Wavelength(self.voltage)

    def setup_scanning_parameters(self):
        self.scan_row = self.data_4D.shape[0]
        self.scan_col = self.data_4D.shape[1]
        self.scan_x_len = self.step_size*(self.scan_col-1)
        self.scan_y_len = self.step_size*(self.scan_row-1)
        self.scan_angle_step_x= self.wave_len/self.scan_x_len
        self.scan_angle_step_y= self.wave_len/self.scan_y_len
        #now set scanning reciprocal space spatial frequency and angles.
        self.scan_angles_x=np.arange(self.scan_col)-np.fix(self.scan_col/2)
        self.scan_angles_y=np.arange(self.scan_row)-np.fix(self.scan_row/2)
        self.scan_angles_x *= self.scan_angle_step_x;
        self.scan_angles_y *= self.scan_angle_step_y;
        #print("angle step in the x direction is: ", self.scan_angle_step_x)
        
    def center_ronchigrams(self):
        Ronchi_mean=np.mean(self.data_4D,(0,1))
        com = center_of_mass(Ronchi_mean)
        for i in range(self.data_4D.shape[0]):
            for j in range (self.data_4D.shape[1]):
                self.data_4D[i,j,:,:] = shift(self.data_4D[i,j,:,:], (np.array(self.data_4D[i,j,:,:].shape)/2-com).astype(int))
                

    def find_rotation(self,sigma = 2):
        icom = iCoM(self)
        icom.run(sigma = sigma)
        self.rotation_angle_deg  = icom.find_rotation()[0]
        self.rotation_angle = self.rotation_angle_deg/180*np.pi
        return self.rotation_angle_deg


    def estimate_aperture_size(self):
        Ronchi_mean=np.mean(self.data_4D,(0,1))
        Ronchi_norm = (Ronchi_mean - np.amin(Ronchi_mean)) / np.ptp(Ronchi_mean)
        self.BFdisk = np.ones(Ronchi_norm.shape) * (Ronchi_norm > self.threshold)
        self.edge = (np.sum(np.abs(np.gradient(self.BFdisk)), axis=0)) > self.threshold 
        xx,yy = np.meshgrid(np.arange(0,Ronchi_mean.shape[1]),np.arange(0,Ronchi_mean.shape[0]))
        self.center_x = np.sum(self.BFdisk*xx/np.sum(self.BFdisk))
        self.center_y = np.sum(self.BFdisk*yy/np.sum(self.BFdisk))
        self.aperture_radius = np.average(np.sqrt((xx - self.center_x) ** 2 + (yy - self.center_y) ** 2)[self.edge])
        self.calibration = self.aperturesize/self.aperture_radius
        
        if (np.count_nonzero(self.BFdisk)==0):
            print('Warning, no BF disk detected, you might decrease threshold')
            return 1
    

    ## private    
    def _init_figure(self, rows, cols,figsize,num=None):
        fig,ax =plt.subplots(rows,cols,figsize=figsize,num=num)
        return fig, ax
        
        


    def plot_aperture(self):
        aperture_round  = circle(self.center_x, self.center_y,self.aperture_radius)
        Ronchi_mean=np.mean(self.data_4D,(0,1))
        
        fig, ax = self._init_figure(1,3, (17,4),num = 'Aperture')
        im=ax[0].imshow(Ronchi_mean,cmap=cm.nipy_spectral)
        fig.colorbar(im, ax=ax[0])        
        ax[0].set_title('Averaged ronchigram')
        im2=ax[1].imshow(self.BFdisk,cmap=cm.nipy_spectral)
        ax[1].set_title('Bright field disk')
        im3=ax[2].imshow(self.edge)
        ax[2].plot(aperture_round[0],aperture_round[1], linewidth=10) 
        ax[2].set_title('Aperture edge')
        
        plt.show()


# now we truncate and shift the ronchigram in order to use it in the Trotter generation.
# The center of the new ronchigram should be close to the center pixel

    def truncate_ronchigram(self, expansion_ratio = None):
        if  expansion_ratio == None:
            self.x_hwindow_size  = int(self.data_4D.shape[3]/2)
            self.y_hwindow_size  = int(self.data_4D.shape[2]/2)
            window_start_x = 0 
            window_start_y = 0 
            self.data_4D_trunc=self.data_4D

            Ronchi_angle_step=self.aperturesize/self.aperture_radius
            self.Ronchi_angles_x=(np.arange(self.data_4D.shape[3])-self.center_x)*Ronchi_angle_step
            self.Ronchi_angles_y=(np.arange(self.data_4D.shape[2])-self.center_y)*Ronchi_angle_step  
            #self.calibration = Ronchi_angle_step = self.aperturesize/self.data_4D_trunc.shape[3]*2


        else:           
            self.x_hwindow_size = int(np.fix(self.aperture_radius*expansion_ratio))
            self.y_hwindow_size = int(np.fix(self.aperture_radius*expansion_ratio))       
            window_start_x = int(np.fix(self.center_x)) - self.x_hwindow_size
            window_start_y = int(np.fix(self.center_y)) - self.y_hwindow_size
            new_center_x = self.center_x - window_start_x
            new_center_y = self.center_y - window_start_y
            self.center_x = new_center_x
            self.center_y = new_center_y

            Ronchi_angle_step=self.aperturesize/self.aperture_radius
            self.Ronchi_angles_x=(np.arange(self.x_hwindow_size*2+1)-new_center_x)*Ronchi_angle_step
            self.Ronchi_angles_y=(np.arange(self.y_hwindow_size*2+1)-new_center_y)*Ronchi_angle_step
        if  expansion_ratio != None:
            self.data_4D_trunc=np.zeros((self.scan_row, self.scan_col,self.y_hwindow_size*2+1, self.x_hwindow_size*2+1), dtype = self.data_4D.dtype)
            for i in range(self.data_4D.shape[0]):
                for j in range (self.data_4D.shape[1]):
                    ronchi = self.data_4D[i,j,:,:]
                    ronchi[self.BFdisk==0] =0
                    self.data_4D_trunc[i,j,:,:]=ronchi[window_start_y:window_start_y+ self.y_hwindow_size*2+1,
                                                         window_start_x:window_start_x+ self.x_hwindow_size*2+1]
        self.center_x = int(self.data_4D_trunc.shape[3]/2)
        self.center_y = int(self.data_4D_trunc.shape[2]/2)

    # Now we start to Fourier transform the Ronchigram along the probe position and show the trotters.  
    def apply_FT(self):
        self.data_4D_Reciprocal=np.zeros(self.data_4D_trunc.shape,dtype='complex64') 
        for i in range (self.data_4D_trunc.shape[2]): 
            for j in range (self.data_4D_trunc.shape[3]): 
                self.data_4D_Reciprocal[:,:,i,j]=FFT_2D(self.data_4D_trunc[:,:,i,j]) 
 
        self.power_spectra =np.zeros((self.data_4D_trunc.shape[0],self.data_4D_trunc.shape[1])) 
        
        for i in range (self.data_4D_trunc.shape[0]): 
            for j in range (self.data_4D_trunc.shape[1]): 
                g=self.data_4D_Reciprocal[i,j,:,:] 
                self.power_spectra[i,j]=np.sum(g*np.conjugate(g)).real 
                

     
    def remove_zero_frequency(self):
        self.power_spectra[int(self.power_spectra.shape[0]/2)-1:int(self.power_spectra.shape[0]/2)+2,
                           int(self.power_spectra.shape[1]/2)-1:int(self.power_spectra.shape[1]/2)+2] = np.min(self.power_spectra)
                
    def mask_ps(self,d=0,region='lower'):
        if region=='lower':
            self.power_spectra[int(self.power_spectra.shape[1]/2)-int(d):,:] = 0
        
        
    def plot_4D_reciprocal(self,signal = 'amplitude',log=True):
        try:
            import hyperspy.api as hs
        except:
            hs = None
            print('cannot load hyperspy package...')
            print('4D plot not possible')
            return
        #get_ipython().run_line_magic('matplotlib', 'auto') 
        #dict0 = { 'name':'x', 'units':'A', 'scale':self.step_size, 'offset':0}
        #dict1 = { 'name':'y', 'units':'A', 'scale':self.step_size, 'offset':0}
        #dict2 = { 'name':'angle x', 'units':'rad', 'scale':self.scan_angle_step_x, 'offset':self.scan_angles_x[0]}
        #dict3 = { 'name':'angle <', 'units':'rad', 'scale':self.scan_angle_step_y, 'offset':self.scan_angles_y[0]}
        if signal == 'amplitude':
            if log:
                s = hs.signals.Signal2D(np.log(np.abs(self.data_4D_Reciprocal)))
            else:
                s = hs.signals.Signal2D((np.abs(self.data_4D_Reciprocal))) 
        elif signal == 'phase':
            s = hs.signals.Signal2D(np.angle(self.data_4D_Reciprocal))
        else:
            print('signal keyword not understood')
            return
        s.axes_manager[0].name = 'frequency x'
        s.axes_manager[0].units = '1/A'
        s.axes_manager[0].scale = 1/self.step_size
        s.axes_manager[1].name = 'frequency y'
        s.axes_manager[1].units = '1/A'
        s.axes_manager[1].scale = 1/self.step_size
        s.axes_manager[2].name = 'angle x'
        s.axes_manager[2].units = 'mrad'
        s.axes_manager[2].scale = self.aperturesize/self.aperture_radius*1000
        s.axes_manager[2].offset = self.scan_angles_x[0]
        s.axes_manager[3].name = 'angle y'
        s.axes_manager[3].units = 'mrad'
        s.axes_manager[3].scale = self.aperturesize/self.aperture_radius*1000
        s.axes_manager[3].offset = self.scan_angles_y[0]
        s.metadata.General.title = 'FT of 4D data'
        s.plot(cmap='bwr')
        #circle1 = plt.Circle((int(self.data_4D_trunc.shape[3]/2), int(self.data_4D_trunc.shape[2]/2)), 
         #                    self.aperture_radius, color='r',fill=None)
        #plt.gca().add_patch(circle1)
        return s
        
        
        ## will be an own function
    def plot_FT(self):
        fig, ax = self._init_figure(1,2,(12,4),num='Fourier Transform')
        im=ax[0].imshow(self.power_spectra) 
        fig.colorbar(im, ax=ax[0])        
        ax[0].set_title('Power Spectrum') 
        im2=ax[1].imshow(np.log10(1+self.power_spectra)) 
        fig.colorbar(im2, ax=ax[1])        
        ax[1].set_title('Power Spectrum in logrithm') 
        plt.show()

    def plot_trotter(self, frame):
        row,col = frame
        fig, ax = self._init_figure(1,2,figsize=(12,4),num ='Trotters´')
        im=ax[0].imshow(np.abs(self.data_4D_Reciprocal[row,col]))
        fig.colorbar(im, ax=ax[0])        
        im2=ax[1].imshow(np.angle(self.data_4D_Reciprocal[row,col]))
        fig.colorbar(im2, ax=ax[1])        
        ax[0].set_title('amplitude')
        ax[1].set_title('phase')
        plt.show()
                
              
    
    
    def plot_trotters(self,rotation_angle,selected_frames = None, plot_constrains = True, skip = 0):       
        rotation_angle = -rotation_angle/180*np.pi
        self.rotation_angle = rotation_angle
        ind=np.unravel_index(np.argsort(self.power_spectra, axis=None), self.power_spectra.shape) # rank from low to high 
        
        #example_frames = peaks[:-1]
        #    example_frame = ex
        #reverse_ind=(ind[0][::-1],ind[1][::-1]) # rank from high to low
        fig, ax = self._init_figure(3,3,(10,10),num='Calibrate rotation phase')
        fig2, ax2 = self._init_figure(3,3,(10,10),num='Calibrate rotation amplitude')
        #print('fft peaks at x '+str(ind[1][-2::-10]))
        #print('fft peaks at y '+str(ind[0][-2::-10]))
        self.selected_frames = []
        for i in range(9):
            #take 9 bright spots in fft as a example
            frame_x_idx=ind[1][-i-1-skip]
            frame_y_idx=ind[0][-i-1-skip]
            if selected_frames != None:
                if len(selected_frames)>i:
                    frame_x_idx = selected_frames[i][1]
                    frame_y_idx = selected_frames[i][0]

            self.selected_frames.append([frame_y_idx,frame_x_idx])
            scan_x_angle = self.scan_angles_x[frame_x_idx]*np.cos(rotation_angle) - self.scan_angles_y[frame_y_idx]*np.sin(rotation_angle)
            scan_y_angle = self.scan_angles_x[frame_x_idx]*np.sin(rotation_angle) + self.scan_angles_y[frame_y_idx]*np.cos(rotation_angle)
            #Here we need to consider the coordinate difference in imshow. The scan y angle should be opposite.       
            round1=circle(scan_x_angle, -scan_y_angle, self.aperturesize)
            round2=circle(-scan_x_angle, scan_y_angle, self.aperturesize)
            round3=circle(0,0, self.aperturesize)
            im=ax[int(i/3),i%3].imshow(np.angle(self.data_4D_Reciprocal[frame_y_idx,frame_x_idx,:,:]),
                              extent=[self.Ronchi_angles_x.min(),self.Ronchi_angles_x.max(),
                                      self.Ronchi_angles_y.min(),self.Ronchi_angles_y.max()])
            
            im2=ax2[int(i/3),i%3].imshow(np.abs(self.data_4D_Reciprocal[frame_y_idx,frame_x_idx,:,:]),
                              extent=[self.Ronchi_angles_x.min(),self.Ronchi_angles_x.max(),
                                      self.Ronchi_angles_y.min(),self.Ronchi_angles_y.max()])
            #fig.colorbar(im, ax=ax[int(i/3),i%3])    
            if plot_constrains:
                ax[int(i/3),i%3].plot(round1[0],round1[1], linewidth=5, color = 'red')
                ax[int(i/3),i%3].plot(round2[0],round2[1], linewidth=5, color = 'blue')
                ax[int(i/3),i%3].plot(round3[0],round3[1], linewidth=5, color = 'green')
                ax[int(i/3),i%3].plot(scan_x_angle,-scan_y_angle,'r*')
                ax[int(i/3),i%3].plot(-scan_x_angle, scan_y_angle,'b*')
            ax[int(i/3),i%3].set_title('index '+str(frame_y_idx)+' '+str( frame_x_idx))
                
            
            #fig2.colorbar(im, ax=ax2[int(i/3),i%3]) 
            if plot_constrains:
                ax2[int(i/3),i%3].plot(round1[0],round1[1], linewidth=5, color = 'red')
                ax2[int(i/3),i%3].plot(round2[0],round2[1], linewidth=5, color = 'blue')
                ax2[int(i/3),i%3].plot(round3[0],round3[1], linewidth=5, color = 'green')
                ax2[int(i/3),i%3].plot(scan_x_angle,-scan_y_angle,'r*')
                ax2[int(i/3),i%3].plot(-scan_x_angle, scan_y_angle,'b*')
            ax2[int(i/3),i%3].set_title('index '+str(frame_y_idx)+' '+str( frame_x_idx))
        #fig.tight_layout()
        #fig2.tight_layout()
        plt.show()

        
    def plot_higher_order_trotters(self,rotation_angle,selected_frames = None, order = 1,log = True, plot_constrains = True, skip = 0):       
        rotation_angle = -rotation_angle/180*np.pi
        self.rotation_angle = rotation_angle
        ind=np.unravel_index(np.argsort(self.power_spectra, axis=None), self.power_spectra.shape) # rank from low to high 
        self.fig_trotters_phase, ax = self._init_figure(3,3,(10,10),num='Calibrate rotation phase')
        self.fig_trotters_amp, ax2 = self._init_figure(3,3,(10,10),num='Calibrate rotation amplitude')
        self.selected_frames = []
        for i in range(9):
            #take 9 bright spots in fft as a example
            frame_x_idx=ind[1][-i-1-skip]
            frame_y_idx=ind[0][-i-1-skip]
            if selected_frames != None:
                if len(selected_frames)>i:
                    frame_x_idx = selected_frames[i][1]
                    frame_y_idx = selected_frames[i][0]
            self.selected_frames.append([frame_y_idx,frame_x_idx])
            scan_x_angle = self.scan_angles_x[frame_x_idx]*np.cos(rotation_angle) - self.scan_angles_y[frame_y_idx]*np.sin(rotation_angle)
            scan_y_angle = self.scan_angles_x[frame_x_idx]*np.sin(rotation_angle) + self.scan_angles_y[frame_y_idx]*np.cos(rotation_angle)
            #Here we need to consider the coordinate difference in imshow. The scan y angle should be opposite.       
            round1=circle(scan_x_angle, -scan_y_angle, self.aperturesize)
            round2=circle(-scan_x_angle, scan_y_angle, self.aperturesize)
            round3=circle(0,0, self.aperturesize)
            im=ax[int(i/3),i%3].imshow(np.angle(self.data_4D_Reciprocal[frame_y_idx,frame_x_idx,:,:]),
                              extent=[self.Ronchi_angles_x.min(),self.Ronchi_angles_x.max(),
                                      self.Ronchi_angles_y.min(),self.Ronchi_angles_y.max()])
            if log:
                im2=ax2[int(i/3),i%3].imshow(np.log(np.abs(self.data_4D_Reciprocal[frame_y_idx,frame_x_idx,:,:])),
                                  extent=[self.Ronchi_angles_x.min(),self.Ronchi_angles_x.max(),
                                      self.Ronchi_angles_y.min(),self.Ronchi_angles_y.max()])    
            else:
                im2=ax2[int(i/3),i%3].imshow(np.abs(self.data_4D_Reciprocal[frame_y_idx,frame_x_idx,:,:]),
                              extent=[self.Ronchi_angles_x.min(),self.Ronchi_angles_x.max(),
                                      self.Ronchi_angles_y.min(),self.Ronchi_angles_y.max()])
            self.fig_trotters_phase.colorbar(im, ax=ax[int(i/3),i%3])    
            if plot_constrains:
                ax[int(i/3),i%3].plot(round1[0],round1[1], linewidth=2, color = 'red')
                ax[int(i/3),i%3].plot(round2[0],round2[1], linewidth=2, color = 'blue')
                for j in range(1,order):
                    round1b=circle(scan_x_angle*(j+1), -scan_y_angle*(j+1), self.aperturesize)
                    round2b=circle(-scan_x_angle*(j+1), scan_y_angle*(j+1), self.aperturesize)
                    ax[int(i/3),i%3].plot(round1b[0],round1b[1], linewidth=2, color = 'red')
                    ax[int(i/3),i%3].plot(round2b[0],round2b[1], linewidth=2, color = 'blue')
                ax[int(i/3),i%3].plot(round3[0],round3[1], linewidth=2, color = 'green')
                ax[int(i/3),i%3].plot(scan_x_angle,-scan_y_angle,'r*')
                ax[int(i/3),i%3].plot(-scan_x_angle, scan_y_angle,'b*')
            ax[int(i/3),i%3].set_title('index '+str(frame_y_idx)+' '+str( frame_x_idx))
                
            
            self.fig_trotters_amp.colorbar(im, ax=ax2[int(i/3),i%3]) 
            if plot_constrains:
                ax2[int(i/3),i%3].plot(round1[0],round1[1], linewidth=2, color = 'red')
                ax2[int(i/3),i%3].plot(round2[0],round2[1], linewidth=2, color = 'blue')
                ax2[int(i/3),i%3].plot(round3[0],round3[1], linewidth=2, color = 'green')
                for j in range(1,order):
                    round1b=circle(scan_x_angle*(j+1), -scan_y_angle*(j+1), self.aperturesize)
                    round2b=circle(-scan_x_angle*(j+1), scan_y_angle*(j+1), self.aperturesize)
                    ax2[int(i/3),i%3].plot(round1b[0],round1b[1], linewidth=2, color = 'red')
                    ax2[int(i/3),i%3].plot(round2b[0],round2b[1], linewidth=2, color = 'blue')
                ax2[int(i/3),i%3].plot(scan_x_angle,-scan_y_angle,'r*')
                ax2[int(i/3),i%3].plot(-scan_x_angle, scan_y_angle,'b*')
            ax2[int(i/3),i%3].set_title('index '+str(frame_y_idx)+' '+str( frame_x_idx))


    

class SVD_AC():
    def __init__(self, data4D, trotters_nb=8):
        self.data4D = data4D
        self.trotters_nb = trotters_nb
        #self.Ronchi_xx,self.Ronchi_yy=np.meshgrid(self.data4D.Ronchi_angles_x,self.data4D.Ronchi_angles_y)
        self.theta_x=(np.arange(self.data4D.data_4D_trunc.shape[3])-self.data4D.center_x)*self.data4D.calibration
        self.theta_y=(np.arange(self.data4D.data_4D_trunc.shape[2])-self.data4D.center_y)*self.data4D.calibration
        self.theta_xx,self.theta_yy=np.meshgrid(self.theta_x,self.theta_y)
 
        
        
    def find_trotters(self,skip=1):
        ind=np.unravel_index(np.argsort(self.data4D.power_spectra, axis=None), self.data4D.power_spectra.shape) 
        self.data4D.selected_frames = []
        #take 9 bright spots in fft as a default
        j = 0
        for i in range(self.trotters_nb):            
            frame_x_idx=ind[1][-i-1-skip+j]
            frame_y_idx=ind[0][-i-1-skip+j]
            
            scan_x_angle = self.data4D.scan_angles_x[frame_x_idx]*np.cos(self.data4D.rotation_angle) - self.data4D.scan_angles_y[frame_y_idx]*np.sin(self.data4D.rotation_angle)
            scan_y_angle = self.data4D.scan_angles_x[frame_x_idx]*np.sin(self.data4D.rotation_angle) + self.data4D.scan_angles_y[frame_y_idx]*np.cos(self.data4D.rotation_angle)
                        
            while  scan_x_angle**2+scan_y_angle**2>=self.data4D.aperturesize**2:
                j+=1
                frame_x_idx=ind[1][-i-1-skip+j]
                frame_y_idx=ind[0][-i-1-skip+j]

                scan_x_angle = self.data4D.scan_angles_x[frame_x_idx]*np.cos(self.data4D.rotation_angle) - self.data4D.scan_angles_y[frame_y_idx]*np.sin(self.data4D.rotation_angle)
                scan_y_angle = self.data4D.scan_angles_x[frame_x_idx]*np.sin(self.data4D.rotation_angle) + self.data4D.scan_angles_y[frame_y_idx]*np.cos(self.data4D.rotation_angle)
                if j>=100:
                    print('cannot find more trotters than '+str(len(self.data4D.selected_frames)))
                    return
                
            self.data4D.selected_frames.append([frame_y_idx,frame_x_idx])

        
        print(str(len(self.data4D.selected_frames)) +' trotters found')
        
    def calc_aberrationfunction(self):
        func_aberr   = Aberr_func(self.theta_x,self.theta_y,self.aberration_coeffs,5)
        self.func_transfer= np.exp(-1j*2*np.pi/(self.data4D.wave_len*1e-10)*func_aberr)
        
    def calc_aperturefunction(self):
        theta=np.sqrt(self.theta_xx**2+self.theta_yy**2)
        func_objApt=np.ones(theta.shape)
        func_objApt[theta>self.data4D.aperturesize]=0
        dose=np.sum(self.data4D.data_4D)/(self.data4D.data_4D.shape[0]*self.data4D.data_4D.shape[1])
        scaling=np.sqrt(dose/np.sum(func_objApt))
        self.func_objApt=scaling*func_objApt

    def calc_probefunction(self):  
        self.calc_aberrationfunction()
        self.calc_aperturefunction()
        A=self.func_objApt*self.func_transfer
        self.probe=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A)))

    def plot_probe(self):
        fig, ax = self.data4D._init_figure(1,2, (10,4),num = 'probe function')
        im = ax[0].imshow(np.angle(self.probe))
        ax[0].set_title('probe phase')
        ax[0].set_frame_on(False)
        ax[0].xaxis.set_ticks([])
        ax[0].yaxis.set_ticks([])
        im = ax[1].imshow(np.abs(self.probe))
        ax[1].set_title('probe amplitude')
        ax[1].set_frame_on(False)
        ax[1].xaxis.set_ticks([])
        ax[1].yaxis.set_ticks([])
        plt.show()
    
    def run_svd(self,iterations = 50, coefficients = [],cf =0.5,order = 3,aberr_order_seq=[]):
        if len(coefficients)!=25:
            coefficients = np.zeros(25)
        aberr_input=-coefficients
        if len(aberr_order_seq)!=iterations or  not (np.all((aberr_order_seq >= 1) & (aberr_order_seq <= 5))):
            aberr_order_seq=np.ones(iterations).astype(int)
            if order==2:
                aberr_order_seq[::2] = 2
            elif order==3:
                aberr_order_seq[1::3] = 2
                aberr_order_seq[2::3] = 3
            elif order==4:
                aberr_order_seq[1::4] = 2
                aberr_order_seq[2::4] = 3
                aberr_order_seq[3::4] = 4
            elif order==5:
                aberr_order_seq[1::5] = 2
                aberr_order_seq[2::5] = 3
                aberr_order_seq[3::5] = 4
                aberr_order_seq[4::5] = 5
        else:
            print('using custom sequence')
        for itt in range(iterations):
            print('Process',round((itt+1)/iterations*100),'%',end='\r')
            """
            Nitt: numbers of iteration
            """
            aberr_order=aberr_order_seq[itt]
            # Prepare the matrix
            j=-1
            for i in range(self.trotters_nb):
                j=j+1
                yy=self.data4D.selected_frames[i][0]
                xx=self.data4D.selected_frames[i][1]
                    
                g=self.data4D.data_4D_Reciprocal[yy,xx,:,:].copy()

                scan_x_angle = self.data4D.scan_angles_x[xx]*np.cos(self.data4D.rotation_angle) - self.data4D.scan_angles_y[yy]*np.sin(self.data4D.rotation_angle)
                scan_y_angle = self.data4D.scan_angles_x[xx]*np.sin(self.data4D.rotation_angle) + self.data4D.scan_angles_y[yy]*np.cos(self.data4D.rotation_angle)
                RTrotter,LTrotter,Trotter= mask_trotter(g,self.theta_x,self.theta_y, scan_x_angle,scan_y_angle,self.data4D.aperturesize)
                    
                Trotter_correction=Phase_compen(g,aberr_input,self.theta_x,self.theta_y,scan_x_angle,scan_y_angle,
                                                self.data4D.aperturesize,self.data4D.wave_len)


                ST_phase=np.angle(Trotter_correction)
                ST_phase[Trotter[0]==0]=0
                ST_phase_unwrap=unwrap_phase(ST_phase)
                ST_phase_unwrap[Trotter[0]==0]=0
                ST_phase_unwrap=unwrap_phase(ST_phase_unwrap)
                ST_phase_unwrap[Trotter[0]==0]=0

                if j==0:
                    Matrix,phase=Matrix_prep(LTrotter, ST_phase_unwrap, self.theta_x,self.theta_y, scan_x_angle,scan_y_angle, aberr_order,self.trotters_nb,j)
                else:
                    Matrix_buffer,phase_buffer=Matrix_prep(LTrotter, ST_phase_unwrap, self.theta_x,self.theta_y, scan_x_angle,scan_y_angle, aberr_order,self.trotters_nb,j)

                    Matrix=np.vstack((Matrix,Matrix_buffer))
                    phase=np.hstack((phase,phase_buffer))        

            coefficients_buffer=-Coefficients_cal(Matrix,phase,self.data4D.wave_len)
            aberr_delta=np.zeros(coefficients_buffer.shape[0],dtype=float)

            if aberr_order==1:
                aberr_delta[:3]=coefficients_buffer[:3]
            elif aberr_order==2:
                aberr_delta[:7]=coefficients_buffer[:7]
            elif aberr_order==3:
                aberr_delta[:12]=coefficients_buffer[:12]
            elif aberr_order==4:
                aberr_delta[:18]=coefficients_buffer[:18]
            elif aberr_order==5:
                aberr_delta[:25]=coefficients_buffer[:25]


           
            aberr_input=aberr_input+aberr_delta*cf
        self.aberration_coeffs  =   aberr_input   
        return aberr_input
        
         

    def plot_corrected_trotters(self,frames,aberrations):
        from scipy.ndimage import rotate
        self.corrected_trotters = []
        for i in range(len(frames)):
            single_trotter=self.data4D.data_4D_Reciprocal[frames[i][0],frames[i][1]].copy()
            
            scan_x_angle = self.data4D.scan_angles_x[frames[i][1]]*np.cos(self.data4D.rotation_angle) - self.data4D.scan_angles_y[frames[i][0]]*np.sin(self.data4D.rotation_angle)
            scan_y_angle = self.data4D.scan_angles_x[frames[i][1]]*np.sin(self.data4D.rotation_angle) + self.data4D.scan_angles_y[frames[i][0]]*np.cos(self.data4D.rotation_angle)
            
            Trotter_correction,AA,single_trotter=Phase_compen(single_trotter,aberrations,self.theta_x,self.theta_y,scan_x_angle,scan_y_angle,
                                                self.data4D.aperturesize,self.data4D.wave_len, return_AA = True)
      
            self.corrected_trotters.append(
                [np.angle(single_trotter),
                 np.angle(AA),
                 np.angle(Trotter_correction)])
        
        fig, ax = self.data4D._init_figure(3,len(self.corrected_trotters), (len(self.corrected_trotters)*3,5),
                                    num = 'corrected trotters')
        for i in range(len(self.corrected_trotters)):
            for j in range(3):
                im = ax[j,i].imshow(self.corrected_trotters[i][j])
                fig.colorbar(im,ax=ax[j,i])
                
            ax[0,i].set_title('index '+str(frames[i][0])+' '+str( frames[i][1]))

        ax[0,0].set_ylabel('uncorrected')
        ax[1,0].set_ylabel('calculated')
        ax[2,0].set_ylabel('corrected')
        plt.show()
        
 
      
    def print_aberration_coefficients(self,coeffs):
        print('C10  = ',str(round(coeffs[0]*1e9,3)),' nm')
        print('C12a = ',str(round(coeffs[1]*1e9,3)),' nm')
        print('C12b = ',str(round(coeffs[2]*1e9,3)),' nm')
        print('C21a = ',str(round(coeffs[5]*1e9,3)),' nm')
        print('C21b = ',str(round(coeffs[6]*1e9,3)),' nm')
        print('C23a = ',str(round(coeffs[3]*1e9,3)),' nm')
        print('C23b = ',str(round(coeffs[4]*1e9,3)),' nm')
        print('C30  = ',str(round(coeffs[7]*1e6,3)),' um')
        print('C32a = ',str(round(coeffs[10]*1e6,3)),' um')
        print('C32b = ',str(round(coeffs[11]*1e6,3)),' um')
        print('C34a = ',str(round(coeffs[8]*1e6,3)),' um')
        print('C34b = ',str(round(coeffs[9]*1e6,3)),' um')
        print('C41a = ',str(round(coeffs[16]*1e3,3)),' mm')
        print('C41b = ',str(round(coeffs[17]*1e3,3)),' mm')
        print('C43a = ',str(round(coeffs[14]*1e3,3)),' mm')
        print('C43b = ',str(round(coeffs[15]*1e3,3)),' mm')
        print('C45a = ',str(round(coeffs[12]*1e3,3)),' mm')
        print('C45b = ',str(round(coeffs[13]*1e3,3)),' mm')
        print('C50  = ',str(round(coeffs[18]*1e3,3)),' mm')
        print('C52a = ',str(round(coeffs[23]*1e3,3)),' mm')
        print('C52b = ',str(round(coeffs[24]*1e3,3)),' mm')
        print('C54a = ',str(round(coeffs[21]*1e3,3)),' mm')
        print('C54b = ',str(round(coeffs[22]*1e3,3)),' mm')
        print('C56a = ',str(round(coeffs[19]*1e3,3)),' mm')
        print('C56b = ',str(round(coeffs[20]*1e3,3)),' mm')
  
        


class SSB():
    
    def __init__(self, data4D):
        self.data4D = data4D

    def integrate_trotter(self,LTrotterSum,scan_x_idxs,scan_y_idxs):
        for scan_x_idx in scan_x_idxs:
            #print('Progress ' +str(scan_y_idx/self.data4D.scan_row*100)+' %\t', end='\r')
            for scan_y_idx in scan_y_idxs:        
                scan_x_angle = self.data4D.scan_angles_x[scan_x_idx]*np.cos(self.data4D.rotation_angle) - self.data4D.scan_angles_y[scan_y_idx]*np.sin(self.data4D.rotation_angle)
                scan_y_angle = self.data4D.scan_angles_x[scan_x_idx]*np.sin(self.data4D.rotation_angle) + self.data4D.scan_angles_y[scan_y_idx]*np.cos(self.data4D.rotation_angle)
                single_trotter=self.data4D.data_4D_Reciprocal[scan_y_idx,scan_x_idx,:,:]
                
                dx,dy = np.meshgrid(self.data4D.Ronchi_angles_x + scan_x_angle, self.data4D.Ronchi_angles_y + scan_y_angle)
                d1 = np.sqrt(dx*dx+dy*dy)
                dx,dy = np.meshgrid(self.data4D.Ronchi_angles_x - scan_x_angle, self.data4D.Ronchi_angles_y - scan_y_angle)
                d2 = np.sqrt(dx*dx+dy*dy) 
                dx,dy = np.meshgrid(self.data4D.Ronchi_angles_x, self.data4D.Ronchi_angles_y)
                d3 = np.sqrt(dx*dx+dy*dy)

                RTrotter_mask = np.ones(single_trotter.shape)
                LTrotter_mask = np.ones(single_trotter.shape)
                if scan_x_idx != np.fix(self.data4D.scan_col/2) or scan_y_idx != np.fix(self.data4D.scan_row/2):
                    RTrotter_mask[d1>=self.data4D.aperturesize]=0
                    RTrotter_mask[d3>=self.data4D.aperturesize]=0
                    RTrotter_mask[d2<self.data4D.aperturesize]=0
                    LTrotter_mask[d2>=self.data4D.aperturesize]=0
                    LTrotter_mask[d3>=self.data4D.aperturesize]=0
                    LTrotter_mask[d1<self.data4D.aperturesize]=0
                RTrotter_phase = np.angle(single_trotter)
                LTrotter_phase = np.angle(single_trotter)
                RTrotter_amp = np.abs(single_trotter)
                LTrotter_amp = np.abs(single_trotter)
                RTrotter_phase[RTrotter_mask==0] =0
                LTrotter_phase[LTrotter_mask==0] =0
                RTrotter_amp[RTrotter_mask==0] =0
                LTrotter_amp[LTrotter_mask==0] =0
                Lpixel_num =np.sum(LTrotter_mask)
                Rpixel_num =np.sum(RTrotter_mask)
                
                if Lpixel_num ==0:              
                    LTrotterSum[scan_y_idx,scan_x_idx] = 0
                else:
                    LTrotter = LTrotter_amp*np.exp(1j*(LTrotter_phase))
                    LTrotterSum[scan_y_idx,scan_x_idx] = np.sum(LTrotter[:])#/Lpixel_num

                    
    def integrate_trotter_higher_order(self,LTrotterSum,scan_x_idxs,scan_y_idxs,order = 2):
        
        for scan_x_idx in scan_x_idxs:
            #print('Progress ' +str(scan_y_idx/self.data4D.scan_row*100)+' %\t', end='\r')
            for scan_y_idx in scan_y_idxs: 
                #if scan_x_idx!=44 or  scan_y_idx!=32:
                #    continue
                scan_x_angle = self.data4D.scan_angles_x[scan_x_idx]*np.cos(self.data4D.rotation_angle) - self.data4D.scan_angles_y[scan_y_idx]*np.sin(self.data4D.rotation_angle)
                scan_y_angle = self.data4D.scan_angles_x[scan_x_idx]*np.sin(self.data4D.rotation_angle) + self.data4D.scan_angles_y[scan_y_idx]*np.cos(self.data4D.rotation_angle)
                single_trotter=self.data4D.data_4D_Reciprocal[scan_y_idx,scan_x_idx,:,:]
                dx,dy = np.meshgrid(self.data4D.Ronchi_angles_x + scan_x_angle*order, self.data4D.Ronchi_angles_y + scan_y_angle*order)
                d1 = np.sqrt(dx*dx+dy*dy)
                dx,dy = np.meshgrid(self.data4D.Ronchi_angles_x + scan_x_angle*(order-1), 
                                    self.data4D.Ronchi_angles_y + scan_y_angle*(order-1))
                d1b = np.sqrt(dx*dx+dy*dy)
                dx,dy = np.meshgrid(self.data4D.Ronchi_angles_x - scan_x_angle*order, 
                                    self.data4D.Ronchi_angles_y - scan_y_angle*order)
                d2 = np.sqrt(dx*dx+dy*dy) 
                dx,dy = np.meshgrid(self.data4D.Ronchi_angles_x - scan_x_angle*(order-1), 
                                    self.data4D.Ronchi_angles_y - scan_y_angle*(order-1))
                d2b = np.sqrt(dx*dx+dy*dy) 
                dx,dy = np.meshgrid(self.data4D.Ronchi_angles_x, self.data4D.Ronchi_angles_y)
                d3 = np.sqrt(dx*dx+dy*dy)

                RTrotter_mask = np.ones(single_trotter.shape)
                LTrotter_mask = np.ones(single_trotter.shape)
                if scan_x_idx != np.fix(self.data4D.scan_col/2) or scan_y_idx != np.fix(self.data4D.scan_row/2):
                    RTrotter_mask = (d1<=self.data4D.aperturesize)*(d1b<=self.data4D.aperturesize)
                    LTrotter_mask = (d2<=self.data4D.aperturesize)*(d2b<=self.data4D.aperturesize)
                
                RTrotter_phase = np.angle(single_trotter)
                LTrotter_phase = np.angle(single_trotter)
                RTrotter_amp = np.abs(single_trotter)
                LTrotter_amp = np.abs(single_trotter)
                RTrotter_phase[RTrotter_mask==0] =0
                LTrotter_phase[LTrotter_mask==0] =0
                RTrotter_amp[RTrotter_mask==0] =0
                LTrotter_amp[LTrotter_mask==0] =0
                Lpixel_num =np.sum(LTrotter_mask)
                Rpixel_num =np.sum(RTrotter_mask)
                
                #Trotter_phase = np.angle(single_trotter)
                #Trotter_amp = np.abs(single_trotter)
                #Trotter_mask=np.logical_or(RTrotter_mask,LTrotter_mask)
                #Trotter_phase[Trotter_mask==0] =0
                #Trotter_amp[Trotter_mask==0] =0
                
                #TrotterPixelNum[scan_y_idx,scan_x_idx]=Lpixel_num
                if scan_y_idx*order>=self.data4D.scan_row or scan_x_idx*order>=self.data4D.scan_col:
                    continue
                    print(scan_y_idx,scan_x_idx)
                if Lpixel_num ==0:              
                    LTrotterSum[scan_y_idx*order,scan_x_idx*order] = 0
                else:
                    LTrotter = LTrotter_amp*np.exp(1j*(LTrotter_phase))
                    LTrotterSum[scan_y_idx*order,scan_x_idx*order] = np.sum(LTrotter[:])#/Lpixel_num
                #if Rpixel_num ==0:
                 #   RTrotterSum[scan_y_idx,scan_x_idx] = 0
                #else:
                #    RTrotter = RTrotter_amp*np.exp(1j*(RTrotter_phase))
                #    RTrotterSum[scan_y_idx,scan_x_idx] = np.sum(RTrotter[:])#/Rpixel_num

                
            
    def integrate_trotter_AC(self,LTrotterSum,RTrotterSum,scan_x_idxs,scan_y_idxs, aberrations):
        #+self.Ronchi_xx,self.Ronchi_yy=np.meshgrid(self.data4D.Ronchi_angles_x,self.data4D.Ronchi_angles_y)
        theta_x=(np.arange(self.data4D.data_4D_trunc.shape[3])-self.data4D.center_x)*self.data4D.calibration
        theta_y=(np.arange(self.data4D.data_4D_trunc.shape[2])-self.data4D.center_y)*self.data4D.calibration
        theta_xx,theta_yy=np.meshgrid(theta_x,theta_y)
        theta=np.sqrt(theta_xx**2+theta_yy**2)
        #self.corrected_trotters = np.zeros(self.data4D.data_4D_Reciprocal.shape,dtype='complex128')
        for scan_x_idx in scan_x_idxs:
            #print('Progress ' +str(scan_y_idx/self.data4D.scan_row*100)+' %\t', end='\r')
            for scan_y_idx in scan_y_idxs:  
                single_trotter=self.data4D.data_4D_Reciprocal[scan_y_idx,scan_x_idx,:,:]

                if scan_x_idx != np.fix(self.data4D.scan_col/2) or scan_y_idx != np.fix(self.data4D.scan_row/2):

                    scan_x_angle = self.data4D.scan_angles_x[scan_x_idx]*np.cos(self.data4D.rotation_angle) - self.data4D.scan_angles_y[scan_y_idx]*np.sin(self.data4D.rotation_angle)
                    scan_y_angle = self.data4D.scan_angles_x[scan_x_idx]*np.sin(self.data4D.rotation_angle) + self.data4D.scan_angles_y[scan_y_idx]*np.cos(self.data4D.rotation_angle)
                    (RTrotter_mask,RTrotter_phase,RTrotter_amp),(LTrotter_mask,LTrotter_phase,
LTrotter_amp),(Trotter_mask,Trotter_phase,Trotter_amp) = mask_trotter(single_trotter,theta_x,theta_y, scan_x_angle,scan_y_angle,self.data4D.aperturesize)
                                  
                    Trotter_correction=Phase_compen(single_trotter,aberrations,theta_x,theta_y,scan_x_angle,scan_y_angle,
                                                self.data4D.aperturesize,self.data4D.wave_len)
            
        ########
                else:
                    Trotter_correction = single_trotter
                    RTrotter_mask = np.ones(single_trotter.shape)
                    LTrotter_mask = np.ones(single_trotter.shape)


                RTrotter_phase = np.angle(Trotter_correction)
                LTrotter_phase = np.angle(Trotter_correction)
                RTrotter_amp = np.abs(Trotter_correction)
                LTrotter_amp = np.abs(Trotter_correction)
                RTrotter_phase[RTrotter_mask==0] =0
                LTrotter_phase[LTrotter_mask==0] =0
                RTrotter_amp[RTrotter_mask==0] =0
                LTrotter_amp[LTrotter_mask==0] =0
                Lpixel_num =np.sum(LTrotter_mask)
                Rpixel_num =np.sum(RTrotter_mask)
                #TrotterPixelNum[scan_y_idx,scan_x_idx]=Lpixel_num
                if Lpixel_num ==0:
                    LTrotterSum[scan_y_idx,scan_x_idx] = 0
                    RTrotterSum[scan_y_idx,scan_x_idx] = 0
                else:
                    LTrotter = LTrotter_amp*np.exp(1j*(LTrotter_phase))
                    RTrotter = RTrotter_amp*np.exp(1j*(RTrotter_phase))
                    LTrotterSum[scan_y_idx,scan_x_idx] = np.sum(RTrotter[:])#/Lpixel_num
                    RTrotterSum[scan_y_idx,scan_x_idx] = np.sum(LTrotter[:])#/Rpixel_num
                    
    

    def run(self, aberrations = [],order=1):
        self.corrected_trotters = []
        self.corrected_4D = np.zeros(self.data4D.data_4D_Reciprocal.shape,dtype=self.data4D.data_4D_Reciprocal.dtype)
        # Now we start to make the reconstruction based on the Trotter information
        if self.data4D.workers>1:
            multiprocessing.set_start_method("fork",force=True)
            chunknb = self.data4D.workers
            #self.LTrotterSum = np.zeros((self.data4D.scan_row,self.data4D.scan_col),dtype=complex)
            #mp_arr = Array('f', self.data4D.scan_row*self.data4D.scan_col) # shared, can be used from multiple processes
            # then in each new process create a new numpy array using:
            #arr = np.frombuffer(mp_arr.get_obj(),c.c_float) # mp_arr and arr share the same memory
            # make it two-dimensional
            #self.LTrotterSum  = arr.reshape((self.data4D.scan_row,self.data4D.scan_col))#.astype(complex) # b and arr share the same memory
            shared_array_base = multiprocessing.Array(c.c_double, self.data4D.scan_row*self.data4D.scan_col*2)
            shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
            self.LTrotterSum = shared_array.view(np.complex128).reshape(self.data4D.scan_row,self.data4D.scan_col)
            if len(aberrations) > 0:
                self.RTrotterSum = np.zeros((self.data4D.scan_row,self.data4D.scan_col),dtype=complex)
            #TrotterPixelNum = np.zeros((self.data4D.scan_row,self.data4D.scan_col))
            scan_y_idxs = np.array_split(range(self.data4D.scan_row), chunknb)  
            scan_x_idxs = range(self.data4D.scan_col) 
            processes = []
            #if __name__ == '__main__':
            for i in range(chunknb):
                #print(i)
                if len(aberrations) == 0:
                    p = Process(target=self.integrate_trotter, args=(self.LTrotterSum,scan_x_idxs,scan_y_idxs[i]))
                else:
                    p = Process(target=self.integrate_trotter_AC, args=(self.LTrotterSum,self.RTrotterSum,scan_x_idxs,scan_y_idxs[i],aberrations))

                p.daemon = True
                p.start()
                processes.append(p)
            [p.join() for p in processes]
            
        else:
            self.LTrotterSum = np.zeros((self.data4D.scan_row,self.data4D.scan_col),dtype=complex)
            self.RTrotterSum = np.zeros((self.data4D.scan_row,self.data4D.scan_col),dtype=complex)

            scan_y_idxs = range(self.data4D.scan_row) 
            scan_x_idxs = range(self.data4D.scan_col) 
#            self.integrate_trotter_higher_order(self.LTrotterSum,scan_x_idxs,scan_y_idxs,order=order)
            if len(aberrations) == 0:
                self.integrate_trotter(self.LTrotterSum,scan_x_idxs,scan_y_idxs)
            else:
                self.integrate_trotter_AC(self.LTrotterSum,self.RTrotterSum,scan_x_idxs,scan_y_idxs,aberrations)

            
        objectL = IFFT_2D(self.LTrotterSum)
        #objectR =self.data4D.IFFT_2D(RTrotterSum)
        
        self.complex = objectL
        self.phase = np.angle(objectL)
        self.amplitude = np.abs(objectL)
        
     
        
    def plot_result(self, sample=1, write_file=False):
        self.fig,ax = self.data4D._init_figure(1,2,(12,4),num = 'Result')
        if sample >1:
            phase = np.array(Image.fromarray(self.phase).resize(np.array(self.phase.shape)*sample,resample=PIL.Image.BICUBIC))
            amplitude = np.array(Image.fromarray(self.amplitude).resize(np.array(self.amplitude.shape)*sample,resample=PIL.Image.BICUBIC))
        else:
            phase = self.phase
            amplitude = self.amplitude

        im0 = ax[0].imshow(phase, extent= [0,self.phase.shape[1]*self.data4D.step_size,0,self.phase.shape[0]*self.data4D.step_size])
        self.fig.colorbar(im0,ax=ax[0])        
        ax[0].set_title('phase')
        im1 = ax[1].imshow(amplitude, extent = [0,self.phase.shape[1]*self.data4D.step_size,0,self.phase.shape[0]*self.data4D.step_size])
        self.fig.colorbar(im1, ax=ax[1])        
        ax[1].set_title("amplitude")
        if write_file == True:
            self.fig.savefig(self.data4D.path+'Result.pdf')
        plt.show()
        
    def save(self):
        try:
            from tifffile import tifffile
            save_tif = True
        except:
            print('Warning: cannot save as .tif (tifffile package required); saving as .txt instead')
            save_tif = False
        
        if save_tif:
            tifffile.imsave(self.data4D.path+'phase_ssb_'+self.data4D.file[:-4]+'.tif', self.phase.astype('float32'), imagej=True)
            tifffile.imsave(self.data4D.path+'amplitude_ssb'+self.data4D.file[:-4]+'.tif', self.amplitude.astype('float32'), imagej=True)

        else:
            np.savetxt(self.data4D.path+'phase_ssb'+self.data4D.file[:-4]+'.txt',self.phase)
            np.savetxt(self.data4D.path+'amplitude_ssb'+self.data4D.file[:-4]+'.txt',self.amplitude)



class WDD():
    def __init__(self, data4D):
        self.data4D = data4D
        self.theta_x=(np.arange(self.data4D.data_4D_trunc.shape[3])-self.data4D.center_x)*self.data4D.calibration
        self.theta_y=(np.arange(self.data4D.data_4D_trunc.shape[2])-self.data4D.center_y)*self.data4D.calibration
        self.theta_xx,self.theta_yy=np.meshgrid(self.theta_x,self.theta_y)
        
    def ProbeFunction(self,Ronchi_angles_x,Ronchi_angles_y,coefficients,Aperture,wave_len):
        angle_xx, angle_yy=np.meshgrid(Ronchi_angles_x,Ronchi_angles_y)
        theta=np.sqrt(angle_xx**2+angle_yy**2)
        func_aberr   = Aberr_func(Ronchi_angles_x,Ronchi_angles_y,coefficients,5)
        func_transfer= np.exp(-1j*2*np.pi/(wave_len*1e-10)*func_aberr)
        func_transfer[theta>Aperture]=0
        return func_transfer    
   
    
    
    def run(self,aberrations=np.zeros(25),epsilon = 0.01):
        #generate the Wigner distribution deconvolution of the probe function. W(r, Q)
        #W(r, Q)=FT(P(k)P*(k+Q))
        probe_function_c = self.ProbeFunction(self.theta_x, self.theta_y, aberrations, self.data4D.aperturesize,self.data4D.wave_len)
        WDD_Probe_Zero = IFFT_2D(probe_function_c*np.conjugate(probe_function_c))
        epsilon=(np.max(np.abs(WDD_Probe_Zero)))**2*epsilon
        WDD_Probe= np.zeros(self.data4D.data_4D_Reciprocal.shape,dtype=self.data4D.data_4D_Reciprocal.dtype)
        for scan_y_idx in range (self.data4D.scan_row):
            print('WDD probe generation progress',(scan_y_idx+1)/self.data4D.scan_row*100,'%',end='\r')
            for scan_x_idx in range (self.data4D.scan_col):
                scan_x_angle = self.data4D.scan_angles_x[scan_x_idx]*np.cos(self.data4D.rotation_angle) - self.data4D.scan_angles_y[scan_y_idx]*np.sin(self.data4D.rotation_angle)
                scan_y_angle = self.data4D.scan_angles_x[scan_x_idx]*np.sin(self.data4D.rotation_angle) + self.data4D.scan_angles_y[scan_y_idx]*np.cos(self.data4D.rotation_angle)
                probe_function_n = self.ProbeFunction(self.theta_x + scan_x_angle,self.theta_y + scan_y_angle,aberrations, self.data4D.aperturesize,self.data4D.wave_len)
                WDD_Probe[scan_y_idx,scan_x_idx]= IFFT_2D(probe_function_c* np.conjugate(probe_function_n))
        data_4D_H=np.zeros(self.data4D.data_4D_Reciprocal.shape,dtype=complex)
        for i in range (self.data4D.scan_row):
            for j in range (self.data4D.scan_col):
                data_4D_H[i,j,:,:]=IFFT_2D(self.data4D.data_4D_Reciprocal[i,j,:,:])

        #determine the object WDD and make the Fourier transfrom
        WDD_probe_conj = np.conjugate(WDD_Probe)
        data_4D_D =  WDD_probe_conj* data_4D_H / (WDD_Probe*WDD_probe_conj + epsilon)
        #del data_4D_H
        #data_4D_D=np.zeros(WDD_Obj.shape,dtype=complex)
        for i in range (self.data4D.scan_row):
            for j in range (self.data4D.scan_col):
                data_4D_D[i,j,:,:]=FFT_2D(data_4D_D[i,j,:,:])
        #del WDD_Obj        
        data_4D_D_conj = np.conjugate(data_4D_D)
        D00= data_4D_D[int(np.fix(self.data4D.scan_row/2)), int(np.fix(self.data4D.scan_col/2)), self.data4D.y_hwindow_size, self.data4D.x_hwindow_size]
        D00= np.sqrt(D00)
        del data_4D_D
        #self.Obj_function=data_4D_D_conj[:,:,self.data4D.y_hwindow_size, self.data4D.x_hwindow_size]/D00
        self.Obj_function=data_4D_D_conj[:,:,int(np.fix(data_4D_D_conj.shape[2]/2)),int(np.fix(data_4D_D_conj.shape[3]/2))]/D00
        self.Obj_function =np.fliplr(IFFT_2D(self.Obj_function))
        self.phase = np.angle(self.Obj_function)
        self.amplitude = np.abs(self.Obj_function)
    

    def plot_result(self, sample=1, write_file=False):
        self.fig,ax = self.data4D._init_figure(1,2,(12,4),num = 'Result')
        if sample > 1:
            phase = np.array(Image.fromarray(self.phase).resize(np.array(self.phase.shape)*sample,resample=PIL.Image.BICUBIC))
            amplitude = np.array(Image.fromarray(self.amplitude).resize(np.array(self.amplitude.shape)*sample,resample=PIL.Image.BICUBIC))
        else:
            phase = self.phase
            amplitude = self.amplitude

        im0 = ax[0].imshow(phase, extent= [0,self.phase.shape[1]*self.data4D.step_size,0,self.phase.shape[0]*self.data4D.step_size])
        self.fig.colorbar(im0,ax=ax[0])        
        ax[0].set_title('phase')
        im1 = ax[1].imshow(amplitude, extent = [0,self.phase.shape[1]*self.data4D.step_size,0,self.phase.shape[0]*self.data4D.step_size])
        self.fig.colorbar(im1, ax=ax[1])        
        ax[1].set_title("amplitude")
        if write_file == True:
            self.fig.savefig(self.data4D.path+'Result.pdf')
        plt.show()
        
    def save(self,appendix=''):
        try:
            from tifffile import tifffile
            save_tif = True
        except:
            print('Warning: cannot save as .tif (tifffile package required); saving as .txt instead')
            save_tif = False
        
        if save_tif:
            tifffile.imsave(self.data4D.path+'phase_wdd'+appendix+'.tif', self.phase.astype('float32'), imagej=True)
            tifffile.imsave(self.data4D.path+'amplitude_wdd'+appendix+'.tif', self.amplitude.astype('float32'), imagej=True)

        else:
            np.savetxt(self.data4D.path+'phase_wdd'+appendix+'.txt',self.phase)
            np.savetxt(self.data4D.path+'amplitude_wdd'+appendix+'.txt',self.amplitude)
            
            
            
  
class iCoM:
           
    def __init__(self,data4D):
        self.data4D = data4D
    
    def calc_CoM(self, dat4d, Mask, RCX, RCY, RCal):
        X, Y = np.meshgrid((np.arange(0, dat4d.shape[3]) - RCX)/RCal, (np.arange(0, dat4d.shape[2]) - RCY)/RCal)
        maskeddat4d = dat4d * (Mask > 0)
        return np.array([np.average(maskeddat4d * X, axis=(2, 3)), np.average(maskeddat4d * Y, axis=(2, 3))])
    
    def run(self,sigma = 0):
        from scipy.ndimage import gaussian_filter as gf
        self.icom = self.calc_CoM(self.data4D.data_4D, self.data4D.BFdisk,self.data4D.center_x ,
                              self.data4D.center_y, self.data4D.scan_angle_step_x)
        self.icom[0] =gf(self.icom[0],sigma)
        self.icom[1] =gf(self.icom[1],sigma)
        #ic = self.icom[0]+self.icom[1]
        #f,ax = plt.subplots(1,2)
        #ax[0].imshow(gf(ic,2))
        #ax[1].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(ic)))))
        #icom = get_iCOM(ps.PixelatedSTEM(data_4D.data_4D), transpose = True, angle = 180, save = True, path =data_4D.path)
        #icom.plot()
        
    def find_rotation(self):
        icomx,icomy = self.icom
        divs, rot = [],[]
        A,C,D=[],[],[]
        for t in np.arange(180):
            t*=np.pi/180
            rot_icoms = icomx*np.cos(t)-icomy*np.sin(t),icomx*np.sin(t)+icomy*np.cos(t)
            gXY,gXX=np.gradient(rot_icoms[0])
            gYY,gYX=np.gradient(rot_icoms[1])
            C.append(np.std(gXY-gYX));D.append(np.std(gXX+gYY));A.append(t)
        R=np.average([A[np.argmin(C)],A[np.argmax(D)]])
        return R*180/np.pi,A,C,D

    def plot_result(self, write_file=False):
        self.fig,ax = self.data4D._init_figure(1,2,(12,4),num = 'iCOM')
        ic = self.icom[0]+self.icom[1]
        im0 = ax[0].imshow(ic, extent= [0,self.ic.shape[0]*self.data4D.step_size,0,self.phase.shape[1]*self.data4D.step_size])
        self.fig.colorbar(im0,ax=ax[0])        
        ax[0].set_title('iCoM')
        im1 = ax[1].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(ic)))), 
                           extent = [0,self.ic.shape[0]*self.data4D.step_size,0,self.phase.shape[1]*self.data4D.step_size])
        self.fig.colorbar(im1, ax=ax[1])        
        ax[1].set_title("FFT")
        if write_file == True:
            self.fig.savefig(self.data4D.path+'Result_iCoM.pdf')
        plt.show()
        
    def save(self):
        try:
            from tifffile import tifffile
            save_tif = True
        except:
            print('Warning: cannot save as .tif (tifffile package required); saving as .txt instead')
            save_tif = False
        
        if save_tif:
            tifffile.imsave(self.data4D.path+'icom_x.tif', self.icom[0].astype('float32'), imagej=True)
            tifffile.imsave(self.data4D.path+'icom_y.tif', self.icom[1].astype('float32'), imagej=True)

        else:
            np.savetxt(self.data4D.path+'icom_x.txt',self.icom[0])
            np.savetxt(self.data4D.path+'icom_y.txt',self.icom[1])


 