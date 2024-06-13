import numpy as np
def circle(x0,y0,ridius):
    phi=np.pi*(np.linspace(-1,1,200))
    x=x0+ridius*np.cos(phi)
    y=y0+ridius*np.sin(phi)
    return x,y
def FFT_2D (array):
    result=np.fft.fft2(array)
    result=np.fft.fftshift(result)
    return result
def IFFT_2D (array):
    result=np.fft.ifftshift(array)
    result=np.fft.ifft2(result)
    return result

def Wavelength(Voltage):
    emass = 510.99906
    hc = 12.3984244
    wavelength = hc/np.sqrt(Voltage * (2*emass + Voltage))
    return wavelength