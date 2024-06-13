import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

# The unit of voltage is kV and returned unit of wavelength is angstrom
def Wavelength(Voltage): 
    emass = 510.99906
    hc = 12.3984244
    wavelength = hc/np.sqrt(Voltage * (2*emass + Voltage))
    
    return wavelength

def bin_Q(orig_data, bin_factor=1):
    new_shape = (orig_data.shape[0], 
                 orig_data.shape[1], 
                 orig_data.shape[2]//bin_factor,bin_factor, 
                 orig_data.shape[3]//bin_factor,bin_factor)
    reshaped_data = orig_data.reshape(new_shape)
    binned_data = reshaped_data.mean(axis=-1).mean(axis=-2)
    
    return binned_data

def FFT_2D(array):
    result = np.fft.fftshift(array)
    result = np.fft.fft2(result)
    result = np.fft.ifftshift(result)
    
    return result

def IFFT_2D(array):
    result = np.fft.fftshift(array)
    result = np.fft.ifft2(result)
    result = np.fft.ifftshift(result)
    
    return result

def set_scan(data_4d, scan_step_size, wave_len):
    scan_row, scan_col = (data_4d.shape[0], data_4d.shape[1])
    scan_x_len, scan_y_len = (scan_step_size*(scan_col-1), scan_step_size*(scan_row-1))
    scan_angle_step_x, scan_angle_step_y = (wave_len/scan_x_len, wave_len/scan_y_len)
    scan_angles_x = (np.arange(scan_col) - np.fix(scan_col/2)) * scan_angle_step_x
    scan_angles_y = (np.arange(scan_row) - np.fix(scan_row/2)) * scan_angle_step_y
    
    return scan_angles_x, scan_angles_y

def calibrate_ronchi(data_4d, aperture_size, threshold=0.2):

    Ronchi_mean  =np.mean(data_4d,(0,1))
    Ronchi_norm  =(Ronchi_mean-np.amin(Ronchi_mean))/np.ptp(Ronchi_mean) # Normalize the Ronchi: (average-min)/(max-min)
    BF_binary    =np.ones(Ronchi_mean.shape)*(Ronchi_norm>threshold) # BF_binary is binary: BF area-1, DF area-0
    xx,yy        =np.meshgrid(np.arange(0,Ronchi_norm.shape[1]), np.arange(0,Ronchi_norm.shape[0]))
    center_x     =np.sum(BF_binary*xx/np.sum(BF_binary))
    center_y     =np.sum(BF_binary*yy/np.sum(BF_binary))
    edge         =(np.sum(np.abs(np.gradient(BF_binary)), axis=0))>threshold
    aperture_radius = np.average(np.sqrt((xx-center_x)**2 + (yy-center_y)**2)[edge])
    calibration = aperture_size/aperture_radius
    Ronchi_angles = ((np.arange(data_4d.shape[3]) - center_x) * calibration, (np.arange(data_4d.shape[2]) - center_y) * calibration)
    
    fig = plt.figure(1, figsize=(45,15))
    grid = AxesGrid(fig, 236, nrows_ncols=(1,3),axes_pad=0.5)
    im = grid[0].imshow(np.log(Ronchi_norm*1e4+1), cmap='viridis')
    grid[0].set_title('Averaged ronchigram')
    im = grid[1].imshow(BF_binary, cmap='gray')
    grid[1].set_title('Bright field disk')
    im = grid[2].imshow(edge, cmap='gray')
    grid[2].set_title('Aperture edge')
    
    return Ronchi_norm, BF_binary, edge, center_x, center_y, calibration, Ronchi_angles


def compress_SSB(SSB_matrix, ePIE_matrix):
    
    dim_ssb = SSB_matrix.shape[0]
    dim_ePIE = ePIE_matrix.shape[0]

    compressed_matrix = np.zeros_like(ePIE_matrix, dtype=SSB_matrix.dtype)
    scale = dim_ssb/dim_ePIE
    for i in range(dim_ePIE):
        for j in range(dim_ePIE):
            # Calculate the start and end indices of the original matrix
            row_start = int(np.round(i * scale))
            row_end = int(np.round((i + 1) * scale))
            col_start = int(np.round(j * scale))
            col_end = int(np.round((j + 1) * scale))
            
            # Compute the area average
            compressed_matrix[i, j] = np.mean(SSB_matrix[row_start:row_end, col_start:col_end])
    
    return compressed_matrix


# unit of defocus and Cs in angstrom
def generate_probe(ronchi_angles, aperture, wave_len, defocus=0, Cs=0):
    # (ronchi_x, ronchi_y) = ronchi_angles
    angle_xx,angle_yy = np.meshgrid(ronchi_angles[0], ronchi_angles[1])
    angleSQ = angle_xx**2 + angle_yy**2
    angleSQ2 = angleSQ**2
    chi = 2*np.pi*(defocus*angleSQ/2 + Cs*angleSQ2/4)/wave_len
    mask = np.sqrt(angleSQ)<=aperture
    probe_phase = mask*np.exp(1j*chi)
    probe = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(probe_phase)))

    return probe


def preprocess(data_4D, scan_step_size, aperture_size, calibration, Ronchi_angles, wave_len, iterations, obj_range_guess, 
               defocus=0, Cs=0, ssb_guess=False, ssb_matrix=[]):
    obj_min, obj_max = obj_range_guess
    probe_initial = generate_probe(Ronchi_angles, aperture_size, wave_len)
    
    # sampling
    dk = calibration / wave_len
    dx = 1 / (data_4D.shape[3] * dk)
    dy = 1 / (data_4D.shape[2] * dk)
    ppx = np.arange(data_4D.shape[1]) * scan_step_size
    ppy = np.arange(data_4D.shape[0]) * scan_step_size
    ppX, ppY = np.meshgrid(ppx, ppy)
    xx = np.ceil(ppX.flatten() / dx).astype(int)
    yy = np.ceil(ppY.flatten() / dy).astype(int)
    Xmin = xx - xx.min()
    Xmax = Xmin + data_4D.shape[3]
    Ymin = yy - yy.min()
    Ymax = Ymin + data_4D.shape[2]
    
    probe_range_series = (Xmin, Xmax, Ymin, Ymax)

    obj_sizeX = xx.max() - xx.min() + data_4D.shape[3] + 1
    obj_sizeY = yy.max() - yy.min() + data_4D.shape[2] + 1

    # visualized areas
    row_start = (Ymin.min() + Ymax.min()) // 2
    row_end = (Ymin.max() + Ymax.max()) // 2
    col_start = (Xmin.min() + Xmax.min()) // 2
    col_end = (Xmin.max() + Xmax.max()) // 2
    scan_area = np.array([row_start, row_end, col_start, col_end])
    
    # flatten the 4D dataset to 3D dataset
    num_diff_patts = data_4D.shape[0] * data_4D.shape[1]
    diff_patts = data_4D.reshape(num_diff_patts, data_4D.shape[2], data_4D.shape[3])
    print(f"Flattened diffraction patterns shape: {diff_patts.shape}")

    # initialize a random object function 
    obj = np.random.rand(obj_sizeY, obj_sizeX) * np.exp(1j * ((obj_max - obj_min) * np.random.rand(obj_sizeY, obj_sizeX) + obj_min))

    # correct the power of the initial probe
    probe = probe_initial.copy()
    probe_intensity = np.sum(diff_patts) / num_diff_patts
    probe *= np.sqrt(probe_intensity / (probe.shape[0] * probe.shape[1] * np.sum(np.abs(probe) ** 2)))
    diff_patts = np.sqrt(diff_patts)
    
    if ssb_guess:
        ePIE_matrix = obj[row_start:row_end, col_start:col_end]
        ssb_estimate = compress_SSB(ssb_matrix, ePIE_matrix)
        obj[row_start:row_end, col_start:col_end] = ssb_estimate
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].set_title("Initial Object Phase")
    ax0 = axes[0].imshow(np.angle(obj))
    plt.colorbar(ax0, ax=axes[0])

    axes[1].set_title("Initial Object Amplitude")
    ax1 = axes[1].imshow(np.abs(obj))
    plt.colorbar(ax1, ax=axes[1])

    axes[2].set_title("Initial Probe Amplitude")
    ax2 = axes[2].imshow(np.abs(probe))
    plt.colorbar(ax2, ax=axes[2])
    
    return diff_patts, obj, probe, probe_range_series, scan_area

def run_ePIE(diff_patts, obj, probe, iter_parameters, probe_range_series, scan_area):
    from IPython.display import display, clear_output
    
    beta_obj, beta_probe, iterations = iter_parameters
    zero_constant = 1e-10
    Xmin, Xmax, Ymin, Ymax = probe_range_series
    
    fourier_error = np.zeros((iterations,diff_patts.shape[0]), dtype=float)

    obj_series = [obj.copy()]
    
    for itt in range(iterations):
        clear_output(wait=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].cla()
        axes[0].set_title(f"Phase Image of Object, iteration: {itt+1}" )
        ax0 = axes[0].imshow(np.angle(obj[scan_area[0]:scan_area[1], scan_area[2]:scan_area[3]]))

        axes[1].cla()
        axes[1].set_title("Reconstructed Amplitude Image of Probe")
        ax1 = axes[1].imshow(np.abs(probe))

        
        
        # axes[0,0].cla()
        # axes[0,0].set_title(f"Phase Image of Object, iteration: {itt+1}" )
        # ax0 = axes[0,0].imshow(np.angle(obj[scan_area[0]:scan_area[1], scan_area[2]:scan_area[3]]))

        # axes[0,1].cla()
        # axes[0,1].set_title("Reconstructed Amplitude Image of Probe")
        # ax1 = axes[0,1].imshow(np.abs(probe))

        # axes[1,0].cla()
        # axes[1,0].set_title(f"Phase Image of Object, iteration: {itt+1}" )
        # ax2 = axes[1,0].imshow(np.angle(obj[scan_area[0]:scan_area[1], scan_area[2]:scan_area[3]]))

        # axes[1,1].cla()
        # axes[1,1].set_title("Reconstructed Amplitude Image of Probe")
        # ax3 = axes[1,1].imshow(np.abs(probe))

        plt.show()
        for j in np.random.permutation(diff_patts.shape[0]):
            obj_box = obj[Ymin[j]:Ymax[j], Xmin[j]:Xmax[j]]
            obj_max = np.max(np.abs(obj_box))
            probe_max = np.max(np.abs(probe))

            # create new exit wave function
            exit_wave = obj_box * probe
            EW_kspace = FFT_2D(exit_wave)
            revised_EW_kspace = diff_patts[j, :, :] * np.exp(1j * np.angle(EW_kspace))

            # clear_output(wait=True)
            # fig, axes = plt.subplots(2, 2, figsize=(12, 5))

            # axes[0,0].cla()
            # #axes[0,0].set_title("abs(exit_wave)")
            # axes[0,0].set_title(f"Phase Image of Object, before" )
            # #ax0 = axes[0,0].imshow(np.abs(exit_wave))
            # ax0 = axes[0,0].imshow(np.angle(obj[scan_area[0]:scan_area[1], scan_area[2]:scan_area[3]]))

            
            # update object and probe using ePIE algorithm
            revised_EW = IFFT_2D(revised_EW_kspace)
            update_factor = (revised_EW - exit_wave)
            obj[Ymin[j]:Ymax[j], Xmin[j]:Xmax[j]] += beta_obj * (np.conj(probe) * update_factor) / (probe_max**2 + zero_constant)
            probe += beta_probe * (np.conj(obj_box) * update_factor) / (obj_max**2 + zero_constant)

            fourier_error[itt, j] = np.sum(np.abs(diff_patts[j, :, :] - np.abs(EW_kspace)))

            
            # axes[0,1].cla()
            # axes[0,1].set_title(f"Phase Image of object update, after: {j}" )
            # #ax1 = axes[0,1].imshow(np.angle(obj[scan_area[0]:scan_area[1], scan_area[2]:scan_area[3]]))
            # ax1 = axes[0,1].imshow(np.angle((np.conj(probe) * update_factor) / (probe_max**2 + zero_constant)))
            
            # #axes[0,1].cla()
            # #axes[0,1].set_title("abs(exit_wave)")
            # #ax1 = axes[0,1].imshow(np.abs(exit_wave))

            # axes[1,0].cla()
            # axes[1,0].set_title("abs EW_kspace" )
            # ax2 = axes[1,0].imshow(np.abs(EW_kspace))

            
            # axes[1,1].cla()
            # axes[1,1].set_title("abs revised_EW_kspace")
            # #ax3 = axes[1,1].imshow(np.abs(probe))
            # ax3 = axes[1,1].imshow(np.abs(revised_EW_kspace))
            
            # plt.show()
            # print(j)
            # print('iteration',itt)

        
        obj_series.append(obj.copy())

    error = np.sum(fourier_error, axis=1) / diff_patts.shape[0]
    obj_series = np.array(obj_series)

    # Plot final results
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    clear_output(wait=True)
    axes[0].set_title("Reconstructed Phase Image of Object")
    ax0 = axes[0].imshow(np.angle(obj[scan_area[0]:scan_area[1], scan_area[2]:scan_area[3]]))
    plt.colorbar(ax0, ax=axes[0])
    axes[1].set_title("Reconstructed Amplitude Image of Probe")
    ax1 = axes[1].imshow(np.abs(probe))
    plt.colorbar(ax1, ax=axes[1])
    axes[2].set_title("Error of Iteration")
    ax2 = axes[2].plot(error, 'r')

    return obj, probe, error, obj_series

def run_WASP(diff_patts, obj, probe, iter_parameters, probe_range_series, scan_area):
    from IPython.display import display, clear_output
    beta_obj, beta_probe, iterations = iter_parameters
    zero_constant = 1e-10
    Xmin, Xmax, Ymin, Ymax = probe_range_series
    
    fourier_error = np.zeros((iterations,diff_patts.shape[0]), dtype=float)

    obj_series = [obj.copy()]

    for itt in range(iterations):
        clear_output(wait=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].cla()
        axes[0].set_title(f"Phase Image of Object, iteration: {itt+1}" )
        ax0 = axes[0].imshow(np.angle(obj[scan_area[0]:scan_area[1], scan_area[2]:scan_area[3]]))

        axes[1].cla()
        axes[1].set_title("Reconstructed Amplitude Image of Probe")
        ax1 = axes[1].imshow(np.abs(probe))

        print(f'WASP: {int((itt + 1) / iterations * 100)}%', end='\r')

        numP = np.zeros_like(probe)
        denP = np.zeros_like(probe)
        numO = np.zeros_like(obj)
        denO = np.zeros_like(obj)

        plt.show()
        for j in np.random.permutation(diff_patts.shape[0]):
            obj_box = obj[Ymin[j]:Ymax[j], Xmin[j]:Xmax[j]]
            obj_max = np.max(np.abs(obj_box))
            probe_max = np.max(np.abs(probe))

            # create new exit wave function
            exit_wave = obj_box * probe
            EW_kspace = FFT_2D(exit_wave)
            revised_EW_kspace = diff_patts[j, :, :] * np.exp(1j * np.angle(EW_kspace))

            # update object and probe using WASP algorithm
            revised_EW = IFFT_2D(revised_EW_kspace)
            update_factor = (revised_EW - exit_wave)
            
            obj[Ymin[j]:Ymax[j], Xmin[j]:Xmax[j]] += beta_obj * (np.conj(probe) * update_factor) / (probe_max**2 + zero_constant)
            probe += beta_probe * (np.conj(obj_box) * update_factor) / (obj_max**2 + zero_constant)

            numO[Ymin[j]:Ymax[j], Xmin[j]:Xmax[j]] += np.conj(probe) * revised_EW
            denO[Ymin[j]:Ymax[j], Xmin[j]:Xmax[j]] += np.abs(probe)**2
            numP += np.conj(obj_box) * revised_EW
            denP += np.abs(obj_box)**2

            fourier_error[itt, j] = np.sum(np.abs(diff_patts[j, :, :] - np.abs(EW_kspace)))
        
        obj = numO / (denO + zero_constant)
        probe = numP / (denP + zero_constant)
        obj_series.append(obj.copy())

    error = np.sum(fourier_error, axis=1) / diff_patts.shape[0]
    obj_series = np.array(obj_series)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    axes[0].set_title("Reconstructed Phase Image of Object")
    ax0 = axes[0].imshow(np.angle(obj[scan_area[0]:scan_area[1], scan_area[2]:scan_area[3]]))
    plt.colorbar(ax0, ax=axes[0])

    axes[1].set_title("Reconstructed Amplitude Image of Probe")
    ax1 = axes[1].imshow(np.abs(probe))
    plt.colorbar(ax1, ax=axes[1])

    axes[2].set_title("Error of Iteration")
    ax2 = axes[2].plot(error, 'r')

    return obj, probe, error, obj_series

