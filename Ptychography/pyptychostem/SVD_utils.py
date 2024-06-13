import numpy as np
from numpy.linalg import inv
from skimage.restoration import unwrap_phase

def Aberr_func(x,y,aberrcoeff,order):
    u,v=np.meshgrid(x,y)
    u2=u*u
    u3=u2*u
    u4=u3*u
    
    v2=v*v
    v3=v2*v
    v4=v3*v
    
    C1   = aberrcoeff[0]
    C12a = aberrcoeff[1]
    C12b = aberrcoeff[2]
    C23a = aberrcoeff[3]
    C23b = aberrcoeff[4]
    C21a = aberrcoeff[5]
    C21b = aberrcoeff[6]
    C3   = aberrcoeff[7]
    C34a = aberrcoeff[8]
    C34b = aberrcoeff[9]
    C32a = aberrcoeff[10]
    C32b = aberrcoeff[11]
    C45a = aberrcoeff[12]
    C45b = aberrcoeff[13]
    C43a = aberrcoeff[14]
    C43b = aberrcoeff[15]
    C41a = aberrcoeff[16]
    C41b = aberrcoeff[17]
    C5   = aberrcoeff[18]
    C56a = aberrcoeff[19]
    C56b = aberrcoeff[20]
    C54a = aberrcoeff[21]
    C54b = aberrcoeff[22]
    C52a = aberrcoeff[23]
    C52b = aberrcoeff[24]
    
    if order==1:
        func_aberr=1/2*C1*(u2+v2)+1/2*(C12a*(u2-v2)+2*C12b*u*v)
    elif order==2:
        func_aberr=(1/2*C1*(u2+v2)+1/2*(C12a*(u2-v2)+2*C12b*u*v)+
                    1/3*(C23a*(u3-3*u*v2)+C23b*(-v3+3*u2*v))+
                    1/3*(C21a*(u3+u*v2)+C21b*(v3+u2*v)))
    elif order==3:
        func_aberr=(1/2*C1*(u2+v2)+1/2*(C12a*(u2-v2)+2*C12b*u*v)+
                   1/3*(C23a*(u3-3*u*v2)+C23b*(-v3+3*u2*v))+ 
                   1/3*(C21a*(u3+u*v2)+C21b*(v3+u2*v))+
                   1/4*(C3*(u4+v4+2*u2*v2))+
                   1/4*C34a*(u4-6*u2*v2+v4)+
                   1/4*C34b*(-4*u*v3+4*u3*v)+
                   1/4*C32a*(u4-v4)+
                   1/4*C32b*(2*u3*v+2*u*v3))
    elif order==4:
        func_aberr=(1/2*C1*(u**2 + v**2)+1/2*(C12a*(u**2 - v**2)+C12b*(2*u*v))+
         1/3*(C23a*(u**3 - 3*u*v**2)+C23b*(3*u**2*v - v**3))+
         1/3*(C21a*(u**3 + u*v**2)+C21b*(u**2*v + v**3))+
         1/4*C3*(u**4 + 2*u**2*v**2 + v**4)+
         1/4*(C34a*(u**4 - 6*u**2*v**2 + v**4)+C34b*(4*u**3*v - 4*u*v**3))+
         1/4*(C32a*(u**4 - v**4)+C32b*(2*u**3*v + 2*u*v**3))+
         1/5*(C45a*(u**5 - 10*u**3*v**2 + 5*u*v**4)+C45b*(5*u**4*v - 10*u**2*v**3 + v**5))+
         1/5*(C43a*(u**5 - 2*u**3*v**2 - 3*u*v**4)+C43b*(3*u**4*v + 2*u**2*v**3 - v**5))+
         1/5*(C41a*(u**5 + 2*u**3*v**2 + u*v**4)+C41b*(u**4*v + 2*u**2*v**3 + v**5)))
        
    elif order==5:
        func_aberr=(1/2*C1*(u**2 + v**2)+1/2*(C12a*(u**2 - v**2)+C12b*(2*u*v))+
         1/3*(C23a*(u**3 - 3*u*v**2)+C23b*(3*u**2*v - v**3))+
         1/3*(C21a*(u**3 + u*v**2)+C21b*(u**2*v + v**3))+
         1/4*C3*(u**4 + 2*u**2*v**2 + v**4)+
         1/4*(C34a*(u**4 - 6*u**2*v**2 + v**4)+C34b*(4*u**3*v - 4*u*v**3))+
         1/4*(C32a*(u**4 - v**4)+C32b*(2*u**3*v + 2*u*v**3))+
         1/5*(C45a*(u**5 - 10*u**3*v**2 + 5*u*v**4)+C45b*(5*u**4*v - 10*u**2*v**3 + v**5))+
         1/5*(C43a*(u**5 - 2*u**3*v**2 - 3*u*v**4)+C43b*(3*u**4*v + 2*u**2*v**3 - v**5))+
         1/5*(C41a*(u**5 + 2*u**3*v**2 + u*v**4)+C41b*(u**4*v + 2*u**2*v**3 + v**5))+
         1/6*(C5*(u**6 + 3*u**4*v**2 + 3*u**2*v**4 + v**6))+
         1/6*(C56a*(u**6 - 15*u**4*v**2 + 15*u**2*v**4 - v**6)+
              C56b*(6*u**5*v - 20*u**3*v**3 + 6*u*v**5))+
         1/6*(C54a*(u**6 - 5*u**4*v**2 - 5*u**2*v**4 + v**6)+C54b*(4*u**5*v - 4*u*v**5))+
         1/6*(C52a*(u**6 + u**4*v**2 - u**2*v**4 - v**6)+
              C52b*(2*u**5*v + 4*u**3*v**3 + 2*u*v**5)))
    else:
        print('please choose aberration order up to 3')
    return func_aberr


def mask_trotter(single_trotter,theta_x,theta_y, scan_x_angle,scan_y_angle,Aperture):
    dx,dy = np.meshgrid(theta_x + scan_x_angle, theta_y + scan_y_angle)
    d1 = np.sqrt(dx*dx+dy*dy)
    dx,dy = np.meshgrid(theta_x - scan_x_angle, theta_y - scan_y_angle)
    d2 = np.sqrt(dx*dx+dy*dy)
    dx,dy = np.meshgrid(theta_x, theta_y)
    d3 = np.sqrt(dx*dx+dy*dy)
    
    LTrotter_phase = np.angle(single_trotter)
    LTrotter_amp   =np.abs(single_trotter)
    LTrotter_mask = np.ones(single_trotter.shape)
    LTrotter_mask[d1>Aperture]=0
    LTrotter_mask[d3>Aperture]=0
    LTrotter_mask[d2<Aperture]=0
    LTrotter_phase[LTrotter_mask==0] =0
    LTrotter_amp[LTrotter_mask==0] =0 

    RTrotter_phase = np.angle(single_trotter)
    RTrotter_amp   =np.abs(single_trotter)
    RTrotter_mask = np.ones(single_trotter.shape)
    RTrotter_mask[d1<Aperture]=0
    RTrotter_mask[d3>Aperture]=0
    RTrotter_mask[d2>Aperture]=0
    RTrotter_phase[RTrotter_mask==0] =0
    RTrotter_amp[RTrotter_mask==0] =0
    
    Trotter_phase = np.angle(single_trotter)
    Trotter_amp = np.abs(single_trotter)
    Trotter_mask=np.logical_or(RTrotter_mask,LTrotter_mask)
    Trotter_phase[Trotter_mask==0] =0
    Trotter_amp[Trotter_mask==0] =0
    
    return (RTrotter_mask,RTrotter_phase,RTrotter_amp),(LTrotter_mask,LTrotter_phase,LTrotter_amp),(Trotter_mask,Trotter_phase,Trotter_amp)

def Phase_compen(single_trotter,coefficients,theta_x,theta_y,scan_angles_x,scan_angles_y,Aperture,wave_len, return_AA=False):

    RTrotter,LTrotter,Trotter=mask_trotter(single_trotter,theta_x,theta_y,scan_angles_x,scan_angles_y,Aperture)

    ST_phase=np.angle(single_trotter)
    ST_phase[Trotter[0]==0]=0
    ST_phase_unwrap=unwrap_phase(ST_phase)
    ST_phase_unwrap[Trotter[0]==0]=0
    ST_phase_unwrap=unwrap_phase(ST_phase_unwrap)

    My_trotter=Trotter[2]*np.exp(1j*ST_phase_unwrap)
    My_trotter[Trotter[0]==0]=0

    func_aberr   = Aberr_func(theta_x,theta_y,coefficients,5)
    func_transfer= np.exp(-1j*2*np.pi/(wave_len*1e-10)*func_aberr)

    Ronchi_x_plus,Ronchi_y_plus=theta_x+scan_angles_x,theta_y+scan_angles_y
    func_aberr_plusQ=Aberr_func(Ronchi_x_plus,Ronchi_y_plus,coefficients,5)
    func_transfer_plusQ=np.exp(-1j*2*np.pi/(wave_len*1e-10)*func_aberr_plusQ)

    Ronchi_x_minus,Ronchi_y_minus=theta_x-scan_angles_x,theta_y-scan_angles_y
    func_aberr_minusQ=Aberr_func(Ronchi_x_minus,Ronchi_y_minus,coefficients,5)
    func_transfer_minusQ=np.exp(-1j*2*np.pi/(wave_len*1e-10)*func_aberr_minusQ)
    
    AA_plus=func_transfer*np.conj(func_transfer_plusQ)
    AA_minus=np.conj(func_transfer)*func_transfer_minusQ
    #AA=func_transfer*np.conj(func_transfer_plusQ)+np.conj(func_transfer)*func_transfer_minusQ
    AA_plus_true_phase=unwrap_phase(np.angle(AA_plus))
    AA_plus_true_phase[LTrotter[0]==0]=0
    AA_plus_true_phase=unwrap_phase(AA_plus_true_phase)
    AA_plus_true_phase[LTrotter[0]==0]=0

    AA_minus_true_phase=unwrap_phase(np.angle(AA_minus))
    AA_minus_true_phase[RTrotter[0]==0]=0
    AA_minus_true_phase=unwrap_phase(AA_minus_true_phase)
    AA_minus_true_phase[RTrotter[0]==0]=0

    AA_true_phase=AA_plus_true_phase+AA_minus_true_phase
    AA_true_phase[Trotter[0]==0]=0
    AA=Trotter[2]*np.exp(1j*AA_true_phase)
    AA[Trotter[0]==0]=0

    Trotter_correction=Trotter[2]*np.exp(1j*Trotter[1])*np.exp(-1j*(np.angle(AA)))
    if return_AA:
        return Trotter_correction,AA,My_trotter
    return Trotter_correction

def Matrix_prep(LTrotter,ST_phase_unwrap,theta_x,theta_y,scan_x_angle,scan_y_angle,aberr_order,M,j):
    LT_phase=LTrotter[1]
    LT_amp  =LTrotter[0]
#    aberr_order=2
    Phase=np.zeros(0,dtype=float)
    constvecR=np.zeros(M,dtype=float)
    constvecR[j]=1
    i=0
    for iy in range (LT_phase.shape[0]):
        for ix in range (LT_phase.shape[1]):
            if LT_amp[iy,ix]!=0:
                if i==0:
                    Matrix=OmnilinePrep(theta_x[ix],theta_y[iy],scan_x_angle,scan_y_angle,aberr_order)
                    Matrix=np.hstack((Matrix,constvecR))
                    Phase=np.hstack((Phase,ST_phase_unwrap[iy,ix]))
                else:
                    Matrix_aberration=OmnilinePrep(theta_x[ix],theta_y[iy],scan_x_angle,scan_y_angle,aberr_order)
                    Matrix_aberration=np.hstack((Matrix_aberration,constvecR))
                    Phase=np.hstack((Phase,ST_phase_unwrap[iy,ix]))
                    Matrix=np.vstack((Matrix,Matrix_aberration))
                i+=1
    return (Matrix, Phase)

def OmnilinePrep(kx,ky,qx,qy,nth):
    # AX = b;
    # A is the OmniMatrix, being prepared line by line using this function.
    # X = [C1, C12a, C12b, C23a, C23b, C21a, C21b, C3, C34a, C34b, C32a, C32b]';
    # b = phase{G(Kf,Qp)};

    # G(Kf,Qp) = |A(Kf)|^2.*delta(Qp) + A(Kf-Qp)A.*(Kf)Psi_s(Qp) +  A(Kf)A.*(Kf+Qp)Psi_s.*(-Qp);

    # the phase of A(Kf)A.*(Kf+Qp)Psi_s.*(-Qp)equals to:
    # chi(Kf) - chi(Kf+Qp) - phase(Psi_s(-Qp))
        kx2=kx*kx
        ky2=ky*ky
        kx3=kx2*kx
        ky3=ky2*ky
        kx4=kx3*kx
        ky4=ky3*ky
        ky5=ky4*ky
        kx5=kx4*kx
        ky6=ky5*ky
        kx6=kx5*kx

        sx=kx+qx
        sy=ky+qy
        sx2=sx*sx
        sy2=sy*sy
        sx3=sx2*sx
        sy3=sy2*sy
        sx4=sx3*sx
        sy4=sy3*sy
        sx5=sx4*sx
        sy5=sy4*sy
        sx6=sx5*sx
        sy6=sy5*sy

        if nth==1:
            omniline=1/2*((sx2+sy2)-(kx2+ky2)),1/2*((sx2-sy2)-(kx2-ky2)),sx*sy-kx*ky

        elif nth==2:
            omniline=1/2*((sx2+sy2)-(kx2+ky2)),1/2*((sx2-sy2)-(kx2-ky2)),sx*sy-kx*ky,\
                       1/3*((sx3-3*sx*sy2)-(kx3-3*kx*ky2)),1/3*((-sy3+3*sx2*sy)-(-ky3+3*kx2*ky)),\
                       1/3*((sx3+sx*sy2)-(kx3+kx*ky2)),1/3*((sy3+sx2*sy)-(ky3+kx2*ky))
        elif nth==3:
            omniline=(1/2*((sx2+sy2)-(kx2+ky2)),#C10
                    1/2*((sx2-sy2)-(kx2-ky2)),sx*sy-kx*ky,#C12
                       1/3*((sx3-3*sx*sy2)-(kx3-3*kx*ky2)),1/3*((-sy3+3*sx2*sy)-(-ky3+3*kx2*ky)),#C23
                       1/3*((sx3+sx*sy2)-(kx3+kx*ky2)),1/3*((sy3+sx2*sy)-(ky3+kx2*ky)),#C21
                       1/4*((sx4+sy4+2*sx2*sy2)-(kx4+ky4+2*kx2*ky2)),#C30
                       1/4*((sx4-6*sx2*sy2+sy4)-(kx4-6*kx2*ky2+ky4)),
                       1/4*((-4*sx*sy3+4*sx3*sy)-(-4*kx*ky3+4*kx3*ky)),#C34
                       1/4*((sx4-sy4)-(kx4-ky4)),1/4*((2*sx3*sy+2*sx*sy3)-(2*kx3*ky+2*kx*ky3)))#C32
            
        elif nth==4:
            omniline=(1/2*((sx2+sy2)-(kx2+ky2)),#C10
                    1/2*((sx2-sy2)-(kx2-ky2)),sx*sy-kx*ky,#C12
                       1/3*((sx3-3*sx*sy2)-(kx3-3*kx*ky2)),1/3*((-sy3+3*sx2*sy)-(-ky3+3*kx2*ky)),#C23
                       1/3*((sx3+sx*sy2)-(kx3+kx*ky2)),1/3*((sy3+sx2*sy)-(ky3+kx2*ky)),#C21
                       1/4*((sx4+sy4+2*sx2*sy2)-(kx4+ky4+2*kx2*ky2)),#C30
                       1/4*((sx4-6*sx2*sy2+sy4)-(kx4-6*kx2*ky2+ky4)),
                       1/4*((-4*sx*sy3+4*sx3*sy)-(-4*kx*ky3+4*kx3*ky)),#C34
                       1/4*((sx4-sy4)-(kx4-ky4)),1/4*((2*sx3*sy+2*sx*sy3)-(2*kx3*ky+2*kx*ky3)),#C32
                     1/5*((sx5 - 10*sx3*sy2 + 5*sx*sy4)-(kx5 - 10*kx3*ky2 + 5*kx*ky4)),
                      1/5*((5*sx4*sy - 10*sx2*sy3 + sy5)-(5*kx4*ky - 10*kx2*ky3 + ky5)),#C45
                     1/5*((sx5 - 2*sx3*sy2 - 3*sx*sy4)-(kx5 - 2*kx3*ky2 - 3*kx*ky4)),
                      1/5*((3*sx4*sy + 2*sx2*sy3 - sy5)-(3*kx4*ky + 2*kx2*ky3 - ky5)),#C43
                      1/5*((sx5 + 2*sx3*sy2 + sx*sy4)-(kx5 + 2*kx3*ky2 + kx*ky4)),
                      1/5*((sx4*sy + 2*sx2*sy3 + sy5)-(kx4*ky + 2*kx2*ky3 + ky5)))
            
        elif nth==5:
            omniline=(1/2*((sx2+sy2)-(kx2+ky2)),#C10
                    1/2*((sx2-sy2)-(kx2-ky2)),sx*sy-kx*ky,#C12
                       1/3*((sx3-3*sx*sy2)-(kx3-3*kx*ky2)),1/3*((-sy3+3*sx2*sy)-(-ky3+3*kx2*ky)),#C23
                       1/3*((sx3+sx*sy2)-(kx3+kx*ky2)),1/3*((sy3+sx2*sy)-(ky3+kx2*ky)),#C21
                       1/4*((sx4+sy4+2*sx2*sy2)-(kx4+ky4+2*kx2*ky2)),#C30
                       1/4*((sx4-6*sx2*sy2+sy4)-(kx4-6*kx2*ky2+ky4)),
                       1/4*((-4*sx*sy3+4*sx3*sy)-(-4*kx*ky3+4*kx3*ky)),#C34
                       1/4*((sx4-sy4)-(kx4-ky4)),1/4*((2*sx3*sy+2*sx*sy3)-(2*kx3*ky+2*kx*ky3)),#C32
                     1/5*((sx5 - 10*sx3*sy2 + 5*sx*sy4)-(kx5 - 10*kx3*ky2 + 5*kx*ky4)),
                      1/5*((5*sx4*sy - 10*sx2*sy3 + sy5)-(5*kx4*ky - 10*kx2*ky3 + ky5)),#C45
                     1/5*((sx5 - 2*sx3*sy2 - 3*sx*sy4)-(kx5 - 2*kx3*ky2 - 3*kx*ky4)),
                      1/5*((3*sx4*sy + 2*sx2*sy3 - sy5)-(3*kx4*ky + 2*kx2*ky3 - ky5)),#C43
                      1/5*((sx5 + 2*sx3*sy2 + sx*sy4)-(kx5 + 2*kx3*ky2 + kx*ky4)),
                      1/5*((sx4*sy + 2*sx2*sy3 + sy5)-(kx4*ky + 2*kx2*ky3 + ky5)),#C41
                      1/6*((sx6 + 3*sx4*sy2 + 3*sx2*sy4 + sy6)-(kx6 + 3*kx4*ky2 + 3*kx2*ky4 + ky6)),#C50
                      1/6*((sx6 - 15*sx4*sy2 + 15*sx2*sy4 - sy6)-(kx6 - 15*kx4*ky2 + 15*kx2*ky4 - ky6)),
                      1/6*((6*sx5*sy - 20*sx3*sy3 + 6*sx*sy5)-(6*kx5*ky - 20*kx3*ky3 + 6*kx*ky5)),#C56
                      1/6*((sx6 - 5*sx4*sy2 - 5*sx2*sy4 + sy6)-(kx6 - 5*kx4*ky2 - 5*kx2*ky4 + ky6)),
                      1/6*((4*sx5*sy - 4*sx*sy5)-(4*kx5*ky - 4*kx*ky5)),#C54
                      1/6*((sx6 + sx4*sy2 - sx2*sy4 - sy6)-(kx6 + kx4*ky2 - kx2*ky4 - ky6)),
                      1/6*((2*sx5*sy + 4*sx3*sy3 + 2*sx*sy5)-(2*kx5*ky + 4*kx3*ky3 + 2*kx*ky5)))#C52

        return -np.array(omniline)


def Coefficients_cal(Matrix,phase,wave_len):
    aberrations=SVD_function(Matrix,phase)
    coefficients = aberrations*wave_len*1e-10/(2*np.pi)
    return coefficients # nm

def SVD_function(Matrix,b):
    aberrations=np.zeros(25)
    u,s,vh=np.linalg.svd(Matrix,full_matrices=True)
    d=inv(u).dot(b)
    i_nonzero=np.nonzero(s)[0].shape[0]
    new=np.zeros(min(Matrix.shape[0],Matrix.shape[1]))
    for i in range (i_nonzero):
        new[i]=d[i]/s[i]
    aberrations_buffer=inv(vh).dot(new)
    if aberrations_buffer.shape[0]<=25:
        aberrations[:aberrations_buffer.shape[0]]=aberrations_buffer
    else:
        aberrations=aberrations_buffer[:25]
    return aberrations


    

