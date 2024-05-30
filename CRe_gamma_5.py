'''
This file does all the leptonic gamma-ray stuff taking MeV, cm and s as input. 
'''

import numpy as np
from math import pi, sqrt, exp, floor, ceil
from scipy.interpolate import interp1d

#%% Constants

MeV_to_erg = 1.602e-6 
c = 3e10 # in cm/s, light velocity
m_p = 938.272 # in MeV, proton mass
m_e = 0.511  # in MeV
#n_H = 1 # in cm-3, proton density target
B_0 = 1e-3 #G so cm⁻1/2 g1/2 s-1
e = 4.797e-10 # in cm^3/2 g^1/2 s-1
h_bar = 6.582e-22 # MeV s
e3B_0_MeV = 4.3e-20 # MeV2
M_sol = 2e30 # kg
m_H = 1.67e-27 #u.kg
m_avg = 1.4*m_H
alpha = 1/137
h = 4.1357e-21 # MeV s
h_cgs = 6.6261e-27 # cm2 g s-1
h_bar_cgs = 1.0546e-27 #cm2 g s-1
k_B = 8.6173e-11 # MeV K-1
k_B_cgs = 1.3807e-16 # cm2 g s-2 K-1
m_el = 9.1094e-28 #g
r_0 = e**2/(m_el*(c**2)) # cm electron radius
h_bar_erg = 1.0545919e-27 #erg s
sigma_T = 6.652e-25 #cm2

def integrate(f, x_min, x_max, N_pts = 500):
    g = np.vectorize(f)
    if x_min==x_max:
        int_value=0
    else:
        if x_min == 0:
            z_min = np.log10(1e-1)
        else:
            z_min = np.log10(x_min)
        z_max = np.log10(x_max)
        int_range = np.log(np.logspace(z_min, z_max, N_pts))
        mid_points = (int_range[:-1]+int_range[1:])/2
        y_mid_points = np.exp(mid_points)*g(np.exp(mid_points))
        int_value = np.sum(y_mid_points*np.diff(int_range))
    return int_value

""" def decades(xmin, xmax):
    x1, x2  = np.log10(xmin), np.log10(xmax)
    d1, d2 = floor(x1), ceil(x2)
    return d2 - d1


def integrate(f, xmin, xmax, nptd=10):
    g = np.vectorize(f)
    if xmin == xmax:
        int_value =  0.0   
    else:
        if xmin == 0.0:
            ln_min= np.log(0.1) 
            d_min = np.log10(0.1)
        else:
            ln_min= np.log(xmin)
            d_min = np.log10(xmin)
        ln_max= np.log(xmax)
        d_max = np.log10(xmax)

        d_list = np.array([d_min] + list(range(floor(d_min)+1, ceil(d_max))) + [d_max])

        xd_list = []
        for d1, d2 in zip(d_list[:-1], d_list[1:]):
            xd_list.append(np.logspace(d1, d2, nptd))
        xd_list = sorted(list(set(np.array(xd_list).flatten())))

        ln_x = np.log(xd_list)
        x = np.exp(ln_x)
        y = g(x)
        int_value = np.trapz(y*x, ln_x)
        
    return int_value
 """

#%% Non-thermal Bremsstrahlung emission

def delta(E_gamma, T_e):
    num = E_gamma * m_e
    den = 4 * alpha * T_e * (T_e - E_gamma)
    return num / den


delta_list = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
phi_1_list = [45.79, 45.43, 45.09, 44.11, 42.64, 40.16, 34.97, 29.97, 24.73, 18.09, 13.65]
phi_2_list = [44.46, 44.38, 44.24, 43.65, 42.49, 40.19, 34.93, 29.78, 24.34, 17.28, 12.41]

phi_1 = interp1d(delta_list, phi_1_list, fill_value='extrapolate')
phi_2 = interp1d(delta_list, phi_2_list, fill_value='extrapolate')


def sigma_scat(E_gamma, T_e):
    #log_term = np.log((2 * T_e / m_e) * ((T_e - E_gamma) / E_gamma))
    #phi = 4 * (log_term - (1 / 2))
    delt = delta(E_gamma, T_e)
    term_1 =  (1 + np.power(1 - (E_gamma / T_e), 2)) * phi_1(delt)
    term_2 = - (2 / 3) * (1 - (E_gamma / T_e)) * phi_2(delt)
    pre = (3 / (8 * np.pi)) * alpha * sigma_T 
    return pre * (term_1 + term_2) / E_gamma # cm2 MeV-1


def Phi_e_rel_brem(E_gamma, N_CRe, T_e_max, n):
    def integrand(T_e):
        return sigma_scat(E_gamma, T_e + m_e) * N_CRe(T_e) # in cm-1 MeV-2
    integ = integrate(integrand, E_gamma, T_e_max) # in MeV-1 cm-1
    emi = n * c * integ  # MeV-1 cm-3 s-1
    return E_gamma * E_gamma * emi #MeV cm-3 s-1


#%% Inverse Compton gamma emissivity

def G_3_0(x):
    c_3 = 0.319
    return ((pi**2)/6)*(1+c_3*x)*np.exp(-x)/(1+((pi**2)/6)*c_3*x)

def G_4_0(x):
    c_4 = 6.62
    return (pi**2/6)*(1+c_4*x)*np.exp(-x)/(1+(pi**2/6)*c_4*x)

def g_3(x):
    a_3 = 0.443
    b_3 = 0.54
    alpha_3 = 0.606
    beta_3 = 1.481
    g = a_3*np.power(x, alpha_3)/(1 + b_3*np.power(x, beta_3))
    return 1/(1+g)

def g_4(x):
    a_4 = 0.726
    b_4 = 0.382
    alpha_4 = 0.461
    beta_4 = 1.457
    g = a_4*np.power(x, alpha_4)/(1 + b_4*np.power(x, beta_4))
    return 1/(1+g)

def G_3(x):
    return G_3_0(x)*g_3(x)

def G_4(x):
    return G_4_0(x)*g_4(x)

def N_iso(E_gamma, T_e, T, k_dil):
    "E_ph and E_e in units MeV, T in K"
    # Convert E_ph and E_e to units of m_e*c^2
    E_gamma = E_gamma/m_e
    T_e = T_e/m_e
    # Convert T to units of m_e*c^2
    T = k_B*T/m_e

    z = E_gamma/T_e

    t = 4*T_e*T
    x_0 = z/((1-z)*t)

    tmp=(T/T_e)**2
    pre_1 = 2*(r_0**2)*(m_el**3)*(c**4)*k_dil
    pre_2 = pi*(h_bar_cgs**3)
    pre = pre_1/pre_2
    pre_tmp = pre*tmp

    term_1 = (z**2/(2*(1-z)))*G_3(x_0)
    term_2 = G_4(x_0)

    return pre_tmp*(term_1+term_2)/m_e # MeV-1 s-1


def Phi_e_IC(E_gamma, J_CRe, T_e_max, T, k_dil):
    def integrand(T_e):
        return N_iso(E_gamma, T_e, T, k_dil) * J_CRe(T_e) #  MeV-2 cm-3 s-1
    return E_gamma * E_gamma * integrate(integrand, E_gamma, T_e_max) # MeV cm-3 s-1


#%% Synchrotron radio emission

def syn_photon_E(T_e, B_mG):
    T_e_TeV = T_e*(1e-6)
    return 0.02 * B_mG * (T_e_TeV**2) * 1e-3 #in MeV


def E_c(T_e, B_mG):
    return syn_photon_E(T_e, B_mG)/0.29 #MeV


def R(x):
    num = 1.81*exp(-x)
    den_2 = pow(x, -2/3)+((3.62/pi)**2)
    return num/sqrt(den_2)


def Phi_e_syn(E, J_CRe, B_uG):
    B_mG = 1e-3 * B_uG
    def integrand(T_e):
        x = E/E_c(T_e, B_mG)
        J_val = J_CRe(T_e) # MeV-1 cm-3
        R_val = R(x)
        return J_val*R_val  #MeV-1 cm-3
    integ = integrate(integrand, 1e-3, 1e9) #cm-3
    e3B_0_MeV = B_mG * 4.3e-20
    pre = (sqrt(3)/(2*pi))*(e3B_0_MeV/m_e)*(1/(h_bar*E)) #MeV-1 s-1
    emi = pre * integ
    return E * E * emi # MeV cm-3 s-1 