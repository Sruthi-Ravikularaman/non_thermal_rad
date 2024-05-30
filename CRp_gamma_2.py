'''
This file does all the hadronic gamma-ray stuff taking MeV, cm and s as input. 
User can also choose to take nucleus-nucleus interactions into account.
'''

from math import sqrt, pi, exp
import numpy as np


#%%  Useful definitions
MeV_to_erg = 1.602e-6 
m_p = 938.272 # in MeV, proton mass
m_p_ev = 938.272e6 # in eV
m_pi = 134.977 # in MeV, neutral pion mass
T_p_th = 280 # in MeV, proton kinetic energy threshold for pion production
T_max = 1e11  # in MeV, maximum proton kinetic energy of interest
c = 3e10 # in cm/s, light velocity
#n_H = 100 # in cm-3, proton density target
sigma_R0 = 58.1e-27 # in cm2, cross-section
sigma_R_pp = 31.4e-27 # in cm2, geometric cross-section
T_p_0 = 1e6 # MeV, energy at which the nucleus_nucleus cross-section growth becomes noticeable
M_sol = 2e30 # kg
m_H = 1.67e-27 # kg
m_avg = 1.4*m_H
sigma_R0 = 58.1e-27 # in cm2, cross-section
sigma_R_pp = 31.4e-27 # in cm2, geometr
T_p_0 = 1e6 # MeV, energy at which the nucleus_nucleus cross-section growth becomes noticeable
Y_i_p = 1 # ratio by number of a given projectile i wrt p, assuming only CR p projectiles
Y_j_t = 1 # ratio by number of a target j wrt H, assuming only H in ISM
A_p_list = [1, 4, 14, 25, 56]
Y_p_list = [1, 5.51e-2, 3.25e-3, 1.61e-3, 3.68e-4]
A_t_list = [1, 4, 12, 14, 16, 20, 24, 28, 32, 56]
Y_t_list = [1, 9.59e-2, 4.65e-4, 8.3e-5, 8.3e-4, 1.2e-4, 3.87e-5, 3.69e-5, 1.59e-5, 3.25e-5]

# unit-less relativistic Breit-Wigner distribution
sigma_0 = 7.66e-30  # in cm^2
M_res = 1188.3  # in MeV, resonance mass
W_res = 226.4  # in MeV, resonance width
gamma_cst = M_res*sqrt((M_res**2)+(W_res**2))
K = (sqrt(8)*M_res*W_res*gamma_cst)/(pi*sqrt((M_res**2)+gamma_cst))


#%% Tools

#%% Tools

def velocity(E_kin, rest_mass):
    '''
    Gives the velocity, it's ratio with c and Lorentz factor.

    Parameters
    ----------
    E_kin : float
        Kinetic energy of particle in MeV.
    rest_mass : float
        Rest mass energy of particle in MeV.

    Returns
    -------
    v : float
        Particle velocity in cm s^-1.
    beta : float
        v/c.
    gamma : float
        Lorentz factor.

    '''
    E_kin = E_kin
    rest_mass = rest_mass
    gamma = 1+(E_kin/rest_mass)
    beta = np.sqrt(1-(1/(gamma**2)))
    v = c*beta
    return v, beta, gamma

def integrate(f, x_min, x_max, N_pts = 500):
    g = np.vectorize(f)
    if x_min==x_max:
        int_value=0
    else:
        if x_min == 0:
            z_min = np.log10(1e-50)
        else:
            z_min = np.log10(x_min)
        z_max = np.log10(x_max)
        int_range = np.log(np.logspace(z_min, z_max, N_pts))
        mid_points = (int_range[:-1]+int_range[1:])/2
        y_mid_points = np.exp(mid_points)*g(np.exp(mid_points))
        int_value = np.sum(y_mid_points*np.diff(int_range))
    return int_value


def E_CM_squared(T_p):
    """
    Takes proton kinetic energy and gives the energy available in CM ref squared s.
    :param T_p: MeV
    :return s: MeV^2
    """
    s = 2*m_p*(T_p+(2*m_p))
    return s


def pi_CM(T_p):
    """
    Takes proton kinetic energy and gives the total energy and momentum of the pion in the CM ref.
    :param T_p: MeV
    :return E_pi: MeV
    :return P_pi: MeV
    """
    s = E_CM_squared(T_p)
    E_pi = (s-(4*(m_p**2))+(m_pi**2))/(2*sqrt(s))
    P_pi = sqrt((E_pi**2)-(m_pi**2))
    return E_pi, P_pi


def relat_CM(T_p):
    """
    Takes proton kinetic energy and gives lorentz factor and reduced velocity in the CM frame.
    :param T_p: MeV
    :return beta_CM: 1
    :return lorentz_CM: 1
    """
    s = E_CM_squared(T_p)
    lorentz_CM = (T_p+(2*m_p))/sqrt(s)
    beta_CM = sqrt(1-(1/(lorentz_CM**2)))
    return beta_CM, lorentz_CM


def E_pi_max_LAB(T_p):
    """
    Takes proton kinetic energy and gives maximum pion energy in lab frame.
    :param T_p: MeV
    :return E_pi_max: MeV
    """
    beta_CM, lorentz_CM = relat_CM(T_p)
    E_pi, P_pi = pi_CM(T_p)
    EP_pi = E_pi+(P_pi*beta_CM)
    E_pi_max = lorentz_CM*EP_pi
    return E_pi_max


def relat_pi_LAB(T_p):
    """
    Takes proton kinetic energy and gives the lorentz factor and reduced velocity for the pion in lab frame.
    :param T_p: MeV
    :return beta_CM: 1
    :return lorentz_CM: 1
    """
    lorentz_pi_LAB = E_pi_max_LAB(T_p)/m_pi
    beta_pi_LAB = sqrt(1-(1/(lorentz_pi_LAB**2)))
    return beta_pi_LAB, lorentz_pi_LAB


def E_gamma_ext(T_p):
    """
    Takes proton kinetic energy and gives maximum and minimum energies of photons that can be produced.
    :param T_p: MeV
    :return E_gamma_min: MeV
    :return E_gamma_max: MeV
    """
    beta_pi_LAB, lorentz_pi_LAB = relat_pi_LAB(T_p)
    E_gamma_min = (m_pi/2)*lorentz_pi_LAB*(1-beta_pi_LAB)
    E_gamma_max = (m_pi/2)*lorentz_pi_LAB*(1+beta_pi_LAB)
    return E_gamma_min, E_gamma_max


def T_p_ext(E_gamma):
    """
    For a certain photon energy, makes a list of all the proton energies that contribute
    and returns min and max of this list.
    :param E_gamma: MeV, gamma ray energy
    :return T_p_min: MeV, smallest proton energy capable of producing gamma ray of said energy
    :return T_p_max: MeV, highest proton energy capable of producing gamma ray of said energy
    """
    T_p_list = np.logspace(np.log10(280), 20, 1000)
    i = 0
    T_i = T_p_list[i]
    while (E_gamma < E_gamma_ext(T_i)[0] or E_gamma > E_gamma_ext(T_i)[1]):
        i += 1
        T_i = T_p_list[i]
        
    T_p_min = T_i
    
    j = len(T_p_list)-1
    T_j = T_p_list[j]
    while (E_gamma < E_gamma_ext(T_j)[0] or E_gamma > E_gamma_ext(T_j)[1]):
        j -= 1
        T_j = T_p_list[j]
        
    T_p_max = T_j
    # print("T_p_min", T_p_min, "T_p_max", T_p_max)
    
    return T_p_min, T_p_max
    
    
def XY_vars(T_p, E_gamma):
    """
    Takes the proton kinetic energy and photon energy and gives Y_gamma, Y_gamma_max and X_gamma.
    :param T_p: MeV
    :param E_gamma: MeV
    :return Y_gamma: MeV
    :return Y_gamma_max: MeV
    :return X_gamma: 1
    """
    _,E_gamma_max = E_gamma_ext(T_p)
    Y_gamma = E_gamma+((m_pi**2)/(4*E_gamma))
    Y_gamma_max = E_gamma_max + ((m_pi ** 2) / (4 * E_gamma_max))
    X_gamma = (Y_gamma-m_pi)/(Y_gamma_max-m_pi)
    return Y_gamma, Y_gamma_max, X_gamma


def abg(T_p):
    """
    Takes the proton kinetic energy and gives  alpha, beta and gamma parameters.
    :param T_p: MeV
    :return alpha, beta, gamma: 1
    """
    theta_p = T_p/m_p
    kappa = 3.29-((1/5)*(theta_p**(-3.0/2.0)))
    if (T_p >= T_p_th) and (T_p <= 1000):
        alpha = 1.0
        beta = kappa
        gamma = 0
    else:
        q = (T_p-1000)/m_p
        mu = (5/4)*(q**(5.0/4.0))*np.exp(-(5.0/4.0)*q)
        if (T_p > 1000) and (T_p <= 4000):
            alpha = 1.0
            beta = mu+2.45
            gamma = mu+1.45
        elif (T_p > 4000) and (T_p <= 20000):
            alpha = 1.0
            beta = ((3/2)*mu)+4.95
            gamma = mu+1.50
        elif (T_p > 20000) and (T_p <= 100000):
            alpha = 0.5
            beta = 4.2
            gamma = 1
        else:
            alpha = 0.5
            beta = 4.9
            gamma = 1
    return alpha, beta, gamma


# NUCLEUS-NUCLEUS ENHANCEMENT

def sigma_R(A_p, A_t):
    beta_0 = 2.247-(0.915*(1+(A_t**(-1/3))))
    A_factor = (A_p**(1/3))+(A_t**(1/3))-beta_0*((A_p**(-1/3))+(A_t**(-1/3)))
    sigma = sigma_R0*(A_factor**2)
    return sigma


sum_c_1 = 0
sum_c_2 = 0
for i in range(1, len(A_p_list)):
    sum_c_1 += A_p_list[i]*Y_p_list[i]
for i in range(1, len(A_t_list)):
    sum_c_2 += A_t_list[i]*Y_t_list[i]
eps_c = 1 + ((1/2)*sum_c_1) + ((1/2)*sum_c_2)

sum_1_1 = 0
sum_1_2 = 0
for i in range(1, len(A_p_list)):
    sum_1_1 += Y_p_list[i]*sigma_R(A_p_list[0], A_p_list[i])/sigma_R_pp
for i in range(1, len(A_t_list)):
    sum_1_2 += Y_t_list[i]*sigma_R(A_p_list[0], A_t_list[i])/sigma_R_pp
eps_1 = ((1/2)*sum_1_1)+((1/2)*sum_1_2)

sum_2 = 0
for i in range(1, len(A_p_list)):
    for j in range(1, len(A_t_list)):
        A_sum = (A_p_list[i]*sigma_R(A_p_list[0], A_t_list[j]))+(A_t_list[j]*sigma_R(A_p_list[0], A_p_list[i]))
        sum_2 += Y_p_list[i]*Y_t_list[j]*A_sum/sigma_R_pp
eps_2 = (1/2)*sum_2

#%% Intermediate functions

# Cross-sections

def sigma_single_pion(T_p):
    """
    Takes proton kinetic energy and gives cross-section for single-pion production.
    :param T_p: MeV
    :return sigma_1: cm^2
    """
    s = E_CM_squared(T_p)
    f_BW = (m_p*K)/((((sqrt(s)-m_p)**2-M_res**2)**2)+((M_res**2)*(W_res**2)))
    eta_num = sqrt(((s-(m_pi**2)-4*(m_p**2))**2)-16*(m_pi**2)*(m_p**2))
    eta_den = (2*m_pi*sqrt(s))
    eta = eta_num/eta_den
    sigma_1 = sigma_0*(eta**1.95)*(1+eta+(eta**5))*(f_BW**1.86)
    return sigma_1


def sigma_double_pion(T_p):
    """
    Takes proton kinetic energy and gives cross-section for double-pion production.
    :param T_p: MeV
    :return sigma_2: cm^2
    """
    if (T_p >= 560) and (T_p <= 2000):
        sigma = 5.7/(1+exp((-9.3e-3)*(T_p-1400)))
    elif T_p < 560:
        sigma = 0
    else:
        sigma = -1
    sigma_2 = sigma*(1e-27)
    return sigma_2


def sigma_inel(T_p):
    """
    Takes proton kinetic energy and gives inelastic pion-production cross-section.
    :param T_p: MeV
    :return sigma_3: cm^2
    """
    factor_1 = 30.7-(0.96*np.log(T_p/T_p_th))+(0.18*((np.log(T_p/T_p_th))**2))
    factor = factor_1*((1-((T_p_th/T_p)**1.9))**3)
    sigma_3 = factor*(1e-27)
    return sigma_3


def avg_pi0_multiplicity(T_p):
    """
    Takes proton kinetic energy and gives average pion multiplicity.
    :param T_p: MeV
    :return n_avg: 1
    """
    a_1 = 0.728
    a_2 = 0.596
    a_3 = 0.491
    a_4 = 0.2503
    a_5 = 0.117
    if (T_p >= 1000) and (T_p < 5000):
        Q_p = (T_p-T_p_th)/m_p
        n_avg = (-6e-3)+(0.237*Q_p)-(0.023*(Q_p**2))
    elif (T_p >= 5000):
        xi_p = (T_p-3000)/m_p
        n_avg = a_1*(xi_p**a_4)*(1+np.exp(-a_2*(xi_p**a_5)))*(1-np.exp(-a_3*(xi_p**(1.0/4.0))))
    else:
        n_avg = 0
    return n_avg


def sigma_pion(T_p):
    """
    Takes proton kinetic energy and gives the cross-section of relevant pion-production cross-section.
    :param T_p: MeV
    :return sigma_pi: cm^2
    """
    if (T_p >= T_p_th) and (T_p < 2000):
        sigma_pi = sigma_single_pion(T_p)+sigma_double_pion(T_p)
    elif T_p >= 2000:
        sigma_pi = sigma_inel(T_p)*avg_pi0_multiplicity(T_p)
    else:
        sigma_pi = -1
    return sigma_pi


# Gamma-ray production differential cross-section

def G(T_p):
    sigma_ratio = sigma_inel(T_p)/sigma_inel(T_p_0)
    if sigma_ratio >= 1:
        max = sigma_ratio
    else:
        max = 1
    return 1+np.log(max)


def mod_sigma_inel(A_p, A_t, T_p):
    return sigma_R(A_p, A_t)*G(T_p)


def epsilon(T_p):
    eps = eps_c + ((eps_1 + eps_2) * (sigma_R_pp * G(T_p) / sigma_inel(T_p)))
    return eps


def A_max(T_p):
    """
    Takes proton kinetic energy and returns the peak value of the gamma-ray production differential cross-section.
    :param T_p: MeV
    :return A_peak: cm^2/MeV
    """
    E_pi_max = E_pi_max_LAB(T_p)
    b_0 = 5.9
    theta_p = T_p/m_p
    if (T_p >= T_p_th) and (T_p < 1000):
        A_peak = b_0*sigma_pion(T_p)/E_pi_max
    elif T_p >= 1000:
        if T_p < 5000:
            b_1 = 9.53
            b_2 = 0.52
            b_3 = 0.054
        else:
            b_1 = 9.13
            b_2 = 0.35
            b_3 = 9.7e-3
        A_peak = b_1*(theta_p**-b_2)*exp(b_3*(np.log(theta_p)**2))*sigma_pion(T_p)/m_p
    else:
        A_peak = -1
    return A_peak


def F(T_p, E_gamma):
    """
    Takes proton kinetic energy and photon energy and gives F
    :param T_p: MeV
    :param E_gamma: MeV
    :return F_p: 1
    """
    Y_gamma, Y_gamma_max, X_gamma = XY_vars(T_p, E_gamma)
    C = 3.0*m_pi/Y_gamma_max
    alpha, beta, gamma = abg(T_p)
    num = 1-(X_gamma**alpha)
    den = 1+(X_gamma/C)
    #print('num=',(num**beta)/den)
    F_p = (num**beta)/(den**gamma)
    return F_p


def gamma_diff_sigma(T_p, E_gamma, enh=True):
    """
    Takes proton kinetic energy and photon energy and returns gamma-ray production differential cross-section.
    :param T_p: MeV
    :param E_gamma: MeV
    :param enh: bool, inclus nucleus-nucleus interaction if True
    :return d_sigma: cm^2/MeV
    """
    d_sigma = A_max(T_p)*F(T_p, E_gamma)
    if enh==True:
        d_sigma = epsilon(T_p)*d_sigma
    return d_sigma


#%% Gamma-ray flux

def Phi_pp_gamma(E_gamma, J, n):
    # E_gamma in MeV
    def integrand(T):
        return gamma_diff_sigma(T, E_gamma, True) * J(T) # MeV-2 s-1 sr-1
    T_p_min, _ = T_p_ext(E_gamma)
    T_p_max = 1e9
    integ = integrate(integrand, T_p_min, T_p_max) # MeV-1 s-1 sr-1
    emi = 4 * np.pi * n * integ # MeV^-1 s^-1 cm-3
    return E_gamma * E_gamma  * emi # MeV cm-3 s-1