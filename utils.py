from typing import Callable, Iterable
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dctn, fft, fft2, fftfreq
from scipy.interpolate import interpn
import scipy.sparse as sparse
from scipy.linalg import lu_factor, lu_solve
from numba import jit

def spA(n):
    
    C=-4*sparse.eye(n-1,format='csc')+sparse.diags([np.ones(n-2),np.ones(n-2)],[-1,1],format='csc')
    AA=sparse.kron(sparse.eye(n-1,format='csc'),C,format='csc')+sparse.kron(sparse.diags(np.ones(n-2),1,format='csc'),sparse.eye(n-1,format='csc'),format='csc')+sparse.kron(sparse.diags(np.ones(n-2),-1,format='csc'),sparse.eye(n-1,format='csc'),format='csc')
    return AA

# def finite_difference_2D(A, B, Gamma) : 
    
#     N = len(B)
#     # Gamma = sparse.diags(np.ravel([[factor[0, k]] * (N-2) for k in range(1,N-1)]), 0)

#     lu, piv = lu_factor(A - Gamma)
#     U = lu_solve((lu, piv),B[1:-1, 1:-1].flatten())

#     sols = np.zeros((len(B), len(B)))
#     sols[1:-1:,1:-1]  = U.reshape(N-2, N-2)

#     return sols

def finite_difference_2D(A, B, Gamma, a, b) : 
    
    
    N = len(B)
    h = (b - a) / N  
    lu = sparse.linalg.splu(A - Gamma)
    U = lu.solve(B.flatten())
    
    sols = np.zeros((len(B), len(B)))
    sols  = U.reshape(N, N)
    
    return sols


def compute_dct_coeff(mesh : Iterable) -> np.ndarray :
    """
    compute_dct_coeff  compute dct coeff for a 2D mesh

    Parameters
    ----------
    mesh : Iterable
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """

    coeffs = dctn(mesh,  type = 1, norm = 'forward')  
    coeffs[1:-1:,1:-1] = 2 * coeffs[1:-1:,1:-1]

    return coeffs

def compute_laplatian_values_with_DCT(mesh : Iterable, spectral_coeffs : Iterable) -> np.ndarray :
    """
    compute_laplatian_values_with_DCT compute the laplatian of a function given its dct coefficients

    Parameters
    ----------
    mesh : Iterable
        _description_
    spectral_coeffs : Iterable
        _description_

    Returns
    -------
    np.ndarray
        laplatian values of the function
    """    
    
    laplatian_coeffs = [np.polynomial.chebyshev.chebder(spectral_coeffs, m = 2, axis = 0), np.polynomial.chebyshev.chebder(spectral_coeffs, m = 2, axis = 1)]
    h_laplatian = np.sum([np.polynomial.chebyshev.chebval2d(*mesh,  laplatian_coeffs[0]), np.polynomial.chebyshev.chebval2d(*mesh,  laplatian_coeffs[1])], axis = 0)
    
    return h_laplatian

def compute_gradient_values_with_DCT(mesh : Iterable, spectral_coeffs : Iterable) -> np.ndarray :
    """
    compute_gradient_values_with_DCT compute the gradient of a function given its dct coefficients

    Parameters
    ----------
    mesh : Iterable
        _description_
    spectral_coeffs : Iterable
        _description_

    Returns
    -------
    np.ndarray
        gradient values of the function
    """    
    
    gradient_coeffs = [np.polynomial.chebyshev.chebder(spectral_coeffs, m = 1, axis = 0), np.polynomial.chebyshev.chebder(spectral_coeffs, m = 1, axis = 1)]
    h_gradient = [np.polynomial.chebyshev.chebval2d(*mesh,  gradient_coeffs[0]), np.polynomial.chebyshev.chebval2d(*mesh,  gradient_coeffs[1])]

    return h_gradient

def grids_transfer(mesh_1 : Iterable, mesh_2 : Iterable, values : Iterable) -> np.ndarray : 
    
    """
    grids_transfer transfer values from a regular grid to another (not necessarily regular)

    Parameters
    ----------
    mesh_1 : Iterable
        regular grid
    mesh_2 : Iterable
        another grid
    values : Iterable
        values on grid_1

    Returns
    -------
    np.ndarray
        interpolation of values on mesh_2
    """    
    
    x_i = interpn([mesh_1[0,:,-1], mesh_1[1,0]], values, xi = list(zip(mesh_2[0].ravel(), mesh_2[1].ravel())), method='linear')
    
    return  x_i.reshape(mesh_2[0].shape)

def leap_frog(Delta_t : float, mesh_spectral : Iterable, mesh_finite : Iterable, 
              h_1 : Iterable, h_2 : Iterable, A : Iterable, Gamma : Iterable,  g  : float = .1, 
              beta : float = .01, a : float = .1) :
    
    """
    leap_frog compute the leap frog scheme for the wave equation

    Parameters
    ----------
    Delta_t : float
        _description_
    h : Iterable
        _description_
    u : Iterable
        _description_
    v : Iterable
        _description_

    Returns
    -------
    _type_
        _description_
    """    
    
    factor_2 = 2 * beta / g * Delta_t # see equation I.1.2
    
    
    # interpolation on finite grids to spectral grids
    
    h_2_spectral = grids_transfer(mesh_finite, mesh_spectral, h_2)
    spectral_coeffs = compute_dct_coeff(h_2_spectral) 


    gradient_h = compute_gradient_values_with_DCT(mesh_finite, spectral_coeffs) 

    
    C = - factor_2 *  gradient_h[0]
    h_3 = finite_difference_2D(A, C, Gamma, -1, 1) + h_2  

    
    return h_3


def get_speed_from_height(h : Iterable, mesh_finite : Iterable, mesh_spectral : Iterable, g, f) : 
    
    """
    get_speed_from_height compute the speed from the height

    Parameters
    ----------
    h : Iterable
        height
    mesh : Iterable
        mesh
    g : float
        gravity
    f : float
        coriolis force

    Returns
    -------
    np.ndarray
        speed
    """    
    
    h_spectral = grids_transfer(mesh_finite, mesh_spectral, h)
    spectral_coeffs = compute_dct_coeff(h_spectral) 


    gradient_h = compute_gradient_values_with_DCT(mesh_finite, spectral_coeffs)
    
    return - f / g * gradient_h[1], f / g * gradient_h[0]

def compute_wave_vector_matrix(mesh : Iterable) : 
    """
    compute_wave_vector_matrix compute 2d wave vector matrix

    Parameters
    ----------
    mesh : Iterable
        2D wave height

    Returns
    -------
    _type_
        _description_
    """
    wave_vector_matrix : Iterable = fft2(mesh)
    
    return np.mean(wave_vector_matrix, axis = 0), np.mean(wave_vector_matrix, axis = 1)

def compute_frequency_dependence_wave_vector(mesh_wave : Iterable) :  
    """
    compute_frequency_dependence_wave_vector compute frequency dependence of wave vector

    Parameters
    ----------
    mesh_wave : Iterable
        2D wave height

    Returns
    -------
    _type_
        _description_
    """    
    wave_vector_matrix_over_time_x = np.array([compute_wave_vector_matrix(x) for x in mesh_wave])
    k_x = wave_vector_matrix_over_time_x[:,0]
    k_y = wave_vector_matrix_over_time_x[:,1]
    t = np.arange(len(mesh_wave)) * 10 ** (-3)
    freq = fftfreq(len(t), d = 10 ** (-3))
    w_x = fft(k_x)
    w_y = fft(k_y)
    
    return freq, (w_x, w_y), (k_x, k_y)