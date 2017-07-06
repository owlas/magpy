# distutils: language=c++
import cython
import numpy as np
cimport numpy as np

from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libcpp cimport bool

# array of 3 doubles
cdef extern from "<array>" namespace "std" nogil:
    cdef cppclass d3 "std::array<double,3>":
        d3() except+
        double& operator[](size_t)

# unique_ptr array
cdef extern from "<memory>" namespace "std" nogil:
    cdef cppclass double_up "std::unique_ptr<double[]>":
        double& operator[](size_t) const

# Interface to results struct
cdef extern from "simulation.hpp" namespace "simulation":
    struct results:
        double_up mx
        double_up my
        double_up mz
        double_up field
        double_up time
        size_t N
        double energy_loss

# Interface to LLG simulation
cdef extern from "simulation.hpp" namespace "simulation":
    vector[results] full_dynamics(
        const vector[double] radius,
        const vector[double] anisotropy,
        const vector[d3] anisotropy_axis,
        const vector[d3] magnetisation_direction,
        const vector[d3] location,
        const double magnetisation,
        const double damping,
        const double temperature,
        const bool renorm,
        const double time_step,
        const double end_time,
        const size_t max_samples,
        const long seed )

# Needed to convert numpy arrays to std::vectors
cdef vector[double] arr_to_vec(double *arr, size_t N):
    cdef vector[double] vec
    vec.reserve(N)
    vec.assign(arr, arr+N)
    return vec

cdef vector[d3] arr_to_d3vec(double *arr, size_t N):
    cdef vector[d3] vec
    vec.resize(N)
    for i in range(N):
        for j in range(3):
            vec[i][j] = arr[3*i + j]
    return vec

# --------------
# PYTHON WRAPPER
# --------------
cpdef simulate(
        np.ndarray[double, ndim=1, mode='c'] radius,
        np.ndarray[double, ndim=1, mode='c'] anisotropy,
        np.ndarray[double, ndim=2, mode='c'] anisotropy_axis,
        np.ndarray[double, ndim=2, mode='c'] magnetisation_direction,
        np.ndarray[double, ndim=2, mode='c'] location,
        double magnetisation,
        double damping,
        double temperature,
        bool renorm,
        double time_step,
        double end_time,
        int max_samples,
        int seed ):
    cdef vector[double] c_radius, c_anisotropy
    cdef vector[d3] c_anisotropy_axis, c_magnetisation_direction, c_location

    c_radius = arr_to_vec(&radius[0], radius.shape[0])
    c_anisotropy = arr_to_vec(&anisotropy[0], radius.shape[0])
    c_anisotropy_axis = arr_to_d3vec(&anisotropy_axis[0,0], anisotropy_axis.shape[0])
    c_magnetisation_direction = arr_to_d3vec(&magnetisation_direction[0,0], magnetisation_direction.shape[0])
    c_location = arr_to_d3vec(&location[0,0], location.shape[0])

    cdef vector[results] res = full_dynamics(
        c_radius,
        c_anisotropy,
        c_anisotropy_axis,
        c_magnetisation_direction,
        c_location,
        magnetisation,
        damping,
        temperature,
        renorm,
        time_step,
        end_time,
        max_samples,
        seed )

    # Turn results into python results
    n_particles = radius.shape[0]
    pyresults = {
        'N': n_particles,
        'time': np.array([res[0].time[i] for i in range(max_samples)]),
        'field': np.array([res[0].field[i] for i in range(max_samples)]),
        'x': {
            i: np.array([res[i].mx[j] for j in range(max_samples)])
            for i in range(n_particles)
        },
        'y': {
            i: np.array([res[i].my[j] for j in range(max_samples)])
            for i in range(n_particles)
        },
        'z': {
            i: np.array([res[i].mz[j] for j in range(max_samples)])
            for i in range(n_particles)
        }
    }
    return pyresults

import matplotlib.pyplot as plt
def plot_results(res):
    fg, axs = plt.subplots(nrows=res['N'])
    for idx in range(res['N']):
        axs[idx].plot(res['time'], res['x'][idx], label='x')
        axs[idx].plot(res['time'], res['y'][idx], label='y')
        axs[idx].plot(res['time'], res['z'][idx], label='z')
        axs[idx].legend()
        axs[idx].set_title('Particle {}'.format(idx))
        axs[idx].set_xlabel('Reduced time [dimless]')
    fg.tight_layout()
    return fg
