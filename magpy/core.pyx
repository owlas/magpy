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

# Constants
cdef extern from 'constants.hpp' namespace 'constants':
    const double KB, MU0, GYROMAG

cpdef get_KB():
    return KB
cpdef get_mu0():
    return MU0
cpdef get_gamma():
    return GYROMAG

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
        const bool interactions,
        const bool use_implicit,
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
        bool interactions,
        bool use_implicit,
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
        interactions,
        use_implicit,
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

# -------------
# Model object
# -------------
import matplotlib.pyplot as plt

class EnsembleResults:
    def __init__(self, results):
        self.results = results

    def magnetisation(self, direction='z'):
        return [res.magnetisation(direction) for res in self.results]

    def ensemble_magnetisation(self, direction='z'):
        return np.sum(self.magnetisation(direction), axis=0) / len(self.results)

    def final_state(self):
        return [res.final_state() for res in self.results]

class Results:
    def __init__(self, time, field, x, y, z, N):
        self.time = time
        self.field = field
        self.x = x
        self.y = y
        self.z = z
        self.N = N

    def plot(self):
        fg, axs = plt.subplots(nrows=self.N)
        if self.N==1:
            axs = [axs]
        for idx in range(self.N):
            axs[idx].plot(self.time, self.x[idx], label='x')
            axs[idx].plot(self.time, self.y[idx], label='y')
            axs[idx].plot(self.time, self.z[idx], label='z')
            axs[idx].legend()
            axs[idx].set_title('Particle {}'.format(idx))
            axs[idx].set_xlabel('Reduced time [dimless]')
            fg.tight_layout()
        return fg

    def magnetisation(self, direction='z'):
        return np.sum([vals for vals in getattr(self, direction).values()], axis=0)

    def final_state(self):
        return {
            'x': {k:v[-1] for k,v in self.x.items()},
            'y': {k:v[-1] for k,v in self.y.items()},
            'z': {k:v[-1] for k,v in self.z.items()}
        }

from joblib import Parallel, delayed

class Model:
    def __init__(
            self, radius, anisotropy, anisotropy_axis, magnetisation_direction,
            location, magnetisation, damping, temperature ):
        self.radius = radius
        self.anisotropy = anisotropy
        self.anisotropy_axis = anisotropy_axis
        self.magnetisation_direction = magnetisation_direction
        self.location = location
        self.magnetisation = magnetisation
        self.damping = damping
        self.temperature = temperature
        self.volume = np.array([4./3*np.pi * r**3 for r in self.radius])

    def describe(self):
        print('System stability:')
        print(' '.join([
            '{:.1f}'.format(k*v / KB / self.temperature)
            for k,v in zip(self.anisotropy, self.volume)
        ]))

    def simulate(self, end_time, time_step, max_samples, seed=1001, renorm=False, interactions=True, implict_solve=True):
        res = simulate(
            self.radius, self.anisotropy, self.anisotropy_axis,
            self.magnetisation_direction, self.location, self.magnetisation,
            self.damping, self.temperature, renorm, interactions, implict_solve,
            time_step, end_time, max_samples, seed)
        return Results(**res)


    def simulate_ensemble(self, end_time, time_step, max_samples, seeds, renorm=False, interactions=True, n_jobs=1):
        results = Parallel(n_jobs)(
            delayed(self.simulate)(end_time, time_step, max_samples, seed, renorm, interactions)
            for seed in seeds
        )
        return EnsembleResults(results)
