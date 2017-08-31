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

# Interface to field function generators
# These functions create and return std::function objects
cdef extern from "field.hpp" namespace "field":
    cpdef enum options:
        SINE,
        SQUARE,
        CONSTANT

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
        const long seed,
        const options field_shape,
        const double field_amplitude,
        const double field_frequency
    )

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
    int seed,
    str field_shape='constant',
    double field_amplitude=0.0,
    double field_frequency=0.0 ):

    cdef vector[double] c_radius, c_anisotropy
    cdef vector[d3] c_anisotropy_axis, c_magnetisation_direction, c_location

    c_radius = arr_to_vec(&radius[0], radius.shape[0])
    c_anisotropy = arr_to_vec(&anisotropy[0], radius.shape[0])
    c_anisotropy_axis = arr_to_d3vec(&anisotropy_axis[0,0], anisotropy_axis.shape[0])
    c_magnetisation_direction = arr_to_d3vec(&magnetisation_direction[0,0], magnetisation_direction.shape[0])
    c_location = arr_to_d3vec(&location[0,0], location.shape[0])

    lookup = {'constant': CONSTANT,
              'sine': SINE,
              'square': SQUARE}
    field_code = lookup[field_shape]

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
        seed,
        field_code,
        field_amplitude,
        field_frequency)

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
        self.time = results[0].time
        self.field = results[0].field

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
            location, magnetisation, damping, temperature, field_shape='constant', field_frequency=0.0, field_amplitude=0.0 ):
        self.radius = radius
        self.anisotropy = anisotropy
        self.anisotropy_axis = anisotropy_axis
        self.magnetisation_direction = magnetisation_direction
        self.location = location
        self.magnetisation = magnetisation
        self.damping = damping
        self.temperature = temperature
        self.volume = np.array([4./3*np.pi * r**3 for r in self.radius])
        self.field_shape = field_shape
        self.field_frequency = field_frequency
        self.field_amplitude = field_amplitude

    def describe(self):
        print('System stability:')
        print(' '.join([
            '{:.1f}'.format(k*v / KB / self.temperature)
            for k,v in zip(self.anisotropy, self.volume)
        ]))

    def draw_initial_condition(self):
        if callable(self.magnetisation_direction):
            return self.magnetisation_direction()
        return self.magnetisation_direction

    def simulate(self, end_time, time_step, max_samples, seed=1001, renorm=False, interactions=True, implicit_solve=True):
        res = simulate(
            self.radius, self.anisotropy, self.anisotropy_axis,
            self.draw_initial_condition(), self.location, self.magnetisation,
            self.damping, self.temperature, renorm, interactions, implicit_solve,
            time_step, end_time, max_samples, seed,
            self.field_shape, self.field_amplitude, self.field_frequency)
        return Results(**res)


    def simulate_ensemble(self, end_time, time_step, max_samples, seeds, renorm=False, interactions=True, n_jobs=1, implicit_solve=False):
        results = Parallel(n_jobs, verbose=5)(
            delayed(self.simulate)(end_time, time_step, max_samples, seed, renorm, interactions, implicit_solve,)
            for seed in seeds
        )
        return EnsembleResults(results)
