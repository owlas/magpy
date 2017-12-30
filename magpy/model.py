import numpy as np
from toolz import dicttoolz
from joblib import Parallel, delayed

# When building docs on read the docs we cannot build the cython modules
# So we mock those out here
import os
if 'READTHEDOCS' in os.environ:
    import sys
    from unittest.mock import MagicMock
    sys.modules['magpy.core'] = MagicMock()

from .core import simulate, simulate_dom
from .results import Results, EnsembleResults

class Model:
    """A cluster of interacting magnetic nanoparticles

    A magpy Model describes a cluster of magnetic nanoparticles. The cluster is defined by
    the material properties of the individual particles, their relative positions in 3D
    space, and an initial magnetisation vector. Each particle is modelled as by a single
    macrospin with a uniaxial anisotropy axis.

    The model may be simulated to obtain the time-dependent dynamics of the magnetisation
    vector of the individual particles within the cluster.

    The particle cluster is optionally subjected to a time-varying magnetic field, applied
    along the z-axis. The field can be constant or sine/square varying with a desired
    frequency and amplitude.

    Args:
        radius (list of double): radius of each spherical particle in the ensemble in meters
        anisotropy (list of double): anisotropy constant for the uniaxial anisotropy axis
            of each particle
        anisotropy_axis (list of ndarray[double,3]): unit axis for the direction of the
            anisotropy for each particle
        magnetisation_direction (list of ndarray[double,3]): initial direction of the
            magnetisation vector for each particle
        location (list of ndarray[double,3]): location of each particle in the cluster
            described by x,y,z coordinates
        magnetisation (double): saturation magnetisation of all particles in the cluster (ampres / meter).
            Saturation magnetisation cannot vary between particles.
        damping (double): the damping parameter for all particles in the cluster. Damping
            cannot vary between particles.
        temperature (double): the ambient temperature in Kelvin for the particle cluster.
        field_shape (str, optional): can be either 'constant', 'square' or 'sine' describing
            the time-varying shape of the alternating field. The field is always applied
            along the z-axis. Default is 'constant'
        field_frequency (double, optional): the frequency of the applied field in Hz.
            Default is 0Hz.
        field_amplitude (double, optional): the amplitude of the applied field in Ampres / meter.
            Default is 0A/m

    """
    def __init__(
            self, radius, anisotropy, anisotropy_axis, magnetisation_direction,
            location, magnetisation, damping, temperature, field_shape='constant', field_frequency=0.0, field_amplitude=0.0 ):
        self.radius = np.array(radius)
        self.anisotropy = np.array(anisotropy)
        self.anisotropy_axis = np.array(anisotropy_axis)
        self.magnetisation_direction = np.array(magnetisation_direction)
        self.location = np.array(location)
        self.magnetisation = magnetisation
        self.damping = damping
        self.temperature = temperature
        self.field_shape = field_shape
        self.field_frequency = field_frequency
        self.field_amplitude = field_amplitude

    def simulate(self, end_time, time_step, max_samples, seed=1001, renorm=False, interactions=True, implicit_solve=True, implicit_tol=1e-9):
        """Simulate the dynamics of the particle cluster

        Simulate the time-varying dynamics of the cluster of
        interacting macrospins.  The time-varying dynamics are
        described by the Landau-Lifshitz-Gilbert stochastic
        differential equation, which is integrated using an explicit
        or implicit numerical scheme.

        In order to save memory, the user is required to specify the `max_samples`.
        The output of the time-integrator is up/downsampled to `max_samples` regularly
        spaced intervals using a first-order-hold interpolation. This is useful
        for long simulations with very small time steps that, without downsampling,
        would produce GBs of data very quickly.

        There are two time-integration schemes available:
          - a fully implicit midpoint scheme :cpp:function:`integrator::implicit_midpoint`
          - an explicit predictor-corrector method (Heun scheme) :cpp:function:`integrator::heun`

        Args:
            end_time (float): time to end the simulation (in seconds)
            time_step (float): time step for time-integration solver
            max_samples (int): number of regularly spaced samples of the output
            seed (int, optional): default value is 1001. The random seed for randam
                number generation of the thermal noise. Set for reproducible results.
            renorm (bool, optional): default is False. If True the magnetisation
                of each particle is rescaled (using the 2-norm) to unity at every
                time step.
            interactions (bool, optional): default is True. If False the interactions
                between particles are switched off.
            implicit_solve (bool, optional): default is True. If True a fully-implicit
                stochastic solver is used. If False the explicit Heun scheme is used.
            implicit_tol (float, optional): if using the implicit solver `implicit_tol`
                sets the tolerance of the internal Newton-Raphson method. Default
                is 1e-9
        Returns:
            magpy.Results: a :py:class:`magpy.results.Results` object containing
                the time-dependent magnetisation of the particle system.
        """
        res = simulate( self.radius, self.anisotropy, self.anisotropy_axis,
                        self.magnetisation_direction, self.location, self.magnetisation,
                        self.damping, self.temperature, renorm, interactions,
                        implicit_solve, time_step, end_time, max_samples, seed,
                        self.field_shape, self.field_amplitude,
                        self.field_frequency, implicit_tol)
        return Results(**res)


class EnsembleModel:
    """Ensemble of particle clusters

    The EnsembleModel class represents a non-interacting ensemble of
    particle clusters. It aims to provide a more user-friendly
    alternative than handling a large number of magpy.Model instances
    manually.

    Every member of the ensemble is copied from a base `magpy.Model`
    and is updated from a list of varying parameters. Parameters that
    are not specified as keyword arguments will be identical for every
    member of the ensemble and equivalent to that parameter's value in
    the base model.

    Parameters that should vary for each particle are specified as a
    keyword argument whose value is a list of parameters of length `N`
    (where the i'th value of the list correpsonds to the parameter's value
    for the i'th member of the cluster)

    Args:
        N (int): number of clusters in the ensemble
        base_model (magpy.Model): the base model provides the default parameters
            for every member of the ensemble.
        **kwargs: each argument may be a magpy.Model parameter and a corresponding
            list of `N` parameter values, which override the base model parameters
            for the i'th member of the ensemble.

    """
    def __init__(self, N, base_model, **kwargs):
        self.ensemble_size = N
        self.base_model_params = base_model.__dict__
        self.model_params = [
            dicttoolz.merge(
                self.base_model_params, {key: kwargs[key][i] for key in kwargs}
            ) for i in range(self.ensemble_size)
        ]
        self.models = [
            Model(**params) for params in self.model_params
        ]


    def simulate(self, end_time, time_step, max_samples, random_state,
                 renorm=False, interactions=True, n_jobs=1,
                 implicit_solve=True, implicit_tol=1e-9):
        """Simulate the dynamics of an ensemble of particle clusters

        Simulate the time-varying dynamics of an ensemble of particle
        clusters of interacting macrospins.  The time-varying dynamics
        are described by the Landau-Lifshitz-Gilbert stochastic
        differential equation, which is integrated using an explicit
        or implicit numerical scheme.

        In order to save memory, the user is required to specify the
        `max_samples`.  The output of the time-integrator is
        up/downsampled to `max_samples` regularly spaced intervals
        using a first-order-hold interpolation. This is useful for
        long simulations with very small time steps that, without
        downsampling, would produce GBs of data very quickly.

        There are two time-integration schemes available:
          - a fully implicit midpoint scheme :cpp:function:`integrator::implicit_midpoint`
          - an explicit predictor-corrector method (Heun scheme) :cpp:function:`integrator::heun`

        Args:
            end_time (float): time to end the simulation (in seconds)
            time_step (float): time step for time-integration solver
            max_samples (int): number of regularly spaced samples of the output
            random_state (int, optional): the state is used to generate seeds for
                each of the individual simulations. Set for reproducible results.
            renorm (bool, optional): default is False. If True the magnetisation
                of each particle is rescaled (using the 2-norm) to unity at every
                time step.
            interactions (bool, optional): default is True. If False the interactions
                between particles are switched off.
            implicit_solve (bool, optional): default is True. If True a fully-implicit
                stochastic solver is used. If False the explicit Heun scheme is used.
            implicit_tol (float, optional): if using the implicit solver `implicit_tol`
                sets the tolerance of the internal Newton-Raphson method. Default
                is 1e-9
        Returns:
            magpy.Results: a :py:class:`magpy.results.Results` object containing
                the time-dependent magnetisation of the particle system.

        """
        np.random.seed(random_state)
        sim_seeds = np.random.randint(np.iinfo(np.int32).max, size=self.ensemble_size)
        results = Parallel(n_jobs, verbose=5)(
            delayed(model.simulate)(end_time, time_step, max_samples, seed, renorm, interactions, implicit_solve, implicit_tol)
            for seed, model in zip(sim_seeds, self.models)
        )
        return EnsembleResults(results)


class DOModel:
    """A probabilistic model of a single particle

    A magpy DOModel is a probabilitic model of a single magnetic nanoparticle with a
    uniaxial anisotropy axis. The model has just two possible states: up and down. The
    model is defined by the material properties of the particle, an external field
    (applied along the anisotropy axis), and an initial probability vector (length 2).

    The model is simulated to solve the probability of the system being up and down over
    time. The particle is optionally subjected to a time-varying field along the
    anisotropy axis. The available field types and applicable parameters:
     - `constant`: a constant (time-invariant) field. Specify `field_amplitude`
     - `sine`: a sinusoidal field. Specify `field_amplitude` and `field_frequency`
     - `square`: a square alternating field (switching).
       Specify `field_amplitude` and `field_frequency`
     - `square_f`: a square alternating field with a finite number of cosine Fourier
       series terms. Specify `field_amplitude`, `field_frequency` and `field_n_components`

    Args:
        radius (double): radius of the spherical particle
        anisotropy (double): anisotropy constant for the uniaxial anisotropy axis
        initial_probabilities (ndarray[double,2]): initial probability of the particle down
            and up state respectively.
        magnetisation (double): saturation magnetisation of all particles in the cluster (ampres / meter).
            Saturation magnetisation cannot vary between particles.
        damping (double): the damping constant for the particle.
        temperature (double): the ambient temperature in Kelvin for the particle.
        field_shape (str, optional): can be either 'constant', 'square', 'square_f' or 'sine' describing
            the time-varying shape of the alternating field. The field is always applied
            along the anisotropy axis. Default is 'constant'
        field_frequency (double, optional): the frequency of the applied field in Hz.
            Default is 0Hz.
        field_amplitude (double, optional): the amplitude of the applied field in Ampres / meter.
            Default is 0A/m
        field_n_components (int, optional): applies for `field_shape=='square_f'` only.
            The number of cosine Fourier series components to use for square wave.
            Default is 1

    """
    def __init__(
            self, radius, anisotropy, initial_probabilities, magnetisation,
            damping, temperature, field_shape='constant', field_frequency=0.0,
            field_amplitude=0.0, field_n_components=1):
        self.radius = radius
        self.volume = 4./3 * np.pi * self.radius**3
        self.anisotropy = anisotropy
        self.initial_probabilities = np.array(initial_probabilities)
        self.magnetisation = magnetisation
        self.damping = damping
        self.temperature = temperature
        self.field_shape = field_shape
        self.field_frequency = field_frequency
        self.field_amplitude = field_amplitude
        self.field_n_components = field_n_components

    def simulate(self, end_time, time_step, max_samples):
        """Simulate the state probabilities for the particle

        Simulate the time-varying probabilities of the up/down states of the particle. The
        system is described by a master equation, which is defined by the transition rates
        between the up and down state. The master equation is solved numerically using an
        explicit RK45 solver.

        In order to save memory, the user is required to specify the `max_samples`.
        The output of the time-integrator is up/downsampled to `max_samples` regularly
        spaced intervals using a first-order-hold interpolation. This is useful
        for long simulations with very small time steps that, without downsampling,
        would produce GBs of data very quickly.

        Args:
            end_time (float): time to end the simulation (in seconds)
            time_step (float): time step for time-integration solver
            max_samples (int): number of regularly spaced samples of the output
        Returns:
            magpy.Results: a :py:class:`magpy.results.Results` object containing
                the time-dependent magnetisation of the particle.

        """
        results = simulate_dom(
            self.initial_probabilities, self.volume,
            self.anisotropy, self.temperature, self.magnetisation,
            self.damping, time_step, end_time, max_samples,
            self.field_shape, self.field_amplitude, self.field_frequency,
            self.field_n_components
        )
        return Results(**results)
