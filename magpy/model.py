from toolz import dicttoolz
from joblib import Parallel, delayed

from .core import simulate
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
        self.radius = radius
        self.anisotropy = anisotropy
        self.anisotropy_axis = anisotropy_axis
        self.magnetisation_direction = magnetisation_directpion
        self.location = location
        self.magnetisation = magnetisation
        self.damping = damping
        self.temperature = temperature
        self.field_shape = field_shape
        self.field_frequency = field_frequency
        self.field_amplitude = field_amplitude

    def simulate(self, end_time, time_step, max_samples, seed=1001, renorm=False, interactions=True, implicit_solve=True, implicit_tol=1e-9):
        """Simulate the dynamics of the particle cluster

        Simulate the time-varying dynamics of the cluster of interacting macrospins.
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
        self.ensemble_size = ensemble_size
        self.base_model_params = base_model.__dict__
        self.model_params = [
            dicttoolz.update(
                base_model_params, {key: kwargs[key][i] for key in kwargs}
            ) for i in range(ensemble_size)
        ]
        self.models = [
            Model(**params) for params in self.model_params
        ]


    def simulate_ensemble(self, end_time, time_step, max_samples, seeds, renorm=False, interactions=True, n_jobs=1, implicit_solve=False, implicit_tol=1e-9):
        results = Parallel(n_jobs, verbose=5)(
            delayed(self.simulate)(end_time, time_step, max_samples, seed, renorm, interactions, implicit_solve, implicit_tol)
            for seed in seeds
        )
        return EnsembleResults(results)
