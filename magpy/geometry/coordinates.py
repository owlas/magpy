import numpy as np

from .arkus import ARKUS


def arkus_cluster_coordinates(n_particles, configuration_id, R):
    """Coordinates of particles in an Arkus cluster.

    Returns an array of coordinates for each particle in an Arkus
    cluster of size `n_particles` with a specified `configuration_id`.
    Each configuration id represents a cluster of `n_particles` with a
    different arrangement.

    See :mod:`magpy.geometry.arkus` for more information on Arkus geometries
    and available configurations.

    Args:
        n_particles (int): cluster size
        configuration_id (int): configuration id of the Arkus cluster
            see [link] for more information on available configurations for
            each cluster size
        R (float): distance between particles

    Returns:
        np.ndarray: shape `(n_particles, 3)` array of the coordinates
    """
    return ARKUS[n_particles][configuration_id] * R


def arkus_cluster_random_configuration_id(n_particles):
    """A randomly drawn configuration id for an Arkus cluster.

    Returns a random configuration id for an Arkus cluster of a specified
    size. Each configuration contains `n_particles` in a different arrangement.

    See :mod:`magpy.geometry.arkus` for more information on Arkus geometries
    and available configurations.

    Args:
        n_particles (int): Arkus cluster size

    Returns:
        int: a random configuration id
    """
    return np.random.randint(len(ARKUS[n_particles]))


def arkus_random_cluster_coordinates(n_particles, R):
    """Coordinates of particles in a random Arkus cluster configuration.

    Returns an array of coordinates for particles in an Arkus cluster
    of size `n_particles` with a random configuration. Each configuration
    has a different arrangement of particles. See [link] for info.

    See :mod:`magpy.geometry.arkus` for more information on Arkus geometries
    and possible configurations.

    Args:
        n_particles (int): Arkus cluster size
        R (float): point to point Euclidean distance between coordinates

    Returns:
        np.ndarray: shape `(n_particles,3)` array of coordinates of particles
    """
    configuration = arkus_cluster_random_configuration_id(n_particles)
    return arkus_cluster_coordinates(n_particles, configuration, R)


def chain_coordinates(n_particles, R, direction=np.array([0,0,1])):
    """Coordinates of particles along a straight chain.

    Returns an array of coordinates for particles arranged in a
    perfectly straight chain and regularly spaced. The chain can have
    any direction and number of particles but will start at the origin.

    Example:
        ..code-block:: python

            >>> chain_coordinates(3, 2.5, direction=[1,0,0])
            np.array([0,0,0],
                     [2.5,0,0],
                     [5,0,0])
            >>> # Only unit vector of direction is used:
            >>> chain_coordinates(2, 3.0, direction=[1,2,0])
            np.array([0,0,0],
                     [])

    Todo:
        * Fix the example

    Args:
        n_particles (int): size of the particle chain
        R (float): point to point Euclidean distance between points
        direction (np.ndarray, optional): direction in 3d space to
            construct the chain. The magnitude of the direction is
            has no effect, only its unit direction.
            Default value is the `z`-axis `np.array([0,0,1])`
    Returns:
        np.ndarray: shape `(n_particles,3)` array of coordinates of particles
    """
    unit_direction = np.atleast_2d(direction / np.linalg.norm(direction))
    multipliers = np.atleast_2d(np.arange(n_particles))
    unscaled_chain_coordinates = multipliers.T.dot(unit_direction)
    return R * unscaled_chain_coordinates
