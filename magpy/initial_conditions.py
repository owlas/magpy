import numpy as np
from transforms3d import quaternions

def random_point_on_unit_sphere():
    """Point randomly drawn from a uniform distribution on the unit sphere.

    Randomly picking a point on the unit sphere requires choosing the point
    so that each solid angle on the sphere is equally likely.

    Set `np.random.seed(seed)` for reproducible results.

    Returns:
        np.ndarray: a 3d array with the x,y,z coordinates of a random point"""
    theta = 2.0*np.pi*np.random.rand()
    phi = np.arccos(1 - 2.0*np.random.rand())
    return np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])


def uniform_random_axes(N):
    """Random 3d vectors (axes) uniformly distributed on unit sphere.

    Returns `N` 3d unit vectors drawn from a uniform distribution
    over the unit sphere.

    Set `np.random.seed(seed)` for reproducible results.

    Args:
        N (int): number of random axes to draw.

    Returns:
        np.ndarray: a 2d array size (`Nx3`). Random 3d unit vectors on the
        uniform random sphere.
    """
    return np.array([random_point_on_unit_sphere() for i in range(N)])


def random_quaternion():
    """Random uniformly distributed quaternion

    A quaternion defines an 3d axis of rotation and a corresponding angle
    of rotation. A random quaternion is drawn by first drawing a random
    rotation axis from the uniform random sphere and a uniform angle. Random
    quaternions are used for random rotation operations on arbitrary geometries.

    Returns:
        transforms3d.Quaternion: a random quaternion from a uniform distribution
    """
    return quaternions.axangle2quat(random_point_on_unit_sphere(),
                                    2.0*np.pi*np.random.randn())
