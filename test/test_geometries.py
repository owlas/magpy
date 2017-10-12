import numpy as np

from magpy.geometry.coordinates import chain_coordinates, arkus_cluster_coordinates
from magpy.geometry.coordinates import arkus_cluster_random_configuration_id
from magpy.geometry.arkus import ARKUS


def test_chains():
    coords = chain_coordinates(3, 2.5, [1,0,0])
    ans = np.array([[0,0,0],[2.5,0,0],[5.0,0,0]])
    assert np.all(coords == ans)


def test_arkus_select():
    coords = arkus_cluster_coordinates(7, 3, 5e-9)
    ans = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.618033988749913, 0.0],
        [0.951056516295152, 0.309016994374951, 0.0],
        [-0.587785252292505, 0.809016994374957, -0.000000000000003],
        [0.951056516295155, 1.309016994374955, -0.000000000000003],
        [0.262865556059561, 0.809016994374957, -0.525731112119131],
        [0.262865556059561, 0.809016994374957, 0.525731112119127]
    ]) * 5e-9
    assert np.all(coords == ans)


def test_random_config_ids_are_valid():
    for n_particles in range(1,8):
        ids = [arkus_cluster_random_configuration_id(n_particles)
               for i in range(10000)]
        # Are all the ids valid
        assert(np.all([i in ARKUS[n_particles] for i in ids]))


def test_random_config_id_distribution_6_particle():
    ids = [arkus_cluster_random_configuration_id(6) for i in range(100000)]

    # check mean is correct with fairly weak tolerance (reduce chance of random fail)
    assert(np.abs(np.mean(ids) - 0.5) < 1e-2)
