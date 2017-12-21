from magpy import Model

def test_model():
    single_particle = Model(
        radius = [12e-9],
        anisotropy = [4e4],
        anisotropy_axis = [[0., 0., 1.]],
        magnetisation_direction = [[1., 0., 0.]],
        location = [[0., 0., 0.]],
        damping = 0.1,
        temperature = 300.,
        magnetisation = 400e3
    )

def test_domodel():
    dom = Model(
        radius = 12e-9,
        anisotropy = 4e4,
        initial_probabilities = [0.5, 0.5],
        magnetisation = 400e3,
        damping = 0.1,
        temperature = 300
    )
