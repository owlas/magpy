# <img src="./img/magpy.png" height="80px" alt="magpy">


magpy is a C++ accelerated python package for simulating systems of
magnetic nanoparticles.

**Current status**

[![CircleCI](https://circleci.com/gh/owlas/magpy.svg?style=svg)](https://circleci.com/gh/owlas/magpy)
[![Documentation Status](https://readthedocs.org/projects/magpy/badge/?version=latest)](http://magpy.readthedocs.io/en/latest/?badge=latest)
[![Stories in Ready](https://badge.waffle.io/owlas/magpy.png?label=ready&title=Ready)](https://waffle.io/owlas/magpy)
[![conda-version](https://anaconda.org/owlas/magpy/badges/version.svg)](https://anaconda.org/owlas/magpy/)
![conda-license](https://anaconda.org/owlas/magpy/badges/license.svg)
[![conda-downloads](https://anaconda.org/owlas/magpy/badges/downloads.svg)](https://anaconda.org/owlas/magpy/)
[![conda-link](https://anaconda.org/owlas/magpy/badges/installer/conda.svg)](https://anaconda.org/owlas/magpy/)
[![DOI](https://zenodo.org/badge/76475957.svg)](https://zenodo.org/badge/latestdoi/76475957)

**Features**

 - C++ accelerated time-integration
 - Stochastic Landau-Lifshitz-Gilbert equation
 - Explicit Heun scheme integration
 - Implicit Midpoint method
 - Parallelism at the highest level. Use the power of embarrassingly
   parallel!
 - Thermal activation model for single particles
 - Energy dissipation and SAR calculations

Join the chat at:

[![Join the chat at https://gitter.im/magpy-users/Lobby](https://badges.gitter.im/magpy-users/Lobby.svg)](https://gitter.im/magpy-users/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## Documentation

Getting started guides, example code, tutorials, and the API documentation are
available at http://magpy.readthedocs.io

They can also be built locally with sphinx. Requirements are found in the `enviornment.yml` file:

``` shell
    $ cd docs
    $ cat environment.yml
    $ make html
```

## Installation

Detailed instructions can be found at http://magpy.readthedocs.io

## Tests

### Unit tests

info soon

### Numerical tests

info soon

### Physics tests

info soon

## Contributing

 - Open an issue
 - Make a pull request
 - Join us on gitter
