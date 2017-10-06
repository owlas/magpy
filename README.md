![magpy](./img/magpy.png)

# magpy [![CircleCI](https://circleci.com/gh/owlas/magpy.svg?style=svg)](https://circleci.com/gh/owlas/magpy) [![Stories in Ready](https://badge.waffle.io/owlas/moma.png?label=ready&title=Ready)](https://waffle.io/owlas/moma)

magpy is a C++ accelerated python package for simulating systems of
magnetic nanoparticles.

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

[![gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/oh-moma)

## Documentation

Documentation and example usage can be found here at read the docs

Example notebooks are in this directory

Alternatively build the docs yourself with sphinx

The C++ API docs are built locally using doxygen.

## Installation

Include the following installation procedures:
 - Conda package
 - Build from scratch
 - Intel and non intel
 - Link to install scripts

Moma requires LAPACK and
BLAS routines. Run the [makefile](makeflie) with intel complier:

``` shell
$ make
```

Make with gcc (will need version >=4.9):

``` shell
$ make CXX=g++-4.9
```

### Mac OSX

The easiest way to obtain LACPACK/BLAS is through
[homebrew](http://brew.sh/) by install openblas, which comes with
both. Run the following command:

``` shell
$ brew install homebrew/science/openblas
```

You'll need to link to link to the libraries when running the make command, e.g:

``` shell
$ make CXX=g++-4.9 LDFLAGS=-L/usr/local/opt/openblas/lib CXXFLAGS=-I/usr/local/opt/openblas/include
```

## Tests

The fast unit tests can be run with

``` shell
$ make tests
$ ./test
```

The full test suite includes numerical simulations of convergence. These take a
long time to execute (~5mins). Test suite results are available in `test/output`.

``` shell
$ make run-tests
$ cd test/output
```

## Dependencies

**LAPACK**: On Ubuntu:

``` shell
$ sudo apt install liblapack-dev
```

## Additional notes

Compiling Cython? Make sure to

``` shell
$ export CC=icc
$ export CXX=icpc
```

## Getting started



## Contributing

 - Open an issue
 - Make a pull request
 - Join us on gitter

## Tests

### Unit tests

If you downloaded with conda you can run conda test

Otherwise install pytest and pytest-cpp dependencies then run it on
this dir

### Numerical tests

Tests for the numerical methods are slow. They can be run like this

### Physics tests

We test the physics in these notebooks...
