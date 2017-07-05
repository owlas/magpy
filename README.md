# Oh Moma! [![CircleCI](https://circleci.com/gh/owlas/moma.svg?style=svg)](https://circleci.com/gh/owlas/moma) [![Stories in Ready](https://badge.waffle.io/owlas/moma.png?label=ready&title=Ready)](https://waffle.io/owlas/moma)

A **M**odern **o**pen-source **m**agnetics simulation package.. **a**gain.

Simulate magnetic nano-particles with ease.

**Features**

 - Stochastic Landau-Lifshitz-Gilbert equation
 - Explicit Heun scheme integration
 - OpenMP at the highest level. Never be ashamed of *embarrassingly
   parallel*
 - Json for config files and output
 - Real life logging!
 - You won't believe this is C++

Join the chat at:

[![gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/oh-moma)

## Installation

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

## Configuration

See the [example config file](configs/example.json) for an overview of the options.

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
