Installation
============

The easiest way to install magpy is using the conda repositories. The
only requirement is that you have anaconda or miniconda installed on
your system. Alternatively, you can build the C++ and python code from
source. You might want to do this if you only want the C++ library or
if you would like to access the Intel accelerated code.

Conda
-----

Note packages only exist for Linux and Mac OSX.

1. Go to https://conda.io/miniconda.html and download the Miniconda
   package manager
2. Create a new conda environment and install magpy into it:

   .. code-block:: bash

      $ conda create -n <env_name> python=3
      $ source activate <env_name>
      $ conda install -c owlas magpy
3. Launch python and import magpy

   .. code-block:: bash

      $ python
      $ >>> import magpy

From source
-----------

The instructions below will guide you through the process of building
the code from source on **Linux**.

1. Clone the magpy code

   .. code-block:: bash

      $ git clone https://github.com/owlas/magpy
2. You'll need a C++11 compatible compiler (g++>=4.9 recommended)
3. You will also the LAPACK and BLAS libraries. On Debian systems
   these can be obtained through the `apt repositories <https://packages.ubuntu.com/trusty/liblapacke-dev>`_

   .. code-block:: bash

      $ apt install liblapacke-dev
4. Build the magpy C++ library with with your compiler as ``<CXX>``

   .. code-block:: bash

      $ cd magpy
      $ make CXX=<CXX> libmoma.so
5. You can build and run the tests from the same makefile

   .. code-block:: bash

      $ cd magpy
      $ make CXX=<CXX> run-tests
6. To build the python interface you'll need to obtain all the python
   dependencies in the ``requirements.txt`` file.
7. Once you have all of the dependencies you can install magpy

   .. code-block:: bash

      $ cd magpy
      $ CXX=<CXX> pip install .
      $ python
      $ >>> import magpy


From source (Intel compilers)
-----------------------------

Magpy has been optimised for Intel architectures and you can take
advantage of this by taking a few extra steps:

1. Clone the magpy code

   .. code-block:: bash

      $ git clone https://github.com/owlas/magpy
2. Ensure you have the Intel compilers in your path (``icc`` and
   ``icpc``)
3. Tell magpy where to find your MKL files

   .. code-block:: bash

      $ export MKLROOT=/path/to/mkl/install/directory
4. You will also the LAPACK and BLAS libraries. On Debian systems
   these can be obtained through the `apt repositories <https://packages.ubuntu.com/trusty/liblapacke-dev>`_

   .. code-block:: bash

      $ apt install liblapacke-dev
5. Build the magpy C++ library with the intel compilers. The correct
   build flags should be taken care of for you

   .. code-block:: bash

      $ cd magpy
      $ make CXX=icpc libmoma.so
6. You can build and run the tests from the same makefile

   .. code-block:: bash

      $ cd magpy
      $ make CXX=icpc run-tests
7. To build the python interface you'll need to obtain all the python
   dependencies in the ``requirements.txt`` file.
8. Once you have all of the dependencies you can install magpy

   .. code-block:: bash

      $ cd magpy
      $ CC=icc CXX=icpc pip install .
      $ python
      $ >>> import magpy
