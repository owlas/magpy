# Build magpy library with g++
# Bake in the prefix search path for finding libs
export CPATH=$PREFIX/include
make CXX=g++ LDFLAGS="-L$PREFIX/lib -Wl,-rpath=$PREFIX/lib" libmoma.so

# Install by copying into the prefix
cp libmoma.so $PREFIX/lib

# Build tests against the library
make CXX=g++ test/tests

# Build python interface
CXX=g++ pip install .
