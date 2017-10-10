# Build magpy library
make CXX=g++ libmoma.so
cp libmoma.so $PREFIX/lib

# Build tests
make CXX=g++ test/tests

# Build python interface
CXX=g++ pip install .
