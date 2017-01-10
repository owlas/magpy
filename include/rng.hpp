// rng.hpp
// Class for holding different random number generators
//
// Oliver W. Laslett
// O.Laslett@soton.ac.uk
#ifndef RNG_H
#define RNG_H

#include <random>

/*
  Abstract class specifies that all random number
  generators have a function get that returns one random
  number
*/
class Rng
{
public:
    virtual double get()=0;
};

/*
  Uses the mersenne twister to generate standard normally
  distributed random numbers
*/
class RngMtNorm : public Rng
{
public:
    RngMtNorm( const unsigned long int seed, const double std );
    double get();
private:
    std::mt19937_64 generator;
    std::normal_distribution<double> dist;
};

/*
  Uses the MT to generate normally distributed random numbers.
  Down-samples the path by summing consecutive draws in each dimension.
  Utility for generating coarser Wiener processes.
*/
class RngMtDownsample : public Rng
{
public:
    RngMtDownsample( const unsigned long int seed, const double std,
                     const size_t dim, const size_t down_factor );
    double get();
private:
    void downsample_draw();
    std::mt19937_64 generator;
    std::normal_distribution<double> dist;
    int current_dim;
    std::vector<double> store;
    const size_t D;
    const size_t F;
};

/*
  Provides an interface to a preallocated array of random numbers.
  Specify the array and it's length.
  The first call to .get() will return the value at the 0th index.
  Each subsequent call to .get() will stride the array (default stride
  is 1) and return that value.
  A call to .get() after the end of the array will result in an
  error.
*/
class RngArray : public Rng
{
public:
    RngArray( const double *arr, size_t arr_length, size_t stride=1 );
    double get();
private:
    unsigned int i=0;
    const size_t max;
    const double *arr;
    const size_t stride;
};
#endif
