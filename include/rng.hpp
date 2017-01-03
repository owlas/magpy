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
  Provides an interface to a preallocated array of random numbers
*/
class RngArray : public Rng
{
public:
    RngArray( const double *arr, size_t arr_length );
    double get();
private:
    unsigned int i=0;
    const size_t max;
    const double *arr;
};

#endif
