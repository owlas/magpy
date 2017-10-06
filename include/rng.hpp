/** @file rng.hpp
 * @brief Class for holding different random number generators
 *
 * @author Oliver W. Laslett
 *
 */
#ifndef RNG_H
#define RNG_H

#include <random>


/// Abstract class for random number generators.
class Rng
{
public:
    /// Get a single random number
    /**
     * @returns a single draw from the random number generator
     */
    virtual double get()=0;
};


/// Uses Mersenne Twister to generate normally distributed values.
/**
 * Random numbers have a mean of zero and a user specified standard
 * deviation.
 */
class RngMtNorm : public Rng
{
public:
    RngMtNorm( const unsigned long int seed, const double std );
    double get();
private:
    std::mt19937_64 generator; ///< A Mersenne twister generator instance
    std::normal_distribution<double> dist; ///< A normal distribution instance
};


/// Generate normally distributed values with downsampling
/**
 * Uses the Mersenne Twister to generate normally distributed random
 * numbers. Down-samples the stream of random numbers by summing
 * consecutive draws along each dimension.
 * Function is usually used for generating coarse Wiener processes
 * @deprecated Original purpose of this class (for Wiener processes)
 * has been replaced with the `RngArray` class.
 */
class RngMtDownsample : public Rng
{
public:
    RngMtDownsample( const unsigned long int seed, const double std,
                     const size_t dim, const size_t down_factor );
    double get();
private:
    void downsample_draw();
    std::mt19937_64 generator; ///< A Mersenne Twister generator instance
    std::normal_distribution<double> dist; ///< A normal distribution instance
    int current_dim; ///< Stores the current state of the output dimenstion
    std::vector<double> store; ///< Stores consecutive random numbers
    const size_t D; ///< The number of dimensions required
    const size_t F; ///< The number of consecutive random numbers to downsample
};

/// Provides an `Rng` interface to a predefined array of numbers
class RngArray : public Rng
{
public:
    RngArray( const double *arr, size_t arr_length, size_t stride=1 );
    double get();
private:
    unsigned int i=0; ///< Internal state
    const size_t max; ///< Maximum number of draws available
    const double *arr; ///< Pointer to the predefined array of numbers
    const size_t stride; ///< Number of values to stride for each draw
};
#endif
