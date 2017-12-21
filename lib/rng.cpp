/** @file rng.cpp
 * @brief Class fr holding different random number generators
 *
 */
#include "../include/rng.hpp"
#include <stdexcept>

/// Default constructor for RngMtNorm
/**
 * @param[in] seed seed for random number generator
 * @param[in] std  the standard deviation of the normally distributed
 * numbers
 */
RngMtNorm::RngMtNorm( const unsigned long int seed, const double std )
    : Rng()
    , dist( 0, std )
    , generator( seed ) {}


/// Draw a single normally distributed value from the RNG
/**
 * @returns single normally distributed number
 */
double RngMtNorm::get() { return dist( generator ); }


/// Default constructor for RngMtDownsample
RngMtDownsample::RngMtDownsample( const unsigned long int seed, const double std,
                                  const size_t dim, const size_t down_factor )
    : Rng()
    , dist( 0, std )
    , generator( seed )
    , current_dim( dim )
    , store( dim, 0 )
    , D( dim )
    , F( down_factor ) {}

/// Get a single downsampled value from the random number generator
double RngMtDownsample::get()
{
    if( current_dim >= D )
    {
        downsample_draw();
        current_dim=0;
    }
    unsigned int tmp = current_dim++;
    return store[tmp];
}

void RngMtDownsample::downsample_draw()
{
    for( unsigned int i=0; i<D; i++ )
        store[i] = dist( generator );
    for ( unsigned int j=1; j<F; j++ )
        for( unsigned int i=0; i<D; i++ )
            store[i] += dist( generator );
};


/// Default constructor for RngArray
/**
 * @param _arr        a predefined array of random numbers
 * @param _arr_length length of the predefined array
 * @param _stride     the number of consecutive elements to stride for
 * each call to `.get()`
 */
RngArray::RngArray( const double *_arr, size_t _arr_length, size_t _stride )
    : Rng()
    , arr( _arr )
    , max( _arr_length )
    , stride( _stride ) {}


/// Get the next (possibly stridden) value from the array
/**
 * The first call will always return the value at the 0th index.
 * Each subsequent call will stride the array (default value is 1) and
 * return that value.
 * A call that extends beyond the end of the array will result in an
 * error.
 * @exception std::out_of_range when a call attempts to get a value
 * beyond the maximum length of the predefined array.
 */
double RngArray::get()
{
    if( i<max )
    {
        unsigned int tmp = i;
        i += stride;
        return arr[tmp];
    }
    else
        throw std::out_of_range( "Exceeded allocated random number array size" );
}
