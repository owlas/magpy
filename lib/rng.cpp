// rng.cpp
// implementations of random number generator interfaces
//
// Oliver W. Laslett
// O.Laslett@soton.ac.uk

#include "../include/rng.hpp"
#include <stdexcept>

/*
  Mersenne twister and normal distribution
*/
RngMtNorm::RngMtNorm( const unsigned long int seed, const double std )
    : Rng()
    , dist( 0, std )
    , generator( seed ) {}

double RngMtNorm::get() { return dist( generator ); }

/*
  Mersenne twister and normal distribution with down sampling in each dimension
*/
RngMtDownsample::RngMtDownsample( const unsigned long int seed, const double std,
                                  const size_t dim, const size_t down_factor )
    : Rng()
    , dist( 0, std )
    , generator( seed )
    , current_dim( dim )
    , store( dim, 0 )
    , D( dim )
    , F( down_factor ) {}

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


/*
  Wrapper to array of random numbers
*/
RngArray::RngArray( const double *_arr, size_t _arr_length, size_t _stride )
    : Rng()
    , arr( _arr )
    , max( _arr_length )
    , stride( _stride ) {}

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
