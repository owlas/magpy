#include "../include/field.hpp"
#define _USE_MATH_DEFINES
#include<cmath>

double field::constant( const double h, const double t )
{
    return h;
}

double field::sinusoidal( const double h, const double f, const double t )
{
    return h*std::sin( 2*M_PI*f*t );
}
double field::square( const double h, const double f, const double t )
{
    return h*( int( t*f*2 )%2 ? -1 : 1 );
}

double field::square_fourier( const double h,
                              const double f,
                              const size_t n_components,
                              double t )
{
    double field=0;
    for( unsigned int k=1; k<n_components+1; k++ )
        field += std::sin( 2*M_PI*( 2*k - 1 )*f*t ) / ( 2*k-1 );
    field *= 4/M_PI * h;
    return field;
}
