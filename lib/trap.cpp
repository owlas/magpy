// trap.cpp
// Implementation of trapezoidal integration scheme
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#include "../include/trap.hpp"
double trap::trapezoidal( double *x, double *y, size_t N )
{
    double sum;
    unsigned int i;
    for( sum=0.0, i=1; i<N; i++ )
        sum += ( x[i]-x[i-1] ) * ( y[i]+y[i-1] );
    sum /= 2.0;
    return sum;
};
