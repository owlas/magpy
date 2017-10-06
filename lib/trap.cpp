/** @file trap.cpp
 * @brief Implementation of trapezoidal integration scheme
 * @author Oliver W. Laslett (2016)
 */
#include "../include/trap.hpp"

/// Trapezoidal integration between irregularly spaced points
/**
 * Estimates the definite integral \f$\int_a^b f(x) dx\f$ given values of
 * \p x and corresponding values of \p y \f$=f(x)\p
 * @param[in] x ordered irregularly spaced values between a and b
 * @param[in] y value of the function \f$f(x)\f$ for each point x
 * @param[in] N length of \p x array
 * @returns the trapezoidal sum
 */
double trap::trapezoidal( double *x, double *y, size_t N )
{
    double sum;
    unsigned int i;
    for( sum=0.0, i=1; i<N; i++ )
        sum += ( x[i]-x[i-1] ) * ( y[i]+y[i-1] );
    sum /= 2.0;
    return sum;
}
