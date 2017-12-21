// trap.hpp
// functions for trapezoidal integration scheme
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#ifndef TRAP_H
#define TRAP_H
#include <cstdlib>

/**
 * @namespace trap
 * @brief numerical schemes for computing the area under curves
 * @author Oliver Laslett
 */
namespace trap
{
    /*
      Standard trapezoidal rule approximates the area under the points y=f(x)
      The grid can be non-uniform of length N.
     */
    double trapezoidal( double *x, double *y, size_t N);

    /*
      Just a single term of the trapezoidal summation
    */
    double one_trapezoid( double x1, double x2, double fx1, double fx2 );
}
#endif
