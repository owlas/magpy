// optimisation.cpp
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#include "../include/optimisation.hpp"
#include "../include/easylogging++.h"
#include <cmath>

int optimisation::newton_raphson_1(
    double *x_root,
    const std::function<double(const double )> f,
    const std::function<double(const double )> fdash,
    const double x0,
    const double eps,
    const size_t max_iter )
{
    // Initialise the error (ensure at least one iteration)
    double err = 2*eps;
    double x1, x2=x0;

    // Repeat until convergence
    int iter=max_iter;
    while( ( err > eps ) && ( iter --> 0 ) )
    {
        x1 = x2;
        x2 = x1 - f( x1 )/fdash( x1 );
        err = std::abs( x2 - x1 );
    }
    *x_root=x2;
    return iter==-1 ? iter : 0;
}
