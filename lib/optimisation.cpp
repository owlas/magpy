// optimisation.cpp
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#include "../include/optimisation.hpp"
#include "../include/easylogging++.h"
#include <cmath>

int optimisation::newton_raphson(
    double *x_root,
    double *x_f,
    double *x_fdash,
    double *x_tmp,
    const std::function<void(double *, const double * )> f,
    const std::function<void(double *, const double * )> fdash,
    const double *x0,
    const size_t dim,
    const double eps,
    const size_t max_iter )
{
    // Copy in the initial state
    for( unsigned int i=0; i<dim; i++ )
        x_root[i] = x0[i] + 2*eps;

    // Initialise the error (ensure at least one iteration)
    double err = 2*eps;

    // Repeat until convergence
    int iter=max_iter;
    while( ( err > eps ) && ( iter --> 0 ) )
    {
        for( unsigned int i=0; i<dim; i++ )
            x_tmp[i] = x_root[i];
        f( x_f, x_tmp );
        fdash( x_fdash, x_tmp );
        err = 0;
        for( unsigned int i=0; i<dim; i++ )
        {
            x_root[i] = x_tmp[i] - x_f[i]/x_fdash[i];
            err += std::abs( x_root[i] - x_tmp[i] );
        }
    }
    return iter==-1 ? iter : 0;
}
