// optimisation.hpp
// Contains numerical methods for optimisation and root-finding
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#ifndef OPTIM_H
#define OPTIM_H
#include <functional>

/*
  Newton-Raphson method for root finding.
  Determines x such that f(x) = 0
  Requires f(x) and f'(x) for computation.
  Convergence criteria is the abs error for each dimension of f(x)
  Returns 0 on success. -1 if max_iter reached
 */
namespace optimisation {
    int newton_raphson( double *x_root,
                         double *x_f,
                         double *x_fdash,
                         double *x_tmp,
                         const std::function<void(double *, const double *) > f,
                         const std::function<void(double *, const double *) > fdash,
                         const double *x0,
                         const size_t dim,
                         const double eps=1e-7,
                         const size_t max_iter=1000 );
}
#endif
