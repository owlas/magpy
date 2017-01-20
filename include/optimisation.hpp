// optimisation.hpp
// Contains numerical methods for optimisation and root-finding
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#ifndef OPTIM_H
#define OPTIM_H
#include <functional>
#ifdef USEMKL
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif

namespace optimisation {
    const int SUCCESS = 0b0;
    const int MAX_ITERATIONS_ERR = 0b1;
    const int LAPACK_ERR = 0b10;

    /*
      Newton-Raphson method for root finding.
      Determines x such that f(x) = 0
      Requires f(x) and f'(x) for computation.
      Convergence criteria is the abs error for each dimension of f(x)
      Returns SUCCESS on success. MAX_ITERATIONS_ERR if max_iter reached
    */
    int newton_raphson_1( double *x_root,
                          const std::function<double(const double) > f,
                          const std::function<double(const double) > fdash,
                          const double x0,
                          const double eps=1e-7,
                          const size_t max_iter=1000 );

    /*
      Newton-Raphson method finds the root F(x)=0 where F : N -> N
      Jacobian of F is denoted J where J : N -> NxN
      Newton iteration xn = x - inv(J(x))/F(x)
      This function avoids inverse by solving the linear system:
          J(x)(xn - x) = -F(x)
      Params:
        x_root is size N
         x_tmp is size N
       jac_out is size NxN
          ipiv is size N
            x0 is size N

      NB: error is calculated as the 2-norm of the residual vector (xnext-xprev)
      Returns SUCCESS or LAPACK_ERR (check lapack_err_code) or MAX_ITERATIONS_ERR

      lapack_err_code - 0=ok -i=ith val is illegal i=i is exactly 0 (singular)
    */
    int newton_raphson_noinv(
        double *x_root,
        double *x_tmp,
        double *jac_out,
        lapack_int *ipiv,
        int *lapack_err_code,
        const std::function<void(double*,const double* )> f,
        const std::function<void(double*,const double* )> jacobian,
        const double *x0, const lapack_int dim,
        const double eps=1e-7,
        const size_t max_iter=1000 );
}
#endif
