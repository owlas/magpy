/** @file optimisation.hpp
 * @brief Contains numerical methods for optimisation and root-finding
 * @author Oliver W. Laslett
 * @date 2016
 */
#ifndef OPTIM_H
#define OPTIM_H
#include <functional>
#ifdef USEMKL
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif

/** @file optimisation.cpp
 * @brief Numerical methods for optimisation and root finding.
 * @details Contains Newton Raphson methods for root finding.
 */
namespace optimisation {

    /// Optimisation success return code
    const int SUCCESS = 0; //0b00

    /// Optimisation maximum iterations reached error code
    const int MAX_ITERATIONS_ERR = 1; //0b01

    /// Otimisation internal LAPACK error code
    /**
     * This error code indicates an error ocurred in an internal
     * LAPACK call. Further investigation will be needed to determine the
     * cause.
     */
    const int LAPACK_ERR = 2; //0b10

    int newton_raphson_1( double *x_root,
                          const std::function<double(const double) > f,
                          const std::function<double(const double) > fdash,
                          const double x0,
                          const double eps=1e-7,
                          const size_t max_iter=1000 );

    int newton_raphson_noinv(
        double *x_root,
        double *x_tmp,
        double *jac_out,
        lapack_int *ipiv,
        int *lapack_err_code,
        const std::function<void(double*,double*,const double* )> func_and_jacobian,
        const double *x0, const lapack_int dim,
        const double eps=1e-7,
        const size_t max_iter=1000 );
}
#endif
