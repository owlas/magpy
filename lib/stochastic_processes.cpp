#include "../include/stochastic_processes.hpp"
#ifdef USEMKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

/// Evaluates the derivatives of the master equation given a transition matrix
/**
 * The master equation is simply a linear system of ODEs, with coefficients
 * described by the transition matrix.
 * \f$\frac{\textup{d}x}{\textup{d}t}=Wx\f$
 * @param[out] derivs the master equation derivatives [length dim]
 * @param[in] transition_matrix the [dim x dim] transition matrix
 * (row-major) \f$W\f$
 * @param[in] current_state values of the state vector \f$x\f$
 */
void stochastic::master_equation(
    double *derivs, const double *transition_matrix,
    const double *current_state, const size_t dim )
{
    // Performs W*x
    cblas_dgemv( CblasRowMajor, CblasNoTrans, dim, dim, 1, transition_matrix,
                 dim, current_state, 1, 0, derivs, 1 );
}
