#include "../include/stochastic_processes.hpp"
#ifdef USEMKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

void stochastic::strato_to_ito(
    double *ito_drift, const double *strato_drift,
    const double *diffusion, const double *diffusion_jacobian,
    const size_t n_dims, const size_t wiener_dims )
{
    for( unsigned int i=0; i<n_dims; i++ )
        for( unsigned int j=0; j<wiener_dims; j++ )
            for( unsigned int k=0; k<n_dims; k++ )
                ito_drift[i] = strato_drift[i]
                    + 0.5
                    * diffusion[k*wiener_dims+j]
                    * diffusion_jacobian[i*n_dims*wiener_dims+j*n_dims+k];
}

void stochastic::ito_to_strato(
    double *strato_drift, const double *ito_drift,
    const double *diffusion, const double *diffusion_jacobian,
    const size_t n_dims, const size_t wiener_dims )
{
    for( unsigned int i=0; i<n_dims; i++ )
        for( unsigned int j=0; j<wiener_dims; j++ )
            for( unsigned int k=0; k<n_dims; k++ )
                strato_drift[i] = ito_drift[i]
                    - 0.5
                    * diffusion[k*wiener_dims+j]
                    * diffusion_jacobian[i*n_dims*wiener_dims+j*n_dims+k];
}


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
