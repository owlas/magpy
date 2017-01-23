// stochastic_processes.hpp
// functions for manipulation of general stochastic processes
//
// Oliver W. Laslett (2016)
// O.Laslett@soton.ac.uk
#ifndef STOC_H
#define STOC_H
#include <cstdlib>

namespace stochastic
{
// Computes the ito correction term for a stochastic process from its
// diffusion jacobian. Used to convert between Stratonovich and Ito
// drift derivatives.
    void strato_to_ito( double *ito_drift, const double *strato_drift,
                        const double *diffusion, const double *diffusion_jacobian,
                        const size_t n_dims, const size_t wiener_dims );

    void ito_to_strato( double *strato_drift, const double *ito_drift,
                        const double *diffusion, const double *diffusion_jacobian,
                        const size_t n_dims, const size_t wiener_dims );

    void master_equation(double *derivs, const double *transition_matrix,
                         const double *current_state, const size_t dim );
}

#endif
