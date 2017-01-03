#include "../include/stochastic_processes.hpp"

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

size_t stochastic::downsample_wiener_path( double *inc, size_t N, size_t R )
{
    // If no reduction factor required then return the original increments
    if( R==0 )
        return N;

    for( unsigned int i=0; i<N; i++ )
        if( i%R )
            inc[i/R] += inc[i];
        else
            inc[i/R] = inc[i];
    return N/R;
}
