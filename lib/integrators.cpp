// integrators.cpp
// Implementation of numerical schemes for integration of SDEs and
// ODEs. See header for interface information.

#include "../include/integrators.hpp"

using sde_function = std::function<void(double*,const double*,const double)>;

void integrator::rk4(
    double *next_state,
    double *k1,
    double *k2,
    double *k3,
    double *k4,
    const double *current_state,
    const sde_function derivs,
    const size_t n_dims,
    const double t,
    const double h )
{
    derivs( k1, current_state, t);
    for( unsigned int i=0; i<n_dims; i++ )
        next_state[i] = k1[i] * h/2.0
            + current_state[i];

    derivs( k2, next_state, t + h/2.0 );
    for( unsigned int i=0; i<n_dims; i++ )
        next_state[i] = k2[i] * h/2.0
            + current_state[i];

    derivs( k3, next_state, t + h/2.0 );
    for( unsigned int i=0; i<n_dims; i++ )
        next_state[i] = k3[i] * h
            + current_state[i];

    derivs( k4, next_state, t + h );
    for( unsigned int i=0; i<n_dims; i++ )
        next_state[i] = current_state[i]
            + ( k1[i] + k2[i] + k3[i] + k4[i] ) * h/6.0;
}

void driver::rk4(
    double *states,
    const double* initial_state,
    const sde_function derivs,
    const size_t n_steps,
    const size_t n_dims,
    const double step_size )
{
    for( unsigned int i=0; i<n_dims; i++ )
        states[i] = initial_state[i];

    double *k1 = new double[n_dims];
    double *k2 = new double[n_dims];
    double *k3 = new double[n_dims];
    double *k4 = new double[n_dims];

    for( unsigned int i=1; i<n_steps; i++ )
        integrator::rk4( states+i*n_dims, k1, k2, k3, k4,
                         states+(i-1)*n_dims, derivs, n_dims,
                         i*step_size, step_size );

    delete[] k1; delete[] k2; delete[] k3; delete[] k4;
}

void integrator::eulerm(
    double *next_state,
    double *diffusion_matrix,
    const double *current_state,
    const double *wiener_steps,
    const sde_function drift,
    const sde_function diffusion,
    const size_t n_dims,
    const size_t wiener_dims,
    const double t,
    const double step_size )
{
    drift( next_state, current_state, t );
    diffusion( diffusion_matrix, current_state, t );

    for( unsigned int i=0; i<n_dims; i++ ){
        next_state[i] = current_state[i] * step_size*next_state[i];
        for( unsigned int j=0; j<wiener_dims; j++ )
            next_state[i] = diffusion_matrix[j+i*n_dims]*wiener_steps[j];
    }
}

void driver::eulerm(
    double *states,
    const double* initial_state,
    const double *wiener_process,
    const sde_function drift,
    const sde_function diffusion,
    const size_t n_steps,
    const size_t n_dims,
    const size_t n_wiener,
    const double step_size )
{
    for( unsigned int i=0; i<n_dims; i++ )
        states[i] = initial_state[i];

    double *diffusion_mat = new double[n_dims*n_wiener];
    for( unsigned int i=1; i<n_steps; i++ )
        integrator::eulerm( states+i*n_dims, diffusion_mat, states+(i-1)*n_dims,
                            wiener_process+(i-1)*n_wiener, drift, diffusion,
                            n_dims, n_wiener, i*step_size, step_size );
    delete[] diffusion_mat;
}

void integrator::heun(
    double *next_state,
    double *drift_arr,
    double *trial_drift_arr,
    double *diffusion_matrix,
    double *trial_diffusion_matrix,
    const double *current_state,
    const double *wiener_steps,
    const sde_function drift,
    const sde_function diffusion,
    const size_t n_dims,
    const size_t wiener_dims,
    const double t,
    const double step_size )
{
    drift( drift_arr, current_state, t );
    diffusion( diffusion_matrix, current_state, t );

    for( unsigned int i=0; i<n_dims; i++ ){
        next_state[i] = current_state[i] + step_size*drift_arr[i];
        for( unsigned int j=0; j<wiener_dims; j++ )
            next_state[i] += diffusion_matrix[j+i*wiener_dims]*wiener_steps[j];
    }

    drift( trial_drift_arr, next_state, t + step_size );
    diffusion( trial_diffusion_matrix, next_state, t +step_size );

    for( unsigned int i=0; i<n_dims; i++ )
    {
        next_state[i] = current_state[i]
            + 0.5*step_size*( trial_drift_arr[i] + drift_arr[i] );
        for( unsigned int j=0; j<wiener_dims; j++ )
            next_state[i] += 0.5*wiener_steps[j]
                *( trial_diffusion_matrix[j+i*wiener_dims]
                   + diffusion_matrix[j+i*wiener_dims] );
    }
}

void driver::heun(
    double *states,
    const double* initial_state,
    const double *wiener_process,
    const sde_function drift,
    const sde_function diffusion,
    const size_t n_steps,
    const size_t n_dims,
    const size_t n_wiener,
    const double step_size )
{
    for( unsigned int i=0; i<n_dims; i++ )
        states[i] = initial_state[i];

    double *drift_arr = new double[n_dims];
    double *trial_drift_arr = new double[n_dims];
    double *diffusion_mat = new double[n_dims*n_wiener];
    double *trial_diffusion_mat = new double[n_dims*n_wiener];
    for( unsigned int i=1; i<n_steps; i++ )
        integrator::heun(
            states+i*n_dims, drift_arr, trial_drift_arr, diffusion_mat,
            trial_diffusion_mat, states+(i-1)*n_dims, wiener_process+(i-1)*n_wiener,
            drift, diffusion, n_dims, n_wiener, i*step_size, step_size );

    delete[] drift_arr; delete[] trial_drift_arr;
    delete[] diffusion_mat; delete[] trial_diffusion_mat;
}
