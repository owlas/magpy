// integrators.cpp
// Implementation of numerical schemes for integration of SDEs and
// ODEs. See header for interface information.

#include "../include/integrators.hpp"
#include "../include/optimisation.hpp"

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

// Work = NxM
int integrator::stm(
    double *x,
    double *A_res,
    double *B_res,
    double *aux,
    double *A_RHS,
    double *A_LHS,
    double *B_RHS,
    double *x_trial,
    double *x_opt_tmp,
    double *x_opt_jac,
    lapack_int *x_opt_ipiv,
    const double *x0,
    const double *dw,
    const sde_function A,
    const sde_function B,
    const sde_function Adash,
    const size_t n_dim,
    const size_t w_dim,
    const double t,
    const double dt,
    const double eps,
    const size_t max_iter
    )
{
    // Compute drift
    A( A_res, x0, t );
    // Take an auxiliary step
    B( B_res, x0, t );
    for( unsigned int i=0; i<n_dim; i++ )
    {
        aux[i] = x0[i];
        for( unsigned int j=0; j<w_dim; j++ )
            aux[i] += B_res[i*w_dim+j]*dw[j];
    }

    // Generate the right and left hand side components of the
    // implicit scheme
    auto RHS = [n_dim,w_dim,A,A_RHS,A_res,t,dt,x0,B,B_RHS,B_res,dw](double *out, const double *in)->void
        {
            A( A_RHS, in, t );
            B( B_RHS, in, t );
            for( unsigned int i=0; i<n_dim; i++ )
            {
                out[i] = in[i] - 0.5*A_RHS[i]*dt-x0[i] - 0.5*A_res[i]*dt;
                for( unsigned int j=0; j<w_dim; j++ )
                    out[i] -= 0.5*( B_RHS[i*w_dim+j] + B_res[i*w_dim+j] )*dw[j];
            }
        };
    auto LHS = [n_dim,dt,A_LHS,t,Adash](double *out, const double *in)->void
        {
            Adash( A_LHS, in, t );
            for( unsigned int i=0; i<n_dim; i++ )
                for( unsigned int j=0; j<n_dim; j++ )
                    out[i*n_dim+j] = (i==j) - 0.5*A_LHS[i*n_dim+j]*dt;
        };

    // Our initial guess will be an Euler step (aux already has most of the computation)
    for( unsigned int i=0; i<n_dim; i++ )
        x_trial[i] = aux[i] + A_res[i]*dt;

    // Determine the next step!
    lapack_int err_code;
    auto flag = optimisation::newton_raphson_noinv(
        x, x_opt_tmp, x_opt_jac, x_opt_ipiv, &err_code,
        RHS, LHS, x_trial, n_dim, eps, max_iter );
    return flag==0 ? 0 : -1;
}

int integrator::implicit_midpoint(
    double *x,
    double *dwm,
    double *a_work,
    double *b_work,
    double *adash_work,
    double *bdash_work,
    double *x_guess,
    double *x_opt_tmp,
    double *x_opt_jac,
    lapack_int *x_opt_ipiv,
    const double *x0,
    const double *dw,
    const sde_function A,
    const sde_function B,
    const sde_function Adash,
    const sde_function Bdash,
    const size_t n_dim,
    const size_t w_dim,
    const double t,
    const double dt,
    const double eps,
    const size_t max_iter )
{
    // Implicit method depends on a modified random variable
    // See http://epubs.siam.org/doi/pdf/10.1137/S0036142901395588 for details
    double Ah = std::sqrt( 2*dt*std::abs( std::log( dt ) ) );
    for( unsigned int i=0; i<w_dim; i++ )
        dwm[i] = std::max( -Ah, std::min( Ah, dw[i] ) );

    // The initial guess will be an Euler step
    A( a_work, x0, t );
    B( b_work, x0, t );
    for( unsigned int i=0; i<n_dim; i++ )
    {
        x_guess[i] = a_work[i]*dt;
        for( unsigned int j=0; j<w_dim; j++ )
            x_guess[i] += b_work[i*w_dim+j]*dwm[j];
    }
    // Convert this into (x1+x2)/2
    for( unsigned int i=0; i<n_dim; i++ )
        x_guess[i] = ( x_guess[i]+x0[i] ) / 2;

    // construct F(x)
    auto F = [A,a_work,B,b_work,t,dt,n_dim,x0,w_dim,dwm]
        (double *out, const double *in)->void
        {
            A( a_work, in, t+dt/2 );
            B( b_work, in, t );
            for( unsigned int i=0; i<n_dim; i++ )
            {
                out[i] = in[i] - 0.5*a_work[i]*dt - x0[i];
                for( unsigned int j=0; j<w_dim; j++ )
                    out[i] -= 0.5*b_work[i*w_dim+j]*dwm[j];
            }
        };

    // Construct J(x) - Jacobian of F
    auto J = [Adash,adash_work,t,dt,Bdash,bdash_work,n_dim,w_dim,dwm]
        (double *out, const double *in)->void
        {
            Adash( adash_work, in, t+dt/2 );
            Bdash( bdash_work, in, t+dt/2 );
            for( unsigned int i=0; i<n_dim; i++ )
                for( unsigned int j=0; j<n_dim; j++ )
                {
                    out[i*n_dim+j] = (i==j) - 0.5*adash_work[i*n_dim+j];
                    for( unsigned int k=0; k<w_dim; k++ )
                        out[i*n_dim+j] -= 0.5*bdash_work[i*w_dim*n_dim+k*n_dim+j]*dwm[k];
                }
        };

    // Solve the nonlinear equations to get the next step
    lapack_int err_code;
    auto flag = optimisation::newton_raphson_noinv(
        x, x_opt_tmp, x_opt_jac, x_opt_ipiv, &err_code, F, J,
        x_guess, n_dim, eps, max_iter );

    // Recover the next state from the current
    for( unsigned int i=0; i<n_dim; i++ )
        x[i] = 2*x[i] - x0[i];

    // Return error code
    return flag;
}
