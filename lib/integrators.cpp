// integrators.cpp
// Implementation of numerical schemes for integration of SDEs and
// ODEs. See header for interface information.

#include "../include/integrators.hpp"
#include "../include/optimisation.hpp"
#include <cmath>

using sde_function = std::function<void(double*,const double*,const double)>;
using sde_jac = std::function<void(double*,double*,double*,double*,const double*,const double,const double)>;
namespace ck = integrator::ck_butcher_table;

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
            + ( k1[i] + 2*k2[i] + 2*k3[i] + k4[i] ) * h/6.0;
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

/// RK45 adaptive step deterministic integrator
/**
 * Deterministic Runge-Kutta integrator step with adaptive step size
 */
void integrator::rk45(
    double *next_state,
    double *temp_state,
    double *k1,
    double *k2,
    double *k3,
    double *k4,
    double *k5,
    double *k6,
    double *h_ptr,
    double *t_ptr,
    const double *current_state,
    const sde_function derivs,
    const size_t n_dims,
    const double eps )
{
    bool step_success = false;
    double err;
    double h=*h_ptr;
    double t=*t_ptr;
    while( step_success == false )
    {
        derivs( k1, current_state, t);
        for( unsigned int i=0; i<n_dims; i++ )
            next_state[i] = k1[i] * h * ck::c11
                + current_state[i];

        derivs( k2, next_state, t + h*ck::hc1 );
        for( unsigned int i=0; i<n_dims; i++ )
            next_state[i] = current_state[i]
                + h*( ck::c21 * k1[i] + ck::c22 * k2[i] );

        derivs( k3, next_state, t + h*ck::hc2 );
        for( unsigned int i=0; i<n_dims; i++ )
            next_state[i] = current_state[i]
                + h*( ck::c31*k1[i] + ck::c32*k2[i]
                      + ck::c33*k3[i] );

        derivs( k4, next_state, t + h*ck::hc3 );
        for( unsigned int i=0; i<n_dims; i++ )
            next_state[i] = current_state[i]
                + h*( ck::c41*k1[i] + ck::c42*k2[i]
                      + ck::c43*k3[i] + ck::c44*k4[i] );

        derivs( k5, next_state, t + h*ck::hc4 );
        for( unsigned int i=0; i<n_dims; i++ )
            next_state[i] = current_state[i]
                + h*( ck::c51*k1[i] + ck::c52*k2[i]
                      + ck::c53*k3[i] + ck::c54*k4[i]
                      + ck::c55*k5[i] );

        derivs( k6, next_state, t + h*ck::hc5 );

        // Compute order 5 estimate
        for( unsigned int i=0; i<n_dims; i++ )
            temp_state[i] = current_state[i]
                + h*( ck::x11 * k1[i]
                      + ck::x13 * k3[i]
                      + ck::x14 * k4[i]
                      + ck::x16 * k6[i] );

        // Compute order 4 estimate
        for( unsigned int i=0; i<n_dims; i++)
            next_state[i] = current_state[i]
                + h*( ck::x21 * k1[i]
                      + ck::x23 * k3[i]
                      + ck::x24 * k4[i]
                      + ck::x25 * k5[i]
                      + ck::x26 * k6[i] );

        // Compute the error and scale according to (eps+eps*|state|)
        err=0;
        double mag=0;
        for( unsigned int i=0; i<n_dims; i++ )
            mag += current_state[i] * current_state[i];
        mag = std::pow( mag, 0.5 );
        for( unsigned int i=0; i<n_dims; i++ )
            err += std::pow( std::abs( temp_state[i] - next_state[i] ), 2 );
        err = pow( err, 0.5 );
        err /= ( n_dims*eps*( 1 + mag ) );

        // If relative error is below 1 then step was successful
        // otherwise reduce the step size (max 10x reduction)
        if( err < 1.0 )
            step_success = true;
        else
        {
            double hfactor = 0.84*pow( err, -0.2 );
            hfactor = std::abs( hfactor ) < 0.1 ? 0.1 : hfactor;
            h *= hfactor;
        }
    }
    // Set the new time
    *t_ptr = t+h;

    // Set the next step size
    double hfactor = err==0.0 ? 5.0 : 0.84*std::pow( err, -0.2 );
    hfactor = hfactor > 5 ? 5.0 : hfactor;
    *h_ptr = hfactor*h;
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

void driver::implicit_midpoint(
    double *x,
    const double *x0,
    const double *dw,
    const sde_jac sde,
    const size_t n_dim,
    const size_t w_dim,
    const size_t n_steps,
    const double t0,
    const double dt,
    const double eps,
    const size_t max_iter )
{
    double *dwm = new double[w_dim];
    double *a_work = new double[n_dim];
    double *b_work = new double[n_dim*w_dim];
    double *adash_work = new double[n_dim*n_dim];
    double *bdash_work = new double[n_dim*w_dim*n_dim];
    double *x_guess = new double[n_dim];
    double *x_opt_tmp = new double [n_dim];
    double *x_opt_jac = new double[n_dim*n_dim];
    lapack_int *x_opt_ipiv = new lapack_int[n_dim];

    // First step
    double t = t0;
    int err_code;
    for( unsigned int i=0; i<n_dim; i++ )
        x[i] = x0[i];

    // Take N steps
    for( unsigned int n=0; n<n_steps; n++ )
    {
        t+=dt;
        err_code = integrator::implicit_midpoint(
            x+((n+1)*n_dim), dwm, a_work, b_work, adash_work, bdash_work,
            x_guess, x_opt_tmp, x_opt_jac, x_opt_ipiv, x+(n*n_dim),
            dw+(n*n_dim), sde, n_dim, w_dim, t, dt, eps, max_iter );
        //std::cout << x[n] << " " << dw[n] << " " << x[n+1] << std::endl;
        if( err_code != 0 )
            std::cout << "At time step " << n << " implicit solver errcode:" << err_code << std::endl;
    }

    delete[] dwm;
    delete[] a_work;
    delete[] b_work;
    delete[] adash_work;
    delete[] bdash_work;
    delete[] x_guess;
    delete[] x_opt_tmp;
    delete[] x_opt_jac;
    delete[] x_opt_ipiv;
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
    const sde_jac sde,
    const size_t n_dim,
    const size_t w_dim,
    const double t,
    const double dt,
    const double eps,
    const size_t max_iter )
{
    // Implicit method depends on a modified random variable
    // See http://epubs.siam.org/doi/pdf/10.1137/S0036142901395588 for details
    double Ah = std::sqrt( 2*std::abs( std::log( dt ) ) );
    for( unsigned int i=0; i<w_dim; i++ )
        dwm[i] = std::max( -Ah, std::min( Ah, dw[i] ) ) * std::sqrt( dt );

    // The initial guess will be an Euler step
    sde( a_work, b_work, adash_work, bdash_work, x0, t, t );
    for( unsigned int i=0; i<n_dim; i++ )
    {
        x_guess[i] = a_work[i]*dt;
        for( unsigned int j=0; j<w_dim; j++ )
            x_guess[i] += b_work[i*w_dim+j]*dwm[j];
    }
    // Convert this into (x1+x2)/2
    for( unsigned int i=0; i<n_dim; i++ )
        x_guess[i] = ( x_guess[i]+x0[i] ) / 2;

    // construct F(x) and J(x) - Jacobian of F
    auto FJ = [sde, a_work,b_work,t,dt,n_dim,x0,w_dim,dwm,adash_work,bdash_work]
        (double *fout, double *jacout, const double *in)->void
        {
            // F(X)
            sde(a_work,b_work,adash_work,bdash_work,in,t+dt/2,t);
            for( unsigned int i=0; i<n_dim; i++ )
            {
                fout[i] = in[i] - 0.5*a_work[i]*dt - x0[i];
                for( unsigned int j=0; j<w_dim; j++ )
                    fout[i] -= 0.5*b_work[i*w_dim+j]*dwm[j];
            }

            // J(X)
            for( unsigned int i=0; i<n_dim; i++ )
                for( unsigned int j=0; j<n_dim; j++ )
                {
                    jacout[i*n_dim+j] = (i==j) - 0.5*adash_work[i*n_dim+j];
                    for( unsigned int k=0; k<w_dim; k++ )
                        jacout[i*n_dim+j] -= 0.5*bdash_work[i*w_dim*n_dim+k*n_dim+j]*dwm[k];
                }
        };

    // Solve the nonlinear equations to get the next step
    int err_code;
    auto flag = optimisation::newton_raphson_noinv(
        x, x_opt_tmp, x_opt_jac, x_opt_ipiv, &err_code, FJ,
        x_guess, n_dim, eps, max_iter );

    // Recover the next state from the current
    for( unsigned int i=0; i<n_dim; i++ )
        x[i] = 2*x[i] - x0[i];

    // Return error code
    return flag;
}
